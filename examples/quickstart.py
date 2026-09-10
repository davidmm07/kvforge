"""A guided tour of the runtime.

    python examples/quickstart.py

Each section prints what the engine actually did - blocks allocated, prompt
tokens skipped, forward passes saved - rather than just the generated tokens,
which are meaningless for a random-weight model.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvforge import (
    CacheConfig,
    EngineConfig,
    LLMEngine,
    ModelConfig,
    MoEConfig,
    MultiModalInput,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)

RNG = random.Random(0)


def banner(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def config(**kw) -> EngineConfig:
    model_kw = dict(
        hidden_size=256, num_layers=4, num_heads=8, num_kv_heads=2,
        ffn_hidden_size=512, vocab_size=1024, seed=0,
    ) | kw.pop("model", {})
    cache_kw = dict(block_size=16, num_blocks=512) | kw.pop("cache", {})
    sched_kw = dict(
        max_num_seqs=8, max_num_batched_tokens=1024, max_model_len=2048
    ) | kw.pop("scheduler", {})
    return EngineConfig(
        model=ModelConfig(**model_kw),
        cache=CacheConfig(**cache_kw),
        scheduler=SchedulerConfig(**sched_kw),
        **kw,
    )


def prompt(n: int) -> list[int]:
    return [RNG.randrange(1024) for _ in range(n)]


# ---------------------------------------------------------------- 1. basics


def basic_generation() -> None:
    banner("1. Offline batch generation")
    engine = LLMEngine(config())
    prompts = [prompt(40), prompt(64), prompt(24)]
    outputs = engine.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True))
    for out in outputs:
        print(
            f"  {out.request_id}: prompt={len(out.prompt_token_ids):3d} "
            f"output={len(out.output_token_ids):3d} reason={out.finish_reason} "
            f"ttft={1000 * out.ttft:.0f}ms"
        )
    print(f"\n  {engine.stats.steps} forward passes for {engine.stats.generated_tokens} tokens")
    print(f"  KV cache: {engine.runner.kv_cache_bytes() / 2**20:.0f} MiB across {engine.config.cache.num_blocks} blocks")


# --------------------------------------------------------- 2. prefix caching


def prefix_caching() -> None:
    banner("2. Automatic prefix caching")
    engine = LLMEngine(config())
    system = prompt(192)
    params = SamplingParams(max_tokens=8, ignore_eos=True)

    print("  request        prompt   cached   computed")
    for i in range(4):
        (out,) = engine.generate([system + prompt(8)], params)
        print(
            f"  {out.request_id:<14} {len(out.prompt_token_ids):>6} "
            f"{out.num_cached_tokens:>8} {len(out.prompt_token_ids) - out.num_cached_tokens:>10}"
        )
    stats = engine.kv.pool.stats
    print(f"\n  block cache: {stats.hits}/{stats.queries} requests hit, {stats.evictions} evictions")
    print("  the first request populates the cache; the rest reuse its blocks")


# ------------------------------------------------------------ 3. multimodal


def multimodal_keys() -> None:
    banner("3. Multimodal prefix-cache keys")
    engine = LLMEngine(config())
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    # Identical placeholder token ids, different image content.
    placeholders = [7] * 64
    tail = prompt(8)

    for mm_hash in ("sha256:cat", "sha256:cat", "sha256:dog"):
        rid = engine.add_request(
            placeholders + tail,
            params,
            multi_modal_inputs=[MultiModalInput(mm_hash, 0, 64)],
        )
        (out,) = [o for o in engine.run_to_completion() if o.request_id == rid]
        print(f"  image={mm_hash:<14} cached_tokens={out.num_cached_tokens}")
    print("\n  the repeat of 'cat' hits; 'dog' misses despite identical token ids")


# --------------------------------------------------------- 4. hybrid memory


def hybrid_attention() -> None:
    banner("4. Hybrid attention: two KV cache groups")
    engine = LLMEngine(
        config(model={"num_layers": 4, "attention_pattern": "hybrid:2", "sliding_window": 64})
    )
    for i, group in enumerate(engine.kv.groups):
        window = f"window={group.window}" if group.window else "global"
        print(f"  group {i}: {group.kind:<8} layers={list(group.layer_ids)} {window}")

    engine.generate([prompt(96)], SamplingParams(max_tokens=128, ignore_eos=True))
    used = engine.stats.peak_block_utilisation * (engine.config.cache.num_blocks - 1)
    print(f"\n  peak blocks live: {used:.0f} for a 224-token sequence")
    print("  sliding layers recycled their out-of-window blocks as it grew")


# ---------------------------------------------------------- 5. speculation


def speculative_decoding() -> None:
    banner("5. N-gram speculative decoding")
    repetitive = ([3, 1, 4, 1, 5, 9, 2, 6] * 20)[:160]
    params = SamplingParams(max_tokens=32, ignore_eos=True)

    plain = LLMEngine(config())
    base = plain.generate([repetitive], params)[0]

    engine = LLMEngine(config(speculative=SpeculativeConfig(num_speculative_tokens=4)))
    spec = engine.generate([repetitive], params)[0]

    print(f"  without speculation: {plain.stats.steps:3d} forward passes")
    print(f"  with speculation:    {engine.stats.steps:3d} forward passes")
    print(f"  drafts {spec.num_draft_tokens} proposed, {spec.num_accepted_tokens} accepted")
    print(f"  identical output: {base.output_token_ids == spec.output_token_ids}")


# ------------------------------------------------------------------ 6. MoE


def mixture_of_experts() -> None:
    banner("6. Mixture of experts")
    engine = LLMEngine(config(model={"moe": MoEConfig(num_experts=8, top_k=2)}))
    engine.generate([prompt(48)], SamplingParams(max_tokens=8, ignore_eos=True))
    print("  8 experts, top-2 routing, grouped dispatch")
    print(f"  {engine.stats.steps} forward passes, output verified against dense")
    print("  see tests/test_moe_and_hybrid.py for the equivalence proof")


# ---------------------------------------------------- 7. memory pressure


def memory_pressure() -> None:
    banner("7. Preemption under memory pressure")
    prompts = [prompt(RNG.randrange(60, 100)) for _ in range(6)]
    params = SamplingParams(max_tokens=24, ignore_eos=True)

    roomy = LLMEngine(config())
    expected = roomy.generate(prompts, params)

    tight = LLMEngine(
        config(cache={"num_blocks": 40, "block_size": 16, "enable_prefix_caching": False})
    )
    got = tight.generate(prompts, params)

    same = all(a.output_token_ids == b.output_token_ids for a, b in zip(expected, got))
    print(f"  512 blocks: {roomy.scheduler.num_preemptions} preemptions")
    print(f"   40 blocks: {tight.scheduler.num_preemptions} preemptions")
    print(f"  outputs identical: {same}")
    print("  a full cache costs throughput, never correctness")


if __name__ == "__main__":
    basic_generation()
    prefix_caching()
    multimodal_keys()
    hybrid_attention()
    speculative_decoding()
    mixture_of_experts()
    memory_pressure()
    print()
