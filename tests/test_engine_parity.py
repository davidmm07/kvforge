"""End-to-end parity: none of the runtime optimisations may change the output.

Each test runs the same prompts twice - once through a configuration that
exercises an optimisation, once through a baseline - and demands identical
tokens. The strongest of them compares against ``dense_generate``, which shares
only the weights and recomputes everything from scratch with no cache at all.
"""

import random

import pytest

from kvforge import (
    CacheConfig,
    EngineConfig,
    LLMEngine,
    ModelConfig,
    MoEConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from kvforge.models.reference import dense_generate


def make_config(**overrides) -> EngineConfig:
    model_kw = dict(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_hidden_size=128,
        vocab_size=128,
        seed=1234,
    ) | overrides.pop("model", {})
    cache_kw = dict(block_size=8, num_blocks=256) | overrides.pop("cache", {})
    sched_kw = dict(
        max_num_seqs=8, max_num_batched_tokens=512, max_model_len=512
    ) | overrides.pop("scheduler", {})
    return EngineConfig(
        model=ModelConfig(**model_kw),
        cache=CacheConfig(**cache_kw),
        scheduler=SchedulerConfig(**sched_kw),
        **overrides,
    )


def make_prompts(n, lo=12, hi=60, seed=0):
    rng = random.Random(seed)
    return [
        [rng.randrange(128) for _ in range(rng.randrange(lo, hi))] for _ in range(n)
    ]


GREEDY = SamplingParams(max_tokens=12, ignore_eos=True)


def test_engine_matches_cacheless_dense_reference():
    """The headline claim: paging + batching == textbook recompute."""
    engine = LLMEngine(make_config())
    prompts = make_prompts(3, seed=1)
    outputs = engine.generate(prompts, GREEDY)
    for prompt, out in zip(prompts, outputs):
        expected = dense_generate(engine.runner.model, prompt, GREEDY.max_tokens)
        assert out.output_token_ids == expected


def test_batched_matches_one_at_a_time():
    """Batching a request with others must not change its tokens."""
    prompts = make_prompts(6, seed=2)
    batched = LLMEngine(make_config()).generate(prompts, GREEDY)

    solo = []
    for prompt in prompts:
        engine = LLMEngine(make_config(scheduler={"max_num_seqs": 1}))
        solo.append(engine.generate([prompt], GREEDY)[0])

    for a, b in zip(batched, solo):
        assert a.output_token_ids == b.output_token_ids


def test_chunked_prefill_matches_whole_prompt_prefill():
    prompts = make_prompts(4, lo=60, hi=120, seed=3)
    whole = LLMEngine(
        make_config(scheduler={"max_num_batched_tokens": 1024, "enable_chunked_prefill": False})
    ).generate(prompts, GREEDY)
    # A budget far below the prompt length forces prompts to be split across
    # several steps and interleaved with other sequences' decodes.
    chunked = LLMEngine(
        make_config(scheduler={"max_num_batched_tokens": 24})
    ).generate(prompts, GREEDY)
    for a, b in zip(whole, chunked):
        assert a.output_token_ids == b.output_token_ids


def test_prefix_caching_does_not_change_output():
    shared = [7] * 40  # long enough to fill several blocks
    prompts = [shared + [i, i + 1, i + 2] for i in range(4)]
    off = LLMEngine(make_config(cache={"enable_prefix_caching": False})).generate(prompts, GREEDY)

    # Requests must arrive in separate waves for the cache to help: a block is
    # only published after the forward pass that filled it, so requests admitted
    # in the same step all miss.
    engine_on = LLMEngine(make_config(cache={"enable_prefix_caching": True}))
    on = [engine_on.generate([p], GREEDY)[0] for p in prompts]

    for a, b in zip(off, on):
        assert a.output_token_ids == b.output_token_ids
    # The three later requests each reused the shared 40-token prefix.
    assert sum(o.num_cached_tokens for o in on) > 0
    assert engine_on.kv.pool.stats.hits == 3
    assert all(o.num_cached_tokens >= 32 for o in on[1:])


def test_prefix_cache_hit_requires_an_identical_prefix():
    """A different token anywhere in a block must break the chain."""
    engine = LLMEngine(make_config(cache={"enable_prefix_caching": True}))
    base = [3] * 40
    engine.generate([base + [1, 2]], GREEDY)
    # Same length, differs in the very first token: no block may be shared.
    (out,) = engine.generate([[4] + [3] * 39 + [1, 2]], GREEDY)
    assert out.num_cached_tokens == 0


def test_preemption_under_memory_pressure_preserves_output():
    """A cache too small for the working set must degrade, not corrupt."""
    prompts = make_prompts(6, lo=40, hi=70, seed=4)
    roomy = LLMEngine(make_config()).generate(prompts, GREEDY)

    tight = LLMEngine(
        make_config(
            cache={"num_blocks": 24, "enable_prefix_caching": False},
            scheduler={"max_num_seqs": 6, "max_num_batched_tokens": 512},
        )
    )
    pressured = tight.generate(prompts, GREEDY)

    assert tight.scheduler.num_preemptions > 0, "test did not actually create pressure"
    for a, b in zip(roomy, pressured):
        assert a.output_token_ids == b.output_token_ids


def test_sliding_window_model_matches_dense_windowed_reference():
    config = make_config(model={"attention_pattern": "sliding", "sliding_window": 24})
    engine = LLMEngine(config)
    prompts = make_prompts(2, lo=70, hi=90, seed=5)
    outputs = engine.generate(prompts, GREEDY)
    for prompt, out in zip(prompts, outputs):
        expected = dense_generate(engine.runner.model, prompt, GREEDY.max_tokens)
        assert out.output_token_ids == expected


def test_hybrid_model_matches_dense_reference():
    """Global and sliding-window layers in one model, two KV cache groups."""
    config = make_config(
        model={"num_layers": 4, "attention_pattern": "hybrid:2", "sliding_window": 16}
    )
    engine = LLMEngine(config)
    assert engine.kv.num_groups == 2
    prompts = make_prompts(2, lo=60, hi=80, seed=6)
    outputs = engine.generate(prompts, GREEDY)
    for prompt, out in zip(prompts, outputs):
        expected = dense_generate(engine.runner.model, prompt, GREEDY.max_tokens)
        assert out.output_token_ids == expected


def test_moe_model_matches_dense_reference():
    config = make_config(model={"moe": MoEConfig(num_experts=4, top_k=2)})
    engine = LLMEngine(config)
    prompts = make_prompts(2, seed=7)
    outputs = engine.generate(prompts, GREEDY)
    for prompt, out in zip(prompts, outputs):
        expected = dense_generate(engine.runner.model, prompt, GREEDY.max_tokens)
        assert out.output_token_ids == expected


def test_speculative_decoding_is_lossless_under_greedy():
    """n-gram speculation must reproduce the non-speculative token stream."""
    # Repetitive prompts so the n-gram proposer actually fires.
    prompts = [[1, 2, 3, 4, 5] * 8, [9, 8, 7] * 12]
    baseline = LLMEngine(make_config()).generate(prompts, GREEDY)
    engine = LLMEngine(
        make_config(speculative=SpeculativeConfig(num_speculative_tokens=3))
    )
    spec = engine.generate(prompts, GREEDY)

    for a, b in zip(baseline, spec):
        assert a.output_token_ids == b.output_token_ids
    assert engine.stats.draft_tokens > 0, "proposer never fired"


@pytest.mark.parametrize("block_size", [1, 4, 16, 32])
def test_block_size_does_not_change_output(block_size):
    prompts = make_prompts(3, seed=8)
    ref = LLMEngine(make_config(cache={"block_size": 8, "num_blocks": 256})).generate(prompts, GREEDY)
    got = LLMEngine(
        make_config(cache={"block_size": block_size, "num_blocks": 512})
    ).generate(prompts, GREEDY)
    for a, b in zip(ref, got):
        assert a.output_token_ids == b.output_token_ids
