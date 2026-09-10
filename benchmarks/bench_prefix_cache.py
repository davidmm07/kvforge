"""How much prefill work automatic prefix caching removes.

Workload: a served chat endpoint. Four system prompts of 256 tokens each, reused
across 32 requests with short distinct user turns. This is the common case in
production - agents, RAG, few-shot templates and chat all resend a long stable
prefix - and it is where prefix caching pays for itself.

The metric that matters is *prompt tokens actually computed*. Wall clock follows
it, but the token count is the hardware-independent statement.
"""

from __future__ import annotations

import time

from benchmarks.common import build_config, chat_workload, save, table
from kvforge import LLMEngine, SamplingParams

NUM_REQUESTS = 32
SYSTEM_LEN = 256
MAX_TOKENS = 24


def run(prefix_caching: bool) -> dict:
    # A small max_num_seqs makes requests arrive in waves, so later ones can hit
    # blocks the earlier ones published. Admitting all 32 at once would give
    # every request a cold cache no matter the policy.
    config = build_config(prefix_caching=prefix_caching, max_num_seqs=4, num_blocks=1024)
    engine = LLMEngine(config)
    prompts = chat_workload(NUM_REQUESTS, system_prompt_len=SYSTEM_LEN, seed=42)

    start = time.perf_counter()
    outputs = engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True))
    elapsed = time.perf_counter() - start

    prompt_tokens = sum(len(p) for p in prompts)
    cached = sum(o.num_cached_tokens for o in outputs)
    ttfts = sorted(o.ttft for o in outputs)
    return {
        "prefix_caching": prefix_caching,
        "prompt_tokens": prompt_tokens,
        "prefill_tokens_computed": engine.stats.prefill_tokens,
        "cached_tokens": cached,
        "token_hit_rate": round(cached / prompt_tokens, 4),
        "requests_with_a_hit": sum(1 for o in outputs if o.num_cached_tokens > 0),
        "forward_passes": engine.stats.steps,
        "wall_s": round(elapsed, 3),
        "mean_ttft_ms": round(1000 * sum(ttfts) / len(ttfts), 1),
        "p90_ttft_ms": round(1000 * ttfts[int(0.9 * len(ttfts))], 1),
        "output_tokens": engine.stats.generated_tokens,
    }


def main() -> dict:
    off = run(prefix_caching=False)
    on = run(prefix_caching=True)
    assert off["output_tokens"] == on["output_tokens"]

    saved = off["prefill_tokens_computed"] - on["prefill_tokens_computed"]
    summary = {
        "workload": f"{NUM_REQUESTS} requests, {SYSTEM_LEN}-token system prompt x4, {MAX_TOKENS} output tokens",
        "runs": [off, on],
        "prefill_tokens_saved": saved,
        "prefill_reduction": round(saved / off["prefill_tokens_computed"], 4),
        "ttft_speedup": round(off["mean_ttft_ms"] / on["mean_ttft_ms"], 2),
        "wall_speedup": round(off["wall_s"] / on["wall_s"], 2),
    }
    print(
        table(
            [off, on],
            [
                ("prefix_caching", "prefix caching"),
                ("prefill_tokens_computed", "prefill tokens computed"),
                ("token_hit_rate", "prompt token hit rate"),
                ("forward_passes", "forward passes"),
                ("mean_ttft_ms", "mean TTFT (ms)"),
                ("p90_ttft_ms", "p90 TTFT (ms)"),
                ("wall_s", "wall (s)"),
            ],
        )
    )
    print(
        f"\nprefill tokens avoided: {saved} "
        f"({summary['prefill_reduction']:.1%}), TTFT {summary['ttft_speedup']}x, "
        f"wall {summary['wall_speedup']}x"
    )
    save("prefix_cache", summary)
    return summary


if __name__ == "__main__":
    main()
