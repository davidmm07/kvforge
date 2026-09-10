"""N-gram speculative decoding: tokens per forward pass.

Decode is the latency-bound half of serving: one forward pass per token per
sequence. Speculation breaks that 1:1 relationship by verifying ``k`` drafted
tokens in a single pass, so the metric to watch is **generated tokens per
forward pass**. All runs below use the same batch of 8 sequences, so the
baseline is 8 tokens per pass and the reduction in *forward passes* is the
speculation gain.

The workload is deliberately repetitive so that the n-gram proposer fires - that
is the regime it exists for (RAG, code editing, agent loops that echo state).
On non-repetitive traffic the proposer finds nothing, acceptance falls to zero,
and the correct outcome is that nothing changes except a little wasted work. The
sweep over ``k`` shows both sides of that trade.
"""

from __future__ import annotations

import time

from benchmarks.common import build_config, repetitive_workload, save, table
from kvforge import LLMEngine, SamplingParams, SpeculativeConfig

NUM_REQUESTS = 8
MAX_TOKENS = 48


def run(num_speculative_tokens: int, prompts) -> dict:
    spec = (
        SpeculativeConfig(num_speculative_tokens=num_speculative_tokens)
        if num_speculative_tokens > 0
        else None
    )
    engine = LLMEngine(build_config(speculative=spec, max_num_seqs=8, prefix_caching=False))
    start = time.perf_counter()
    outputs = engine.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True))
    elapsed = time.perf_counter() - start
    stats = engine.stats
    return {
        "k": num_speculative_tokens,
        "forward_passes": stats.steps,
        "generated_tokens": stats.generated_tokens,
        "tokens_per_forward": round(stats.tokens_per_forward, 3),
        "draft_tokens": stats.draft_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "acceptance_rate": round(stats.acceptance_rate, 3),
        "wall_s": round(elapsed, 3),
        "tokens": [o.output_token_ids for o in outputs],
    }


def main() -> dict:
    prompts = repetitive_workload(NUM_REQUESTS, seed=11)
    runs = [run(k, prompts) for k in (0, 1, 2, 4, 6)]

    # Losslessness is the whole point: every k must emit the same token stream.
    baseline_tokens = runs[0].pop("tokens")
    for r in runs[1:]:
        assert r.pop("tokens") == baseline_tokens, f"k={r['k']} changed the output"

    base, best = runs[0], max(runs[1:], key=lambda r: r["tokens_per_forward"])
    summary = {
        "workload": f"{NUM_REQUESTS} repetitive prompts, {MAX_TOKENS} output tokens each",
        "lossless": True,
        "runs": runs,
        "best_k": best["k"],
        "forward_pass_reduction": round(1 - best["forward_passes"] / base["forward_passes"], 4),
        "tokens_per_forward_gain": round(best["tokens_per_forward"] / base["tokens_per_forward"], 2),
        "wall_speedup": round(base["wall_s"] / best["wall_s"], 2),
    }
    print(
        table(
            runs,
            [
                ("k", "draft tokens (k)"),
                ("forward_passes", "forward passes"),
                ("tokens_per_forward", "tokens / forward (batch of 8)"),
                ("acceptance_rate", "acceptance rate"),
                ("wall_s", "wall (s)"),
            ],
        )
    )
    print(
        f"\noutput identical for every k (lossless). "
        f"best k={best['k']}: {summary['forward_pass_reduction']:.1%} fewer forward passes, "
        f"{summary['wall_speedup']}x wall"
    )
    save("spec_decode", summary)
    return summary


if __name__ == "__main__":
    main()
