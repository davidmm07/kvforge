"""Continuous batching versus static batching.

Static batching runs a fixed group of sequences until *every one* of them
finishes, then starts the next group. When output lengths vary - which they
always do - the short sequences sit in the batch doing nothing while the longest
one finishes, and no new request can start.

Continuous batching admits a waiting request the moment a slot frees. Both modes
here use the same engine, the same batch width and the same workload; only the
admission policy differs.
"""

from __future__ import annotations

import random
import time

from benchmarks.common import build_config, random_prompt, save, table
from kvforge import LLMEngine, SamplingParams

NUM_REQUESTS = 32
BATCH_WIDTH = 8


def make_workload(seed: int = 3):
    """Prompts of similar size but wildly different output lengths."""
    rng = random.Random(seed)
    prompts, params = [], []
    for _ in range(NUM_REQUESTS):
        prompts.append(random_prompt(rng, rng.randrange(48, 96)))
        # Heavy-tailed output lengths: the case static batching handles worst.
        params.append(SamplingParams(max_tokens=rng.choice([4, 6, 8, 12, 48]), ignore_eos=True))
    return prompts, params


def run_continuous(prompts, params) -> dict:
    engine = LLMEngine(build_config(max_num_seqs=BATCH_WIDTH, prefix_caching=False))
    start = time.perf_counter()
    for prompt, sp in zip(prompts, params):
        engine.add_request(prompt, sp)
    engine.run_to_completion()
    return summarise("continuous", engine, time.perf_counter() - start)


def run_static(prompts, params) -> dict:
    """Emulate static batching: one wave at a time, drained before the next."""
    engine = LLMEngine(build_config(max_num_seqs=BATCH_WIDTH, prefix_caching=False))
    start = time.perf_counter()
    for i in range(0, len(prompts), BATCH_WIDTH):
        for prompt, sp in zip(prompts[i : i + BATCH_WIDTH], params[i : i + BATCH_WIDTH]):
            engine.add_request(prompt, sp)
        engine.run_to_completion()
    return summarise("static", engine, time.perf_counter() - start)


def summarise(mode: str, engine: LLMEngine, elapsed: float) -> dict:
    stats = engine.stats
    return {
        "mode": mode,
        "forward_passes": stats.steps,
        "generated_tokens": stats.generated_tokens,
        "mean_batch_size": round(stats.mean_batch_size, 2),
        "wall_s": round(elapsed, 3),
        "output_tok_per_s": round(stats.generated_tokens / elapsed, 1),
    }


def main() -> dict:
    prompts, params = make_workload()
    static = run_static(prompts, params)
    continuous = run_continuous(prompts, params)
    assert static["generated_tokens"] == continuous["generated_tokens"]

    summary = {
        "workload": f"{NUM_REQUESTS} requests, batch width {BATCH_WIDTH}, output lengths 4-48",
        "runs": [static, continuous],
        "throughput_speedup": round(
            continuous["output_tok_per_s"] / static["output_tok_per_s"], 2
        ),
        "forward_pass_reduction": round(
            1 - continuous["forward_passes"] / static["forward_passes"], 4
        ),
        "batch_occupancy_gain": round(
            continuous["mean_batch_size"] / static["mean_batch_size"], 2
        ),
    }
    print(
        table(
            [static, continuous],
            [
                ("mode", "mode"),
                ("forward_passes", "forward passes"),
                ("mean_batch_size", "mean batch size"),
                ("generated_tokens", "output tokens"),
                ("wall_s", "wall (s)"),
                ("output_tok_per_s", "output tok/s"),
            ],
        )
    )
    print(
        f"\n{summary['throughput_speedup']}x throughput, "
        f"{summary['forward_pass_reduction']:.1%} fewer forward passes, "
        f"{summary['batch_occupancy_gain']}x batch occupancy"
    )
    save("continuous_batching", summary)
    return summary


if __name__ == "__main__":
    main()
