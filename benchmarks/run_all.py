"""Run every benchmark and regenerate ``docs/BENCHMARKS.md``.

    python -m benchmarks.run_all

Each benchmark prints a markdown table; this script captures them, wraps them in
the interpretation that belongs with the numbers, and writes the document. The
raw numbers also land in ``results/*.json``.
"""

from __future__ import annotations

import contextlib
import io
import platform
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchmarks import (  # noqa: E402
    bench_continuous_batching,
    bench_hybrid_memory,
    bench_moe,
    bench_prefix_cache,
    bench_spec_decode,
)

DOC = Path(__file__).resolve().parents[1] / "docs" / "BENCHMARKS.md"

SECTIONS = [
    (
        "Automatic prefix caching",
        bench_prefix_cache,
        """A served chat endpoint: four 256-token system prompts reused across 32
requests with short distinct user turns.

The number to read is **prefill tokens computed**. Prefix caching does not make
attention faster; it removes prefill work that has already been done. Hit rate
here is bounded by block granularity and by the last-block hold-back (a request
must always have at least one token to run a forward pass on), so ~76% of a
~90%-shared prompt is close to the ceiling for this workload.

TTFT improves by roughly the same factor as the prefill work removed, which is
the expected relationship: for a request whose prompt is mostly cached, TTFT is
dominated by the prefill it still has to do.""",
    ),
    (
        "Continuous batching vs static batching",
        bench_continuous_batching,
        """Same engine, same batch width, same workload; only the admission policy
differs. Static batching drains a wave of 8 before starting the next, so short
sequences idle in the batch while the longest finishes.

**Forward passes** and **mean batch size** are the policy-level results and they
are hardware independent: 35% fewer passes at 1.5x the occupancy. Wall-clock
gain is much smaller here, and the reason is worth stating plainly - on CPU with
a reference attention kernel, decode is compute-bound, so a batch twice as wide
costs nearly twice as much and the saved passes do not convert into saved time.
On a GPU, decode at these batch sizes is memory-bandwidth-bound: the weights are
read once per pass regardless of batch size, so a wider batch is close to free
and the forward-pass reduction converts almost fully into throughput. The policy
result is the transferable one; the wall clock is an artefact of the kernel.""",
    ),
    (
        "N-gram speculative decoding",
        bench_spec_decode,
        """Repetitive prompts (standing in for RAG / code-edit / agent traffic),
batch of 8, sweeping the number of draft tokens `k`.

Every value of `k` produces a **byte-identical token stream**, asserted in the
benchmark itself - that is the property rejection sampling buys, and the reason
speculation is safe to enable by default rather than being a quality trade.

Acceptance rate falls as `k` grows, which is the expected shape: later draft
positions are conditioned on earlier ones being right. The absolute acceptance
numbers here are pessimistic because the benchmark model has random weights, so
its continuations do not actually echo the prompt the way a trained model's do;
what the sweep demonstrates is the mechanism and its cost curve, not a quality
figure for a real checkpoint.""",
    ),
    (
        "MoE dispatch",
        bench_moe,
        """Grouped (sort-by-expert) dispatch against dense evaluation of every expert
on every token.

The FLOP ratio is `num_experts / top_k`, and measured speedup tracks it without
reaching it: sorting, gathering and scattering are real costs, and many small
per-expert GEMMs use the hardware worse than one large one. That gap between
achieved and theoretical is exactly what fused grouped-GEMM kernels close.""",
    ),
    (
        "Hybrid attention KV footprint",
        bench_hybrid_memory,
        """One sequence grown to 32k tokens through the real allocator, measuring
physically live blocks per KV cache group.

Sliding-window layers recycle blocks the moment they leave the window, so their
footprint plateaus at `window / block_size + 1` blocks and never grows again.
Global layers grow linearly forever. A `hybrid:4` stack pays the linear cost on
one layer in four, and the saving converges toward 75% as context grows.

KV footprint is what caps concurrency: at 32k context this is the difference
between holding one sequence per GB and holding three and a half.""",
    ),
]

HEADER = """# Benchmarks

Every number here is produced by `python -m benchmarks.run_all`, which also
writes `results/*.json`. Re-running regenerates this file.

**Read these as runtime results, not kernel results.** kvforge's attention is a
readable reference implementation in plain PyTorch, so absolute throughput is not
the point and is not competitive with a real engine. What the benchmarks measure
is work *avoided* by the runtime - prefill tokens not recomputed, forward passes
not run, expert FLOPs not spent, KV bytes not held - and those quantities carry
over to any kernel underneath.
"""


def main() -> None:
    parts = [HEADER, f"\n## Environment\n"]
    parts.append(
        f"- {platform.python_implementation()} {platform.python_version()}, "
        f"torch {torch.__version__}, device `cpu`\n"
        f"- {platform.processor() or platform.machine()}, {platform.system()} {platform.release()}\n"
        f"- generated {date.today().isoformat()}\n"
    )

    for title, module, commentary in SECTIONS:
        print(f"==> {title}", flush=True)
        start = time.perf_counter()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            module.main()
        elapsed = time.perf_counter() - start
        print(f"    {elapsed:.1f}s", flush=True)

        parts.append(f"\n## {title}\n")
        parts.append(f"\n{commentary}\n")
        parts.append(f"\n```\n{buffer.getvalue().strip()}\n```\n")

    parts.append(
        "\n## Reproducing\n\n"
        "```bash\n"
        "pip install -e '.[dev]'\n"
        "python -m benchmarks.run_all\n"
        "```\n"
    )
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text("".join(parts), encoding="utf-8")
    print(f"\nwrote {DOC}")


if __name__ == "__main__":
    main()
