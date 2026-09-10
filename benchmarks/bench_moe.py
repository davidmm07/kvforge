"""Grouped MoE dispatch versus dense evaluation.

Dense evaluation runs every expert on every token and masks. Grouped dispatch
sorts the (token, expert) pairs so each expert sees only its own rows. The
expected ratio is ``num_experts / top_k`` in FLOPs; the measured ratio is lower
because sorting, gathering and scattering are not free, and because small
per-expert GEMMs use the hardware worse than one big one.

That gap is the whole reason fused grouped-GEMM kernels exist. Measuring it here
makes the size of the prize explicit.
"""

from __future__ import annotations

import time

import torch

from benchmarks.common import save, table
from kvforge.layers.moe import SparseMoEBlock

HIDDEN, FFN = 512, 1024
CONFIGS = [
    # (num_experts, top_k, num_tokens)
    (8, 2, 1),  # single-token decode
    (8, 2, 64),
    (8, 2, 512),  # prefill chunk
    (32, 4, 512),  # many small experts
    (64, 8, 512),
]
REPEATS = 20


def bench(fn, x, repeats=REPEATS) -> float:
    fn(x)  # warm up allocator and any lazy init
    start = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    return (time.perf_counter() - start) / repeats


def main() -> dict:
    torch.manual_seed(0)
    rows = []
    for num_experts, top_k, num_tokens in CONFIGS:
        moe = SparseMoEBlock(HIDDEN, FFN, num_experts, top_k)
        x = torch.randn(num_tokens, HIDDEN)
        with torch.inference_mode():
            dense_ms = 1000 * bench(lambda t: moe(t, mode="dense"), x)
            grouped_ms = 1000 * bench(lambda t: moe(t, mode="grouped"), x)
            torch.testing.assert_close(
                moe(x, mode="grouped"), moe(x, mode="dense"), atol=1e-4, rtol=1e-4
            )
        rows.append(
            {
                "experts": num_experts,
                "top_k": top_k,
                "tokens": num_tokens,
                "dense_ms": round(dense_ms, 3),
                "grouped_ms": round(grouped_ms, 3),
                "speedup": round(dense_ms / grouped_ms, 2),
                "flop_ratio": round(num_experts / top_k, 1),
            }
        )

    print(
        table(
            rows,
            [
                ("experts", "experts"),
                ("top_k", "top-k"),
                ("tokens", "tokens"),
                ("dense_ms", "dense (ms)"),
                ("grouped_ms", "grouped (ms)"),
                ("speedup", "speedup"),
                ("flop_ratio", "FLOP ratio"),
            ],
        )
    )
    summary = {"hidden": HIDDEN, "ffn": FFN, "repeats": REPEATS, "rows": rows}
    save("moe_dispatch", summary)
    return summary


if __name__ == "__main__":
    main()
