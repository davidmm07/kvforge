"""KV memory for global, sliding-window and hybrid attention stacks.

This drives the real ``KVCacheManager`` - the same allocate/recycle path the
engine uses - for one sequence growing to 32k tokens, and records the physical
blocks live at each length. No model runs, because none is needed: this measures
the allocator, and the allocator is what decides how many concurrent sequences a
given amount of HBM can hold.

Sliding-window layers recycle blocks that leave the window, so their footprint
plateaus. Global layers grow linearly forever. A hybrid stack pays the linear
cost on only the fraction of layers that are global, which is why the pattern
exists: at 32k context a ``hybrid:4`` stack (one global layer in four) holds
about a quarter of the KV a fully global stack of the same shape would.
"""

from __future__ import annotations

from benchmarks.common import BENCH_MODEL, save, table
from kvforge.config import CacheConfig, ModelConfig
from kvforge.memory.manager import KVCacheManager
from kvforge.request import Request
from kvforge.sampling import SamplingParams

BLOCK_SIZE = 16
WINDOW = 1024
NUM_LAYERS = 32
LENGTHS = [1024, 4096, 8192, 16384, 32768]
CHUNK = 256


def kv_bytes_per_block(model: ModelConfig, block_size: int, num_layers: int) -> int:
    """One block covers ``block_size`` slots in every layer of its group, K and V."""
    element_size = 4  # float32
    return num_layers * 2 * block_size * model.num_kv_heads * model.head_dim * element_size


def measure(pattern: str) -> dict:
    model = ModelConfig(
        **(BENCH_MODEL | {"num_layers": NUM_LAYERS, "attention_pattern": pattern, "sliding_window": WINDOW})
    )
    cache = CacheConfig(BLOCK_SIZE, num_blocks=1 + max(LENGTHS) // BLOCK_SIZE * 2, enable_prefix_caching=False)
    kv = KVCacheManager(model, cache)

    request = Request("seq", [0] * CHUNK, SamplingParams(max_tokens=max(LENGTHS)))
    for _ in range(max(LENGTHS)):
        request.append_output_token(0)

    row: dict = {"pattern": pattern}
    at_length = {}
    length = 0
    while length < max(LENGTHS):
        step = min(CHUNK, max(LENGTHS) - length)
        assert kv.allocate_slots(request, step), "pool sized too small for the sweep"
        request.num_computed_tokens += step
        length += step
        if length in LENGTHS:
            per_group = []
            for gi, group in enumerate(kv.groups):
                live = sum(
                    1 for b in kv.req_blocks["seq"][gi] if b is not kv.pool.null_block
                )
                per_group.append(
                    {
                        "kind": group.kind,
                        "layers": len(group.layer_ids),
                        "live_blocks": live,
                        "bytes": live * kv_bytes_per_block(model, BLOCK_SIZE, len(group.layer_ids)),
                    }
                )
            at_length[length] = {
                "groups": per_group,
                "total_mb": round(sum(g["bytes"] for g in per_group) / 2**20, 2),
            }
    row["at_length"] = at_length
    kv.free(request)
    return row


def main() -> dict:
    patterns = ["full", "hybrid:4", "sliding"]
    measured = {p: measure(p) for p in patterns}

    rows = []
    for length in LENGTHS:
        row = {"context": length}
        for pattern in patterns:
            row[pattern] = measured[pattern]["at_length"][length]["total_mb"]
        row["hybrid_saving"] = f"{1 - row['hybrid:4'] / row['full']:.0%}"
        rows.append(row)

    print(
        f"{NUM_LAYERS} layers, {BENCH_MODEL['num_kv_heads']} KV heads x {BENCH_MODEL['head_dim']} dim, "
        f"fp32, window={WINDOW}, block_size={BLOCK_SIZE}\n"
    )
    print(
        table(
            rows,
            [
                ("context", "context tokens"),
                ("full", "all-global (MB)"),
                ("hybrid:4", "hybrid:4 (MB)"),
                ("sliding", "all-sliding (MB)"),
                ("hybrid_saving", "hybrid saving"),
            ],
        )
    )
    summary = {
        "layers": NUM_LAYERS,
        "window": WINDOW,
        "block_size": BLOCK_SIZE,
        "kv_heads": BENCH_MODEL["num_kv_heads"],
        "head_dim": BENCH_MODEL["head_dim"],
        "rows": rows,
        "detail": measured,
    }
    save("hybrid_memory", summary)
    return summary


if __name__ == "__main__":
    main()
