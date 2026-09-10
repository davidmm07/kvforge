# Benchmarks

Every number here is produced by `python -m benchmarks.run_all`, which also
writes `results/*.json`. Re-running regenerates this file.

**Read these as runtime results, not kernel results.** kvforge's attention is a
readable reference implementation in plain PyTorch, so absolute throughput is not
the point and is not competitive with a real engine. What the benchmarks measure
is work *avoided* by the runtime - prefill tokens not recomputed, forward passes
not run, expert FLOPs not spent, KV bytes not held - and those quantities carry
over to any kernel underneath.

## Environment
- CPython 3.13.14, torch 2.14.0+cpu, device `cpu`
- AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD, Windows 11
- generated 2026-09-05

## Automatic prefix caching

A served chat endpoint: four 256-token system prompts reused across 32
requests with short distinct user turns.

The number to read is **prefill tokens computed**. Prefix caching does not make
attention faster; it removes prefill work that has already been done. Hit rate
here is bounded by block granularity and by the last-block hold-back (a request
must always have at least one token to run a forward pass on), so ~76% of a
~90%-shared prompt is close to the ceiling for this workload.

TTFT improves by roughly the same factor as the prefill work removed, which is
the expected relationship: for a request whose prompt is mostly cached, TTFT is
dominated by the prefill it still has to do.

```
| prefix caching | prefill tokens computed | prompt token hit rate | forward passes | mean TTFT (ms) | p90 TTFT (ms) | wall (s) |
|---|---|---|---|---|---|---|
| False | 9385 | 0.0 | 192 | 2451.3 | 4607.6 | 4.943 |
| True | 2217 | 0.7638 | 192 | 1645.6 | 3008.9 | 3.367 |

prefill tokens avoided: 7168 (76.4%), TTFT 1.49x, wall 1.47x
```

## Continuous batching vs static batching

Same engine, same batch width, same workload; only the admission policy
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
result is the transferable one; the wall clock is an artefact of the kernel.

```
| mode | forward passes | mean batch size | output tokens | wall (s) | output tok/s |
|---|---|---|---|---|---|
| static | 120 | 3.4 | 408 | 1.908 | 213.8 |
| continuous | 78 | 5.23 | 408 | 1.648 | 247.5 |

1.16x throughput, 35.0% fewer forward passes, 1.54x batch occupancy
```

## N-gram speculative decoding

Repetitive prompts (standing in for RAG / code-edit / agent traffic),
batch of 8, sweeping the number of draft tokens `k`.

Every value of `k` produces a **byte-identical token stream**, asserted in the
benchmark itself - that is the property rejection sampling buys, and the reason
speculation is safe to enable by default rather than being a quality trade.

Acceptance rate falls as `k` grows, which is the expected shape: later draft
positions are conditioned on earlier ones being right. The absolute acceptance
numbers here are pessimistic because the benchmark model has random weights, so
its continuations do not actually echo the prompt the way a trained model's do;
what the sweep demonstrates is the mechanism and its cost curve, not a quality
figure for a real checkpoint.

```
| draft tokens (k) | forward passes | tokens / forward (batch of 8) | acceptance rate | wall (s) |
|---|---|---|---|---|
| 0 | 48 | 8.0 | 0.0 | 1.056 |
| 1 | 40 | 9.6 | 0.713 | 1.013 |
| 2 | 40 | 9.6 | 0.548 | 1.005 |
| 4 | 39 | 9.846 | 0.404 | 1.043 |
| 6 | 39 | 9.846 | 0.328 | 1.051 |

output identical for every k (lossless). best k=4: 18.8% fewer forward passes, 1.01x wall
```

## MoE dispatch

Grouped (sort-by-expert) dispatch against dense evaluation of every expert
on every token.

The FLOP ratio is `num_experts / top_k`, and measured speedup tracks it without
reaching it: sorting, gathering and scattering are real costs, and many small
per-expert GEMMs use the hardware worse than one large one. That gap between
achieved and theoretical is exactly what fused grouped-GEMM kernels close.

```
| experts | top-k | tokens | dense (ms) | grouped (ms) | speedup | FLOP ratio |
|---|---|---|---|---|---|---|
| 8 | 2 | 1 | 4.373 | 1.201 | 3.64 | 4.0 |
| 8 | 2 | 64 | 23.719 | 9.54 | 2.49 | 4.0 |
| 8 | 2 | 512 | 75.144 | 25.975 | 2.89 | 4.0 |
| 32 | 4 | 512 | 300.454 | 55.135 | 5.45 | 8.0 |
| 64 | 8 | 512 | 548.436 | 113.049 | 4.85 | 8.0 |
```

## Hybrid attention KV footprint

One sequence grown to 32k tokens through the real allocator, measuring
physically live blocks per KV cache group.

Sliding-window layers recycle blocks the moment they leave the window, so their
footprint plateaus at `window / block_size + 1` blocks and never grows again.
Global layers grow linearly forever. A `hybrid:4` stack pays the linear cost on
one layer in four, and the saving converges toward 75% as context grows.

KV footprint is what caps concurrency: at 32k context this is the difference
between holding one sequence per GB and holding three and a half.

```
32 layers, 2 KV heads x 64 dim, fp32, window=1024, block_size=16

| context tokens | all-global (MB) | hybrid:4 (MB) | all-sliding (MB) | hybrid saving |
|---|---|---|---|---|
| 1024 | 32.0 | 32.0 | 32.0 | 0% |
| 4096 | 128.0 | 62.0 | 40.0 | 52% |
| 8192 | 256.0 | 94.0 | 40.0 | 63% |
| 16384 | 512.0 | 158.0 | 40.0 | 69% |
| 32768 | 1024.0 | 286.0 | 40.0 | 72% |
```

## Reproducing

```bash
pip install -e '.[dev]'
python -m benchmarks.run_all
```
