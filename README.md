# kvforge

[![CI](https://github.com/davidmm07/kvforge/actions/workflows/ci.yml/badge.svg)](https://github.com/davidmm07/kvforge/actions/workflows/ci.yml)

An LLM inference runtime built to be read: **paged KV cache**, **automatic
prefix caching**, **continuous batching with chunked prefill**, **hybrid
global + sliding-window attention**, **MoE dispatch**, and **speculative
decoding**, in about 2,400 lines of PyTorch with no dependency beyond `torch`.

It is a study of the parts of vLLM/SGLang-class engines that decide how much
inference costs: not the kernels, but the memory manager and the scheduler that
sit above them. Every optimisation is accompanied by a test proving it does not
change a single output token, and a benchmark measuring what it actually saves.

```bash
git clone https://github.com/davidmm07/kvforge.git && cd kvforge
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip
python examples/quickstart.py     # guided tour
python -m pytest                  # 86 tests
python -m benchmarks.run_all      # regenerates docs/BENCHMARKS.md
```

---

## Background: what the KV cache is

*Skip if you already know. Everything below assumes this much.*

Attention projects every token into three tensors: a **query**, a **key** and a
**value**. A token's query is compared against the keys of every earlier token to
decide where to attend, and the resulting weights are applied to those tokens'
values. **KV** is those keys and values.

During generation, the keys and values of tokens already in the sequence never
change: token 500's K and V are identical whether you are producing token 501 or
token 5,000. So you compute them once and keep them, and that store is the **KV
cache**. Queries are not cached, because a query is only ever used by the token
that produced it.

Recomputing them instead is the O(n²) path in
[`models/reference.py`](kvforge/models/reference.py): correct, obviously so, and
the reason that file is a test oracle rather than something you would serve with.

The cache is also the dominant memory consumer in serving:

```
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
```

For the 32-layer model in the hybrid benchmark (2 KV heads, 64 head dim, fp32)
that is **32 KB per token**, so a single 32k-token conversation holds a gigabyte
of KV before you serve a second user. Everything this repo does is about making
that number smaller or making the memory go further:

| feature | what it does to the KV cache |
|---|---|
| paging | hands out fixed-size blocks on demand instead of reserving the worst case per sequence |
| prefix caching | lets requests sharing a prompt prefix share the physical blocks holding its KV |
| sliding windows | frees KV that a layer's attention window has already moved past |
| preemption | evicts a whole sequence's KV when the pool runs dry, and recomputes it later |

---

## What it does, measured

Full methodology and interpretation in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| | result | what it means |
|---|---|---|
| **Prefix caching** | **76% of prompt tokens never recomputed**, 1.5x TTFT | 32 chat requests over four shared 256-token system prompts |
| **Continuous batching** | **35% fewer forward passes**, 1.5x batch occupancy | vs static batching, same width, output lengths 4–48 |
| **Speculative decoding** | **32 → 15 forward passes**, byte-identical output | n-gram drafting, k=4, on repetitive input |
| **MoE grouped dispatch** | **2.9–4.8x** over dense expert evaluation | 8–64 experts, top-2/4/8 |
| **Hybrid attention** | **1024 MB → 286 MB** of KV at 32k context | 32 layers, one global in four, 1k window |

> These are *runtime* results: work avoided, not kernels made faster. kvforge's
> attention is a readable reference implementation, so absolute throughput is not
> competitive with a production engine and is not meant to be. Prefill tokens not
> recomputed, forward passes not run and KV bytes not held are properties of the
> runtime, and they carry over to whatever kernel sits underneath.

---

## The pieces

```
kvforge/
├── memory/
│   ├── block.py          intrusive O(1) free list over physical blocks
│   ├── pool.py           refcounting, LRU eviction, the prefix-cache index
│   ├── prefix_cache.py   hash chaining, multimodal extra keys
│   └── manager.py        block tables, hybrid KV cache groups, window recycling
├── core/
│   ├── scheduler.py      continuous batching, chunked prefill, preemption
│   ├── model_runner.py   ragged batch construction, per-group addressing
│   ├── batch.py          the model's view of a step
│   └── engine.py         the step loop
├── layers/
│   ├── attention.py      paged attention: varlen queries, GQA, sliding window
│   ├── moe.py            top-k router, grouped vs dense dispatch
│   └── rope.py
├── models/
│   ├── transformer.py    decoder wired for paged, continuously batched inference
│   └── reference.py      dense cache-free twin, the correctness oracle
└── spec/
    ├── ngram.py          prompt-lookup drafting
    └── rejection.py      distribution-preserving verification
```

### Paged KV cache

Fixed-size blocks, one block table per sequence, physical layout decoupled from
logical position. Block 0 is a reserved null block so that recycled slots keep
block tables dense instead of needing a sentinel in the kernel. The free list is
intrusive, because a prefix-cache hit has to pull an arbitrary block out of the
middle of it and that happens once per cached block per request.

Blocks are freed **tail-first** so a finished sequence's ending is recycled
before its prefix, which is the part the next request is likely to share.

### Automatic prefix caching

`h_i = H(h_{i-1}, tokens[i*B:(i+1)*B], extra_keys)`. Chaining on the parent hash
is what makes sharing sound: identical tokens under different prefixes hash
differently, so equal hashes imply equal context.

`extra_keys` carries multimodal content hashes. An image and an audio clip both
expand to identical placeholder token ids, so without them a cached image block
could be handed to an audio request.

Three rules that are easy to get wrong, and are tested here: only *full* blocks
are hashed; blocks are published *after* the forward pass that filled them,
never at allocation; and the last block is held back so a fully cached prompt
still has a token to run on.

### Continuous batching

One forward pass mixes decodes (1 query token), prefill chunks, and speculative
verifies (`1+k` tokens) as a ragged batch. Running sequences get the token
budget before new ones are admitted. Chunked prefill stops a long prompt from
stalling every in-flight decode, and a chunk that does not finish a prompt emits
no logits at all, so the `lm_head` GEMM is skipped for it.

When the pool runs dry, the **most recently admitted** request is preempted, its
blocks returned, and it is requeued at the **front**. Evicting the newest is
what prevents the livelock where everyone is preempted just before finishing.
Only KV is lost. The tokens remain, so it resumes as a (usually prefix-cached)
prefill.

### Hybrid attention

`attention_pattern="hybrid:4"` makes every fourth layer global and the rest
sliding-window, the Gemma-3 / Ministral shape. Layers are partitioned into **KV
cache groups** sharing one block pool. Sliding groups recycle blocks that leave
the window and hand the kernel a block table already compacted to the live
window, so both memory and score-matrix size are O(window) instead of
O(seq_len).

Prefix caching is deliberately **disabled** for hybrid models: a global-group hit
says nothing about whether the sliding groups still hold the matching window.
The reasoning and the fix are in [`docs/DESIGN.md`](docs/DESIGN.md#hybrid-models-and-prefix-caching).
It fails toward correctness rather than toward speed.

### Speculative decoding

N-gram (prompt-lookup) drafting with rejection-sampling verification. `k` drafts
cost one forward pass and yield up to `k+1` tokens, the extra one being free
because the target already computed those logits while verifying.

The interesting part is the KV bookkeeping: a verify writes KV for all `1+k`
positions but only `1+a` are kept. Nothing is erased: `num_computed_tokens`
simply does not advance past the rejected ones, so their slots are overwritten
next step, and the invariant that keeps prefix-cache publishing correct
(`num_computed_tokens == num_tokens - 1`) survives accepts and rejects alike.

Greedy needs no special case: a temperature-0 distribution is a point mass, so
acceptance degenerates to an equality check and rejection deterministically
resamples the argmax.

---

## Correctness

86 tests, and the load-bearing ones are equivalence tests rather than smoke
tests. [`models/reference.py`](kvforge/models/reference.py) is an independent
dense, cache-free implementation sharing only the weights; the engine must match
it token for token across dense, sliding-window, hybrid and MoE models.

Every optimisation is separately asserted to be invisible:

| property | test |
|---|---|
| paging == dense attention (scattered block ids) | `test_paged_attention.py` |
| batched == one request at a time | `test_batched_matches_one_at_a_time` |
| chunked == whole-prompt prefill | `test_chunked_prefill_matches_whole_prompt_prefill` |
| prefix cache on == off | `test_prefix_caching_does_not_change_output` |
| preemption == no preemption | `test_preemption_under_memory_pressure_preserves_output` |
| block size 1 / 4 / 16 / 32 all equal | `test_block_size_does_not_change_output` |
| grouped MoE == dense MoE | `test_grouped_dispatch_matches_dense_evaluation` |
| speculation == no speculation | `test_speculative_decoding_is_lossless_under_greedy` |

Rejection sampling gets a statistical test rather than an equality test: 40,000
samples, total-variation distance from the target distribution bounded below
0.01, for both the point-mass proposer and the general draft-distribution case.
A subtly biased sampler passes equality tests; it does not pass this one.

Tests use adversarial layouts on purpose: shuffled non-monotonic physical block
ids, caches sized 10x too small, prefill budgets far below prompt length.

---

## Using it

```python
from kvforge import (
    LLMEngine, EngineConfig, ModelConfig, CacheConfig,
    SchedulerConfig, SpeculativeConfig, SamplingParams,
)

engine = LLMEngine(EngineConfig(
    model=ModelConfig(
        hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=2,
        attention_pattern="hybrid:4", sliding_window=1024,   # Gemma-3 shape
    ),
    cache=CacheConfig(block_size=16, num_blocks=2048, enable_prefix_caching=True),
    scheduler=SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=2048),
    speculative=SpeculativeConfig(num_speculative_tokens=4),
))

# Offline: submit everything, get results in order.
outputs = engine.generate(prompts, SamplingParams(temperature=0.7, max_tokens=128))

# Online: add requests any time, drive one step at a time.
engine.add_request(prompt_token_ids, SamplingParams(max_tokens=64))
while engine.scheduler.has_work:
    for finished in engine.step():
        print(finished.request_id, finished.output_token_ids)
```

Weights are random by design. This is a runtime, and a random-weight model of
the right *shape* exercises every code path a checkpoint would without a
multi-gigabyte download. Correctness comes from parity against the dense
reference, which is a stronger check than eyeballing generated text. Loading a
real checkpoint needs a weight loader and a tokenizer, not runtime changes.

---

## Limitations

Named rather than hidden; full list with reasoning in
[`docs/DESIGN.md`](docs/DESIGN.md#limitations-and-roadmap).

- Reference attention kernel that materialises scores. CPU, float32, no CUDA.
- Prefix caching off for hybrid models.
- Random weights; no tokenizer, no checkpoint loader.
- No tensor/pipeline parallelism, no quantisation.
- Multimodal support is the cache-key half only: no encoder, no embedding merge.
- The prefix cache keys on a 64-bit BLAKE2b digest with no token re-check on a
  hit, so a collision is possible in principle. At 64 cryptographic bits the
  risk is negligible, and it is the same trade production engines make, but
  worth stating.

Next steps in rough value order: a FlashAttention/SDPA backend behind the
existing kernel interface, prefix caching for hybrid models, a checkpoint
loader, EAGLE-style draft-model speculation (the verify path already accepts a
draft distribution), CUDA-graph capture.

---

## Reading order

If you want the ideas rather than the API:

1. [`docs/DESIGN.md`](docs/DESIGN.md): the whole system, with the reasoning.
2. [`memory/pool.py`](kvforge/memory/pool.py): the invariants everything rests on.
3. [`memory/manager.py`](kvforge/memory/manager.py): block tables and window recycling.
4. [`core/scheduler.py`](kvforge/core/scheduler.py): the policy decisions.
5. [`spec/rejection.py`](kvforge/spec/rejection.py): why speculation is free of quality cost.

## References

PagedAttention (Kwon et al., SOSP'23) · Orca continuous batching (Yu et al.,
OSDI'22) · Sarathi-Serve chunked prefill (Agrawal et al., OSDI'24) ·
Speculative decoding (Leviathan et al., ICML'23; Chen et al., 2023) · GQA
(Ainslie et al., EMNLP'23) · MegaBlocks (Gale et al., MLSys'23) · Gemma 2/3
interleaved local-global attention. Full citations in
[`docs/DESIGN.md`](docs/DESIGN.md#references).

## License

MIT
