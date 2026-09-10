# Design

How kvforge is put together, and why each piece is shaped the way it is.

- [The step loop](#the-step-loop)
- [Paged KV cache](#paged-kv-cache)
- [Automatic prefix caching](#automatic-prefix-caching)
- [Hybrid models and KV cache groups](#hybrid-models-and-kv-cache-groups)
- [Scheduling](#scheduling)
- [Speculative decoding](#speculative-decoding)
- [MoE dispatch](#moe-dispatch)
- [Attention kernel](#attention-kernel)
- [How correctness is established](#how-correctness-is-established)
- [Limitations and roadmap](#limitations-and-roadmap)
- [References](#references)

---

## The step loop

```
LLMEngine.step()
  |
  |-- Scheduler.schedule()          decide who runs and for how many tokens
  |     |-- KVCacheManager.get_computed_blocks()    prefix-cache lookup
  |     |-- KVCacheManager.allocate_slots()         block tables, or refuse
  |     `-- preempt / requeue on refusal
  |
  |-- ModelRunner.prepare_input()   ragged tensors + per-group addressing
  |-- ModelRunner.execute()         one forward pass for the whole batch
  |
  |-- Sampler / rejection_sample()  tokens out of logits
  |-- Scheduler.update_from_output()  advance state, publish blocks, free finished
  `-- NgramProposer.propose()       drafts for the next step
```

Everything downstream of `prepare_input` is pure tensor work. The model never
sees a `Request`; it sees `input_ids`, `positions`, `logits_indices` and an
`AttentionMetadata`. That boundary is deliberate — it is what would let the
forward pass be CUDA-graph captured or `torch.compile`d once and replayed, which
is where a real engine claws back the per-step launch overhead that dominates
small-batch decode.

The other consequence of that boundary is that per-step Python work has to be
linear in *scheduled tokens*, never in sequence length. A decode step touches
one token per sequence, so slot mappings and block tables are sliced from state
the manager already holds rather than rebuilt from the sequence.

---

## Paged KV cache

The cache is a fixed pool of fixed-size blocks. Each block holds `block_size`
token slots **in every layer of its group**:

```
k_cache[layer]  :  [num_blocks, block_size, num_kv_heads, head_dim]
v_cache[layer]  :  same
```

A sequence owns a *block table*: a list of physical block ids, indexed by
logical block number. Token at position `p` lives at flat slot
`block_table[p // block_size] * block_size + p % block_size`.

This is the PagedAttention idea (Kwon et al., 2023): decouple the logical
contiguity of a sequence from the physical contiguity of its memory. What it
buys is the elimination of both kinds of waste that contiguous per-sequence
allocation forces on you — internal fragmentation from reserving `max_model_len`
up front, and external fragmentation from variable-size allocations — at the
cost of one indirection in the attention kernel.

**Sizing.** Block size trades two things off. Larger blocks mean shorter block
tables, fewer gathers and coarser (cheaper) bookkeeping; smaller blocks mean
less slack in the last partially filled block of each sequence, and finer prefix
sharing (a shared prefix can only be shared in whole blocks). 16 is the usual
sweet spot and is the default here. `tests/test_engine_parity.py` pins block
sizes 1, 4, 16 and 32 to identical outputs, so the choice is purely a
performance knob.

### Pool invariants

`BlockPool` (`kvforge/memory/pool.py`) maintains:

1. A block is in `free_queue` **iff** `ref_cnt == 0`.
2. A block is in `cached_blocks` **iff** `block_hash is not None`.
3. Those sets overlap. A hashed block with `ref_cnt == 0` is a live prefix-cache
   entry *and* an eviction candidate. Cache entries therefore cost nothing until
   memory is actually needed, and eviction is lazy.
4. Block 0 is the **null block**: never allocated, never freed. Sliding-window
   groups point recycled logical positions at it so block tables stay dense
   rectangular tensors instead of needing a `-1` sentinel in the kernel.

The free list is an *intrusive* doubly linked list — prev/next pointers live on
the block objects. Two operations are hot and both are O(1) as a result:
`popleft()` on allocation, and `remove(block)` when a prefix-cache hit pulls an
arbitrary block out of the middle. With a plain list the second is O(n) and it
runs once per cached block per request.

### Eviction order

Blocks are freed **tail-first**, so a sequence's last blocks enter the FIFO
before its first blocks. Allocation pops from the front. The net effect is that
the *end* of a finished sequence is recycled before its *prefix*, and the prefix
— the part another request is most likely to share — survives longest.

---

## Automatic prefix caching

A block's identity is the hash of everything that produced its KV state:

```
h_i = H(h_{i-1}, tokens[i*B : (i+1)*B], extra_keys)
```

Chaining on `h_{i-1}` is what makes the key sound. Two requests may contain the
same 16 tokens in the middle of completely unrelated prompts, and their
attention states differ, so their blocks must not be shared. Folding the
preceding context into the key means equal hashes imply equal prefixes.

`extra_keys` covers everything the KV depends on that the token ids do not
capture. The important case is **multimodal input**: an image and an audio clip
both expand to a run of identical placeholder token ids, so without the media
content hash in the key, a cached image block could be served to an audio
request. `generate_block_hash_extra_keys` folds in the content hash of every
media span overlapping the block, and
`tests/test_prefix_cache.py::test_multimodal_hash_separates_identical_placeholder_tokens`
pins it.

### Three rules that are easy to get wrong

**Only full blocks are hashed.** A partially filled block's KV is not final;
publishing it would hand an incomplete block to the next request that hit it.

**Publish after the forward pass, never before.** `cache_blocks` runs in
`update_from_output`, once the KV for those tokens has actually been written. A
block published at allocation time would be a valid-looking cache entry
containing zeros. A visible consequence: requests admitted in the *same* step
all miss, because none of them has published yet. That is correct, and the
benchmarks are set up to reflect it (a small `max_num_seqs` so requests arrive
in waves, which is what a real endpoint sees anyway).

**Hold back the last block.** A request whose prompt is 100% cached would have
zero query tokens and no logits to sample from. `get_computed_blocks` therefore
returns at most `num_prompt_blocks - 1` blocks, so there is always at least one
token to run a forward pass on.

### Lifecycle

```
allocate ──▶ fill ──▶ (forward) ──▶ publish hash ──▶ shared by ref_cnt
                                          │
                        owner finishes ───┤
                                          ▼
                              ref_cnt 0, still cached      ◀── hit: touch()
                                          │
                              pool drained │
                                          ▼
                                    evict + reuse
```

---

## Hybrid models and KV cache groups

A **KV cache group** is a set of layers that share memory behaviour. Llama-style
models have one group (all global). Gemma-3 / Ministral-style hybrids have two:
global layers, whose KV grows with the sequence, and sliding-window layers,
whose KV is bounded by the window. `attention_pattern="hybrid:N"` makes every
`N`-th layer global and the rest sliding, ending on a global layer.

Both groups draw from one block pool. Each request holds one block table per
group.

### Recycling a sliding window

When can a sliding-window block be returned? The earliest query position
scheduled in the current step is `num_computed_tokens`, and it attends back to
`num_computed_tokens - window + 1`. Anything strictly below that can never be
read again, so:

```python
first_live_block = max(0, num_computed_tokens - window + 1) // block_size
```

Blocks below that index are freed and their table entries set to the null block.
Deriving the bound from the *earliest* query in the step, rather than from the
sequence length, is what makes it safe with chunked prefill and with
speculative verifies, where a step covers many query positions at once.

### Compacted block tables

The block table for a sliding group is passed to the kernel already sliced to
the live window, with `kv_offsets` giving the absolute token position of its
first slot. The kernel never sees recycled blocks, so both its memory traffic
and its score matrix are O(window) rather than O(seq_len). Global groups take
the same path with `kv_offsets = 0`, so there is one code path, not two.

Measured effect (`benchmarks/bench_hybrid_memory.py`, 32 layers, 1k window):

| context | all-global | hybrid:4 | all-sliding |
|---|---|---|---|
| 4k | 128 MB | 62 MB | 40 MB |
| 32k | 1024 MB | 286 MB | 40 MB |

### Hybrid models and prefix caching

Prefix caching is **disabled** for hybrid models, in `EngineConfig.__post_init__`.

A global-group cache hit says "these tokens' global KV exists". It says nothing
about the sliding groups, which need the *window preceding the resume point* to
be resident — and those blocks were very likely recycled. Serving the hit
without that would silently produce wrong attention for the sliding layers.

Making this work means tracking, per cached prefix, whether each sliding group
still holds a matching window, and falling back to partial recompute when it
does not. That is real work and it is not done here; refusing the hit is the
honest interim behaviour, and it fails toward correctness rather than toward
speed.

---

## Scheduling

One step mixes three kinds of work in a single forward pass: decodes (1 query
token), prefill chunks (a slice of a prompt), and speculative verifies (`1 + k`
query tokens). The batch is ragged, described by `query_start_loc`.

Order of business each step:

1. **Running requests first.** Decodes and resumed prefill chunks get the budget
   before anything new is admitted, so an in-flight sequence is never starved by
   an arriving one.
2. **Then waiting requests**, up to `max_num_seqs` and the remaining token
   budget.

### Chunked prefill

Without it, one 8k-token prompt owns an entire step and every running decode
stalls for its duration. Splitting the prompt into budget-sized chunks lets
decodes ride along in the same batch: a little prefill throughput for a much
flatter inter-token latency distribution.

Chunking is cheap on the output side because a chunk that does not complete a
prompt produces **no logits at all** — `logits_indices` is empty for it, so the
`lm_head` GEMM (`hidden_size x vocab_size`, one of the largest in the model)
never runs for those tokens.

### Preemption by recomputation

Admission is optimistic and the pool is finite, so a running sequence can fail
to get a block. Rather than failing the request, the scheduler evicts the *most
recently admitted* running request, returns its blocks, and pushes it to the
**front** of the waiting queue.

- Evicting the newest keeps the oldest sequences making progress. Evicting the
  oldest instead produces a livelock where requests are repeatedly preempted
  just before finishing.
- Front of the queue, not the back: a preempted request has already waited.
- Only the KV is lost; the tokens are not. The request resumes as a prefill,
  which with prefix caching enabled is often nearly free, since its own blocks
  may still be in the cache.

`tests/test_engine_parity.py::test_preemption_under_memory_pressure_preserves_output`
runs the same workload with a cache 10x too small and demands identical tokens:
a full cache costs throughput, never correctness.

There is no swap-to-CPU path. Recomputation is simpler, and with prefix caching
it is usually cheaper than moving blocks over PCIe.

---

## Speculative decoding

The proposer is n-gram / prompt-lookup: find the most recent earlier occurrence
of the last `n` tokens and propose what followed. No draft model, no extra
weights, no extra forward pass. It works on traffic where output echoes input —
RAG, code editing, agent loops that restate state — and finds nothing otherwise,
which costs a little wasted verify width and nothing else.

Verification is standard rejection sampling (Leviathan et al.; Chen et al.):
accept draft `x` with probability `min(1, p(x)/q(x))`, else resample from the
normalised residual `(p - q)_+` and stop. If all `k` are accepted, a **bonus**
token is sampled from the target distribution at the last position — free,
because the target already computed those logits while verifying. That bonus is
where the win comes from: `k` drafts, one forward pass, up to `k + 1` tokens.

Greedy decoding needs no special case. `Sampler.compute_probs` returns a point
mass on the argmax when `temperature == 0`, so acceptance degenerates to an
equality check and a rejection deterministically resamples the argmax. One code
path serves both modes.

### KV bookkeeping for rejected drafts

This is the part that interacts with paging. A verify step writes KV for all
`1 + k` positions, but only `1 + a` of them (where `a` = accepted) belong to the
final sequence. The runtime does not need to erase anything:

```
num_computed_tokens += num_scheduled - (num_draft - num_accepted)
```

The slots for rejected positions stay allocated and are simply **overwritten
next step**, because `num_computed_tokens` did not advance past them. Blocks
allocated for the optimistic length stay owned by the request and get used as it
grows. The invariant `num_computed_tokens == num_tokens - 1` holds across
accepts and rejects alike, which is what keeps prefix-cache publishing correct:
`cache_blocks` only ever publishes blocks below `num_computed_tokens`, so a
rejected position can never be published.

---

## MoE dispatch

The naive top-k MoE runs every expert on every token and masks — trivially
correct, and `num_experts / top_k` times more FLOPs than necessary.

The grouped path sorts the `num_tokens * top_k` (token, expert) pairs by expert
so each expert's rows are contiguous, then runs one GEMM per expert over just
those rows, scattering results back with `index_add_`. This is the PyTorch shape
of what a fused grouped-GEMM (or the MegaBlocks block-sparse formulation) does
on GPU: same sort, same segment offsets, different inner GEMM.

Measured 2.9–4.8x over dense (`benchmarks/bench_moe.py`), against a FLOP-ratio
ceiling of 4–8x. The gap is sorting, gathering, scattering, and many small GEMMs
using the hardware worse than one large one — which is exactly the gap fused
kernels exist to close.

---

## Attention kernel

`kvforge/layers/attention.py` is a *reference* kernel: ragged queries, paged KV,
GQA, optional sliding window, in readable PyTorch. It materialises the score
matrix, which a real kernel (FlashAttention, the PagedAttention CUDA kernel)
does not — those fuse the gather into an online softmax.

The interface is the one those kernels take (`block_table`, `kv_offsets`,
`seq_lens`, `query_start_loc`), so swapping it is a local change. Keeping the
readable version in the tree has ongoing value beyond documentation: it is the
oracle the runtime is tested against, and
`test_vectorised_kernel_matches_loop_reference` keeps the vectorised path honest
against a per-sequence loop.

One numerical detail: masked positions are filled with `finfo.min`, not `-inf`.
A fully masked row — query padding, which the ragged-to-padded conversion
creates — then softmaxes to a uniform distribution instead of `NaN`. Its output
is discarded on the way out, but a `NaN` there would propagate through the
`einsum` and poison the whole batch.

---

## How correctness is established

81 tests. The load-bearing ones are equivalence tests, not smoke tests.

**Against an independent implementation.** `kvforge/models/reference.py` runs the
same weights with dense attention and no cache at all, recomputing the whole
sequence every token. Engine output must match it token for token — for dense,
sliding-window, hybrid and MoE models.

**Against a baseline configuration.** Each optimisation is asserted to be
invisible in the output:

| Property | Test |
|---|---|
| paging == dense attention | `test_paged_attention.py` (5 shapes, scattered block ids) |
| batched == one-at-a-time | `test_batched_matches_one_at_a_time` |
| chunked == whole-prompt prefill | `test_chunked_prefill_matches_whole_prompt_prefill` |
| prefix cache on == off | `test_prefix_caching_does_not_change_output` |
| preemption == no preemption | `test_preemption_under_memory_pressure_preserves_output` |
| block size 1/4/16/32 all equal | `test_block_size_does_not_change_output` |
| grouped MoE == dense MoE | `test_grouped_dispatch_matches_dense_evaluation` |
| speculation == no speculation | `test_speculative_decoding_is_lossless_under_greedy` |

**Statistically.** Rejection sampling is verified by sampling 40,000 tokens and
bounding the total-variation distance from the target distribution below 0.01 —
for both the point-mass proposer and the general draft-distribution case. An
equality test cannot catch a subtly biased sampler; this can.

Tests deliberately use adversarial layouts: shuffled non-monotonic physical
block ids, caches sized 10x too small, prefill budgets far below prompt length.

---

## Limitations and roadmap

Stated plainly, because knowing what a system does *not* do is part of
describing it.

- **The attention kernel materialises scores.** Reference implementation by
  choice; absolute throughput is not competitive with a real engine and is not
  meant to be. The benchmarks measure work avoided, which transfers.
- **CPU / float32 only.** No CUDA kernels, no quantisation, no paged FP8 KV.
- **Prefix caching is off for hybrid models.** Reason and fix sketched
  [above](#hybrid-models-and-prefix-caching).
- **Random weights.** Correctness is by parity, not by generated text. Loading a
  real checkpoint means a weight loader and a tokenizer, not runtime changes.
- **No tensor / pipeline parallelism.** Single process, single device.
- **The prefix cache keys on a 64-bit hash**, so collisions are possible in
  principle. This is what production engines do; the alternative is an
  O(block_size) memcmp on every lookup. Worth naming rather than hiding.
- **Multimodal support is the cache-key half only.** Placeholder spans hash
  correctly; there is no encoder and no embedding merge.
- **No swap-to-CPU preemption**, by choice — recompute plus prefix caching is
  usually cheaper than PCIe round trips.

Natural next steps, roughly in order of value: a FlashAttention/SDPA backend
behind the existing kernel interface; prefix caching for hybrid models; a real
checkpoint loader; EAGLE-style draft-model speculation (the verify path already
accepts a draft distribution); CUDA-graph capture of the forward pass.

---

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — paged KV cache, block tables, copy-on-write sharing.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 — iteration-level (continuous) batching.
- Agrawal et al., *Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference*, OSDI 2024 — chunked prefill and stall-free batching.
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 — draft-and-verify, rejection sampling.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023 — the distribution-preservation proof.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023.
- Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*, MLSys 2023 — block-sparse / grouped expert dispatch.
- Beltagy et al., *Longformer*, 2020 — sliding-window attention.
- Gemma Team, *Gemma 2* (2024) and *Gemma 3* (2025) — interleaved local/global attention, the hybrid pattern modelled here.
