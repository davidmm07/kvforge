"""Paged attention, written in plain PyTorch.

Cache layout is ``[num_blocks, block_size, num_kv_heads, head_dim]`` per layer,
one tensor for K and one for V. Two things are worth calling out:

**Varlen queries.** Under continuous batching a single forward pass mixes a
128-token prefill chunk, a 1-token decode and a 5-token speculative verify. The
batch is therefore a *ragged* ``[total_tokens, heads, dim]`` tensor described by
``query_start_loc``, not a rectangular ``[batch, seq, ...]`` one. Padding every
sequence to the longest would waste most of the FLOPs in a decode-heavy batch.

**Sliding-window compaction.** A sliding-window layer only ever reads the last
``window`` tokens, so its block table is passed already compacted to the live
window, with ``kv_offsets`` giving the absolute position of its first slot. The
kernel never sees the recycled blocks, so both its memory traffic and its score
matrix stay O(window) instead of O(seq_len).

This is a *reference* kernel: readable, correct, and used as the parity oracle
for the runtime. A production deployment swaps it for FlashAttention or the
PagedAttention CUDA kernel, which fuse the gather into the softmax instead of
materialising the score matrix. The interface here is the one those kernels
take, so the swap is local.
"""

from __future__ import annotations

import math

import torch


def store_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter this step's K/V into their physical slots.

    ``key``/``value`` are ``[num_tokens, num_kv_heads, head_dim]`` and
    ``slot_mapping`` is ``[num_tokens]`` of flat slot indices
    (``block_id * block_size + offset``).
    """
    num_kv_heads, head_dim = key.shape[1], key.shape[2]
    k_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slot_mapping, key)
    v_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slot_mapping, value)


def _ragged_index(query_start_loc: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, ...]:
    """Map each flat token to (sequence index, position within that sequence)."""
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_seqs = q_lens.shape[0]
    seq_idx = torch.repeat_interleave(
        torch.arange(num_seqs, device=q_lens.device), q_lens
    )
    pos_idx = (
        torch.arange(num_tokens, device=q_lens.device)
        - query_start_loc[:-1].repeat_interleave(q_lens)
    )
    return seq_idx, pos_idx, q_lens


def paged_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_offsets: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Attention over a paged KV cache with ragged queries.

    Args:
        query: ``[num_tokens, num_heads, head_dim]``, sequences concatenated.
        k_cache, v_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        block_table: ``[num_seqs, max_num_blocks]`` physical block ids, already
            compacted to the live window for sliding-window layers.
        kv_offsets: ``[num_seqs]`` absolute token position of block_table[:, 0].
        seq_lens: ``[num_seqs]`` total context length, including this step.
        query_start_loc: ``[num_seqs + 1]`` prefix sums of the query lengths.
        window: sliding-window size, or ``None`` for global attention.

    Returns:
        ``[num_tokens, num_heads, head_dim]``.
    """
    num_tokens, num_heads, head_dim = query.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    num_seqs = seq_lens.shape[0]
    groups = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    seq_idx, pos_idx, q_lens = _ragged_index(query_start_loc, num_tokens)
    max_q = int(q_lens.max())

    # Ragged -> padded, so the whole batch is one einsum.
    q_pad = query.new_zeros(num_seqs, max_q, num_heads, head_dim)
    q_pad[seq_idx, pos_idx] = query

    # Gather the KV window for every sequence: [S, nb, B, Hkv, D] -> [S, L, Hkv, D]
    k = k_cache[block_table].flatten(1, 2)
    v = v_cache[block_table].flatten(1, 2)
    kv_len = k.shape[1]

    kv_pos = kv_offsets.unsqueeze(1) + torch.arange(kv_len, device=query.device)
    # Absolute position of every (padded) query slot.
    q_pos = (
        seq_lens.unsqueeze(1)
        - q_lens.unsqueeze(1)
        + torch.arange(max_q, device=query.device)
    )

    valid = (
        (kv_pos.unsqueeze(1) <= q_pos.unsqueeze(2))  # causal
        & (kv_pos.unsqueeze(1) < seq_lens.view(-1, 1, 1))  # inside the context
        & (torch.arange(max_q, device=query.device).view(1, -1, 1) < q_lens.view(-1, 1, 1))
    )
    if window is not None:
        valid &= kv_pos.unsqueeze(1) > (q_pos.unsqueeze(2) - window)

    q5 = q_pad.view(num_seqs, max_q, num_kv_heads, groups, head_dim)
    scores = torch.einsum("sqkgd,slkd->skgql", q5, k) * scale
    # finfo.min rather than -inf: a fully masked row (query padding) then
    # softmaxes to a uniform distribution instead of NaN, and its output is
    # discarded on the way out. NaNs here would poison the whole batch.
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~valid.unsqueeze(1).unsqueeze(1), neg)
    probs = scores.softmax(dim=-1)
    out = torch.einsum("skgql,slkd->sqkgd", probs, v)
    out = out.reshape(num_seqs, max_q, num_heads, head_dim)
    return out[seq_idx, pos_idx]


def paged_attention_ref(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_offsets: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Sequence-at-a-time reference. Slow, obviously correct, used in tests."""
    num_tokens, num_heads, head_dim = query.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    groups = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(query)

    for s in range(seq_lens.shape[0]):
        start, end = int(query_start_loc[s]), int(query_start_loc[s + 1])
        if start == end:
            continue
        q_len, seq_len = end - start, int(seq_lens[s])
        offset = int(kv_offsets[s])
        num_blocks = (seq_len - offset + block_size - 1) // block_size
        ids = block_table[s, :num_blocks]
        k = k_cache[ids].reshape(-1, num_kv_heads, head_dim)[: seq_len - offset]
        v = v_cache[ids].reshape(-1, num_kv_heads, head_dim)[: seq_len - offset]

        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        q = query[start:end]

        scores = torch.einsum("qhd,lhd->hql", q, k) * scale
        q_pos = torch.arange(seq_len - q_len, seq_len, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(offset, seq_len, device=q.device).unsqueeze(0)
        mask = kv_pos <= q_pos
        if window is not None:
            mask &= kv_pos > q_pos - window
        scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        out[start:end] = torch.einsum("hql,lhd->qhd", scores.softmax(-1), v)
    return out


def dense_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Textbook dense attention on ``[seq, heads, dim]``, the parity oracle.

    Used by tests to prove that paging and block reuse do not change the maths.
    """
    seq_len, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    groups = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    k = key.repeat_interleave(groups, dim=1)
    v = value.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,lhd->hql", query, k) * scale
    pos = torch.arange(seq_len, device=query.device)
    mask = pos.unsqueeze(0) <= pos.unsqueeze(1)
    if window is not None:
        mask &= pos.unsqueeze(0) > pos.unsqueeze(1) - window
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    return torch.einsum("hql,lhd->qhd", scores.softmax(-1), v)
