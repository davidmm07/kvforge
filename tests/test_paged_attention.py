"""Paged attention must equal dense attention. No exceptions.

Every test here writes K/V into a paged cache through the same ``store_kv`` the
model uses, with deliberately shuffled physical block ids, then checks that the
kernel produces the dense result. If block layout could change the numbers, the
whole design would be unsound.
"""

import random

import pytest
import torch

from kvforge.layers.attention import (
    dense_causal_attention,
    paged_attention,
    paged_attention_ref,
    store_kv,
)

BLOCK_SIZE = 8
NUM_BLOCKS = 64
NUM_KV_HEADS = 2
HEAD_DIM = 16


def make_cache(dtype=torch.float32):
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)


def scatter_sequence(k_cache, v_cache, key, value, block_ids):
    """Write a whole sequence into the given (non-contiguous) physical blocks."""
    slots = [
        block_ids[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
        for i in range(key.shape[0])
    ]
    store_kv(key, value, k_cache, v_cache, torch.tensor(slots, dtype=torch.long))


@pytest.mark.parametrize("num_heads", [2, 8])  # MHA and GQA (8 q heads / 2 kv heads)
@pytest.mark.parametrize("seq_len", [1, 7, 8, 9, 33])
def test_prefill_matches_dense(num_heads, seq_len):
    torch.manual_seed(0)
    k_cache, v_cache = make_cache()
    q = torch.randn(seq_len, num_heads, HEAD_DIM)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)

    # Scattered, non-monotonic block ids: physical layout must not matter.
    num_blocks = -(-seq_len // BLOCK_SIZE)
    block_ids = random.Random(seq_len).sample(range(1, NUM_BLOCKS), num_blocks)
    scatter_sequence(k_cache, v_cache, k, v, block_ids)

    out = paged_attention(
        q,
        k_cache,
        v_cache,
        block_table=torch.tensor([block_ids], dtype=torch.long),
        kv_offsets=torch.tensor([0], dtype=torch.long),
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.long),
    )
    expected = dense_causal_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_decode_matches_dense_last_row():
    """One query token against a long cached context."""
    torch.manual_seed(1)
    seq_len, num_heads = 40, 8
    k_cache, v_cache = make_cache()
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
    q_all = torch.randn(seq_len, num_heads, HEAD_DIM)

    block_ids = random.Random(7).sample(range(1, NUM_BLOCKS), -(-seq_len // BLOCK_SIZE))
    scatter_sequence(k_cache, v_cache, k, v, block_ids)

    out = paged_attention(
        q_all[-1:],
        k_cache,
        v_cache,
        block_table=torch.tensor([block_ids], dtype=torch.long),
        kv_offsets=torch.tensor([0], dtype=torch.long),
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        query_start_loc=torch.tensor([0, 1], dtype=torch.long),
    )
    expected = dense_causal_attention(q_all, k, v)[-1:]
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_mixed_batch_prefill_decode_and_spec():
    """The batch shape continuous batching actually produces.

    One sequence doing a 20-token prefill chunk, one decoding a single token,
    one verifying 3 speculative tokens - all in one ragged forward pass.
    """
    torch.manual_seed(2)
    num_heads = 8
    k_cache, v_cache = make_cache()
    rng = random.Random(11)

    # (total context length, number of query tokens this step)
    specs = [(20, 20), (35, 1), (17, 3)]
    keys, values, block_tables, queries, q_lens = [], [], [], [], []
    free_ids = list(range(1, NUM_BLOCKS))
    rng.shuffle(free_ids)

    for seq_len, q_len in specs:
        k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
        v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
        num_blocks = -(-seq_len // BLOCK_SIZE)
        ids = [free_ids.pop() for _ in range(num_blocks)]
        scatter_sequence(k_cache, v_cache, k, v, ids)
        keys.append(k)
        values.append(v)
        block_tables.append(ids)
        queries.append(torch.randn(seq_len, num_heads, HEAD_DIM))
        q_lens.append(q_len)

    width = max(len(t) for t in block_tables)
    padded = [t + [0] * (width - len(t)) for t in block_tables]
    q_flat = torch.cat([q[-n:] for q, n in zip(queries, q_lens)])
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.long)

    out = paged_attention(
        q_flat,
        k_cache,
        v_cache,
        block_table=torch.tensor(padded, dtype=torch.long),
        kv_offsets=torch.zeros(len(specs), dtype=torch.long),
        seq_lens=torch.tensor([s for s, _ in specs], dtype=torch.long),
        query_start_loc=qsl,
    )
    expected = torch.cat(
        [
            dense_causal_attention(q, k, v)[-n:]
            for q, k, v, n in zip(queries, keys, values, q_lens)
        ]
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("window", [4, 8, 13])
def test_sliding_window_with_compacted_block_table(window):
    """Sliding-window layers see only the live window, via ``kv_offsets``.

    The kernel is handed a block table that starts partway into the sequence, so
    it must never assume slot 0 is token 0.
    """
    torch.manual_seed(3)
    seq_len, num_heads = 37, 8
    k_cache, v_cache = make_cache()
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
    q_all = torch.randn(seq_len, num_heads, HEAD_DIM)
    ids = random.Random(window).sample(range(1, NUM_BLOCKS), -(-seq_len // BLOCK_SIZE))
    scatter_sequence(k_cache, v_cache, k, v, ids)

    # Only blocks from here on can still be read by a query at seq_len - 1.
    first_live = max(0, seq_len - 1 - window + 1) // BLOCK_SIZE
    out = paged_attention(
        q_all[-1:],
        k_cache,
        v_cache,
        block_table=torch.tensor([ids[first_live:]], dtype=torch.long),
        kv_offsets=torch.tensor([first_live * BLOCK_SIZE], dtype=torch.long),
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        query_start_loc=torch.tensor([0, 1], dtype=torch.long),
        window=window,
    )
    expected = dense_causal_attention(q_all, k, v, window=window)[-1:]
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_vectorised_kernel_matches_loop_reference():
    """The batched einsum path and the per-sequence loop must agree."""
    torch.manual_seed(4)
    num_heads = 8
    k_cache, v_cache = make_cache()
    specs = [(23, 5), (8, 8), (31, 1)]
    block_tables, q_flat, q_lens = [], [], []
    free_ids = list(range(1, NUM_BLOCKS))
    random.Random(5).shuffle(free_ids)
    for seq_len, q_len in specs:
        k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
        v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM)
        ids = [free_ids.pop() for _ in range(-(-seq_len // BLOCK_SIZE))]
        scatter_sequence(k_cache, v_cache, k, v, ids)
        block_tables.append(ids)
        q_flat.append(torch.randn(q_len, num_heads, HEAD_DIM))
        q_lens.append(q_len)

    width = max(len(t) for t in block_tables)
    padded = torch.tensor([t + [0] * (width - len(t)) for t in block_tables], dtype=torch.long)
    args = dict(
        block_table=padded,
        kv_offsets=torch.zeros(len(specs), dtype=torch.long),
        seq_lens=torch.tensor([s for s, _ in specs], dtype=torch.long),
        query_start_loc=torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.long),
    )
    q = torch.cat(q_flat)
    fast = paged_attention(q, k_cache, v_cache, **args)
    slow = paged_attention_ref(q, k_cache, v_cache, **args)
    torch.testing.assert_close(fast, slow, atol=1e-5, rtol=1e-5)
