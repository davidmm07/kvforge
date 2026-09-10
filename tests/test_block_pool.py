"""Block pool invariants: reference counting, LRU order, eviction."""

import pytest

from kvforge.memory.pool import BlockPool


def test_null_block_is_never_handed_out():
    pool = BlockPool(8)
    blocks = pool.allocate(7)
    assert all(b.block_id != 0 for b in blocks)
    assert pool.num_free_blocks == 0


def test_allocate_and_free_roundtrip():
    pool = BlockPool(8)
    blocks = pool.allocate(3)
    assert pool.num_free_blocks == 4
    pool.free(list(reversed(blocks)))
    assert pool.num_free_blocks == 7


def test_refcount_shared_block_survives_one_free():
    pool = BlockPool(8)
    (block,) = pool.allocate(1)
    pool.touch([block])  # a second sequence shares it
    assert block.ref_cnt == 2
    pool.free([block])
    assert block.ref_cnt == 1
    assert pool.num_free_blocks == 6  # still in use, not back in the pool
    pool.free([block])
    assert pool.num_free_blocks == 7


def test_free_order_puts_tail_first_in_the_eviction_queue():
    """Freeing tail-first means the shared prefix is evicted last."""
    pool = BlockPool(8)
    blocks = pool.allocate(3)  # logical order: prefix -> tail
    pool.free(list(reversed(blocks)))
    queue = [b.block_id for b in pool.free_queue.as_list()]
    # Never-used blocks are evicted first; among the freed ones the tail block
    # comes before the prefix block, so the prefix survives longest.
    assert queue[-3:] == [blocks[2].block_id, blocks[1].block_id, blocks[0].block_id]


def test_cached_block_is_reusable_until_actually_evicted():
    pool = BlockPool(4)
    (block,) = pool.allocate(1)
    pool.cache_full_blocks([block], [1234], 0, 1)
    pool.free([block])
    assert block.ref_cnt == 0
    # Unreferenced but still a valid cache entry.
    assert pool.get_cached_block(1234) is block
    pool.touch([block])
    assert block.ref_cnt == 1
    assert pool.num_free_blocks == 2  # pulled back out of the free list


def test_eviction_drops_the_cache_entry():
    pool = BlockPool(3)  # blocks 1 and 2 are usable
    a, b = pool.allocate(2)
    pool.cache_full_blocks([a, b], [11, 22], 0, 2)
    pool.free([b, a])
    assert pool.get_cached_block(11) is a

    # Forcing an allocation must recycle the oldest freed block (b, freed first).
    (taken,) = pool.allocate(1)
    assert taken is b
    assert pool.get_cached_block(22) is None
    assert pool.get_cached_block(11) is a
    assert pool.stats.evictions == 1


def test_allocate_beyond_capacity_raises():
    pool = BlockPool(4)
    with pytest.raises(MemoryError):
        pool.allocate(4)


def test_reset_prefix_cache_requires_idle_blocks():
    pool = BlockPool(4)
    (block,) = pool.allocate(1)
    pool.cache_full_blocks([block], [7], 0, 1)
    assert pool.reset_prefix_cache() is False  # still referenced
    pool.free([block])
    assert pool.reset_prefix_cache() is True
    assert pool.get_cached_block(7) is None
    assert block.block_hash is None


def test_double_free_is_caught():
    pool = BlockPool(4)
    (block,) = pool.allocate(1)
    pool.free([block])
    with pytest.raises(AssertionError):
        pool.free([block])
