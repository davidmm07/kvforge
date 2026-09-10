"""Physical KV blocks and the free list that owns them.

The free list is an *intrusive* doubly linked list: the prev/next pointers live
on the block objects themselves. That is what makes the two hot operations O(1):

* ``popleft()`` on allocation (evict the least-recently-freed block),
* ``remove(block)`` on a prefix-cache hit, where we have to pull an arbitrary
  block out of the middle of the list.

A plain ``list`` would make the second operation O(n), and it runs once per
cached block per request, so it shows up immediately under load.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheBlock:
    """One physical block of ``block_size`` token slots, in every layer."""

    block_id: int
    #: Number of live sequences pointing at this block. Zero means the block is
    #: in the free list, which does *not* mean its contents are gone: a hashed
    #: block with ref_cnt == 0 is still a valid prefix-cache entry until it is
    #: actually reused.
    ref_cnt: int = 0
    #: Hash of (prefix, tokens in this block). ``None`` for a partially filled
    #: block, which can never be shared.
    block_hash: int | None = None

    prev_free: "KVCacheBlock | None" = None
    next_free: "KVCacheBlock | None" = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        assert self.ref_cnt > 0, f"double free of block {self.block_id}"
        self.ref_cnt -= 1

    def reset_hash(self) -> None:
        self.block_hash = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"KVCacheBlock(id={self.block_id}, ref={self.ref_cnt}, hash={self.block_hash})"


class FreeBlockQueue:
    """Intrusive FIFO of free blocks, ordered oldest-freed first.

    Eviction order is therefore LRU over *freed* blocks. Callers free a
    sequence's blocks tail-first so that the blocks nearest the end of a
    sequence are evicted before its prefix: a shared prompt prefix is the part
    most likely to be reused by the next request.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self._head: KVCacheBlock | None = None
        self._tail: KVCacheBlock | None = None
        for block in blocks:
            self._append_unchecked(block)

    def _append_unchecked(self, block: KVCacheBlock) -> None:
        block.prev_free = self._tail
        block.next_free = None
        if self._tail is not None:
            self._tail.next_free = block
        else:
            self._head = block
        self._tail = block

    def popleft(self) -> KVCacheBlock:
        if self._head is None:
            raise IndexError("no free blocks")
        block = self._head
        self.remove(block)
        return block

    def append(self, block: KVCacheBlock) -> None:
        self._append_unchecked(block)
        self.num_free_blocks += 1

    def remove(self, block: KVCacheBlock) -> None:
        prev, nxt = block.prev_free, block.next_free
        if prev is not None:
            prev.next_free = nxt
        else:
            self._head = nxt
        if nxt is not None:
            nxt.prev_free = prev
        else:
            self._tail = prev
        block.prev_free = block.next_free = None
        self.num_free_blocks -= 1

    def as_list(self) -> list[KVCacheBlock]:  # pragma: no cover - test helper
        out, node = [], self._head
        while node is not None:
            out.append(node)
            node = node.next_free
        return out

    def __len__(self) -> int:
        return self.num_free_blocks
