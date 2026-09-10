"""The physical block pool: allocation, reference counting and eviction."""

from __future__ import annotations

from dataclasses import dataclass

from kvforge.memory.block import FreeBlockQueue, KVCacheBlock


@dataclass
class PrefixCacheStats:
    queries: int = 0
    hits: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0


class BlockPool:
    """Owns every physical block and the prefix-cache index over them.

    Invariants:

    * A block is in ``free_queue`` **iff** ``ref_cnt == 0``.
    * A block is in ``cached_blocks`` **iff** ``block_hash is not None``.
    * Those two sets overlap: a hashed, unreferenced block is a live cache entry
      that is also a candidate for eviction. Eviction only happens when the free
      list is drained, so cache entries survive for free until memory is
      actually needed.

    Block 0 is reserved as the *null block*. Sliding-window layers point their
    out-of-window logical blocks at it, which keeps block tables dense (and
    therefore a plain rectangular tensor) without special-casing -1 in the
    attention kernel.
    """

    def __init__(self, num_blocks: int, enable_prefix_caching: bool = True) -> None:
        assert num_blocks >= 2
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks: list[KVCacheBlock] = [KVCacheBlock(i) for i in range(num_blocks)]
        self.null_block = self.blocks[0]
        self.null_block.incr_ref()  # never freed, never allocated

        self.free_queue = FreeBlockQueue(self.blocks[1:])
        self.cached_blocks: dict[int, KVCacheBlock] = {}
        self.stats = PrefixCacheStats()

    # ------------------------------------------------------------------ query

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - 1 - self.num_free_blocks

    def utilisation(self) -> float:
        return self.num_used_blocks / (self.num_blocks - 1)

    def get_cached_block(self, block_hash: int) -> KVCacheBlock | None:
        if not self.enable_prefix_caching:
            return None
        return self.cached_blocks.get(block_hash)

    # ------------------------------------------------------------- allocation

    def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
        """Take ``num_blocks`` fresh blocks, evicting cache entries if needed."""
        if num_blocks > self.num_free_blocks:
            raise MemoryError(
                f"requested {num_blocks} blocks, only {self.num_free_blocks} free"
            )
        out: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self.free_queue.popleft()
            if block.block_hash is not None:
                # Reusing a hashed block destroys its contents, so it has to
                # leave the prefix cache first.
                self._evict(block)
            block.incr_ref()
            out.append(block)
        return out

    def _evict(self, block: KVCacheBlock) -> None:
        if self.cached_blocks.get(block.block_hash) is block:
            del self.cached_blocks[block.block_hash]
            self.stats.evictions += 1
        block.reset_hash()

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """Take a reference on cache-hit blocks, pulling them out of the free list."""
        for block in blocks:
            if block.ref_cnt == 0 and block is not self.null_block:
                self.free_queue.remove(block)
            block.incr_ref()

    def free(self, blocks: list[KVCacheBlock]) -> None:
        """Release references.

        ``blocks`` should be ordered **tail-first**. Blocks land at the back of
        the FIFO in the order given, and allocation pops from the front, so
        passing the tail first means the end of a sequence is evicted before its
        prefix. The prefix is the part another request is likely to share.
        """
        for block in blocks:
            if block is self.null_block:
                continue
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_queue.append(block)

    def cache_full_blocks(
        self,
        blocks: list[KVCacheBlock],
        block_hashes: list[int],
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Register blocks ``[start_idx, end_idx)`` of a sequence in the cache.

        Called after a forward pass has written the KV for those tokens, never
        before: publishing a block whose KV is not yet written would hand
        garbage to the next request that hits it.
        """
        if not self.enable_prefix_caching:
            return
        for i in range(start_idx, end_idx):
            block = blocks[i]
            if block is self.null_block or block.block_hash is not None:
                continue
            block_hash = block_hashes[i]
            existing = self.cached_blocks.get(block_hash)
            if existing is not None and existing is not block:
                # Two in-flight requests filled the same prefix concurrently.
                # Keep the first entry; this block simply stays uncached and is
                # freed normally with its owner.
                continue
            block.block_hash = block_hash
            self.cached_blocks[block_hash] = block

    def reset_prefix_cache(self) -> bool:
        """Drop every cache entry. Fails if any cached block is still in use."""
        if any(b.ref_cnt > 0 for b in self.cached_blocks.values()):
            return False
        for block in self.cached_blocks.values():
            block.reset_hash()
        self.cached_blocks.clear()
        self.stats = PrefixCacheStats()
        return True
