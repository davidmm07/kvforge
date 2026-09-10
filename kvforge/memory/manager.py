"""KV cache manager: block tables, prefix-cache lookup, hybrid layer groups.

A *KV cache group* is a set of layers that share the same memory behaviour. A
Llama-style model has one group (all layers global). A Gemma-3-style hybrid model
has two: the global layers, whose KV grows with the sequence, and the
sliding-window layers, whose KV is bounded by the window. Grouping them lets a
single block pool serve both, with per-group block tables, instead of
provisioning the worst case for every layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvforge.config import AttentionKind, CacheConfig, ModelConfig
from kvforge.memory.block import KVCacheBlock
from kvforge.memory.pool import BlockPool
from kvforge.memory.prefix_cache import extend_block_hashes
from kvforge.request import Request


@dataclass(frozen=True)
class KVCacheGroup:
    kind: AttentionKind
    layer_ids: tuple[int, ...]
    #: Attention window for sliding groups, ``None`` for global groups.
    window: int | None = None


def build_kv_cache_groups(model: ModelConfig) -> list[KVCacheGroup]:
    kinds = model.layer_kinds()
    groups: list[KVCacheGroup] = []
    for kind in ("full", "sliding"):
        layer_ids = tuple(i for i, k in enumerate(kinds) if k == kind)
        if layer_ids:
            groups.append(
                KVCacheGroup(
                    kind=kind,
                    layer_ids=layer_ids,
                    window=model.sliding_window if kind == "sliding" else None,
                )
            )
    return groups


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class KVCacheManager:
    """Owns the mapping from logical token positions to physical block slots."""

    def __init__(self, model: ModelConfig, cache: CacheConfig) -> None:
        self.block_size = cache.block_size
        self.groups = build_kv_cache_groups(model)
        self.pool = BlockPool(cache.num_blocks, cache.enable_prefix_caching)
        self.enable_prefix_caching = cache.enable_prefix_caching

        #: req_id -> per-group list of blocks, indexed by logical block number.
        #: Entries may be the null block for sliding groups that have scrolled
        #: past that position.
        self.req_blocks: dict[str, list[list[KVCacheBlock]]] = {}
        #: How many leading blocks of the global group are already published to
        #: the prefix cache, so we never rescan the whole sequence.
        self.num_cached_blocks: dict[str, int] = {}

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def full_group_index(self) -> int | None:
        for i, g in enumerate(self.groups):
            if g.kind == "full":
                return i
        return None

    # ---------------------------------------------------------- prefix lookup

    def get_computed_blocks(self, request: Request) -> tuple[list[list[KVCacheBlock]], int]:
        """Look the request's prompt up in the prefix cache.

        Returns the hit blocks per group and the number of prompt tokens they
        cover. Nothing is reference-counted here: the scheduler may still decline
        to admit the request, and taking references for a request we do not run
        would leak them.
        """
        empty: list[list[KVCacheBlock]] = [[] for _ in self.groups]
        if not self.enable_prefix_caching or request.num_computed_tokens > 0:
            return empty, 0
        # Prefix caching across a hybrid model needs the sliding groups to still
        # hold a matching window; the config layer disables it for those models.
        if any(g.kind != "full" for g in self.groups):
            return empty, 0

        block_hashes = extend_block_hashes(request, self.block_size)
        num_prompt_blocks = request.num_prompt_tokens // self.block_size

        self.pool.stats.queries += 1
        hits: list[KVCacheBlock] = []
        for i in range(min(num_prompt_blocks, len(block_hashes))):
            block = self.pool.get_cached_block(block_hashes[i])
            if block is None:
                break
            hits.append(block)

        # The model needs at least one token to run a forward pass on, so a
        # request can never be 100% cache hit. Give back the last block.
        if hits and len(hits) * self.block_size == request.num_prompt_tokens:
            hits.pop()

        if hits:
            self.pool.stats.hits += 1
        return [hits], len(hits) * self.block_size

    # ------------------------------------------------------------- allocation

    def _first_live_block(self, group: KVCacheGroup, num_computed: int) -> int:
        """Index of the first block a sliding group must still keep.

        The earliest query position scheduled in this step is ``num_computed``
        and it attends back to ``num_computed - window + 1``. Anything strictly
        below that can never be read again.
        """
        if group.kind != "sliding" or group.window is None:
            return 0
        min_kv_pos = max(0, num_computed - group.window + 1)
        return min_kv_pos // self.block_size

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: list[list[KVCacheBlock]] | None = None,
    ) -> bool:
        """Make room for ``num_new_tokens`` more tokens of ``request``.

        Returns ``False`` without mutating anything if the pool cannot satisfy
        the request; the scheduler then preempts or defers.
        """
        num_computed = request.num_computed_tokens
        total_tokens = num_computed + num_new_tokens
        assert total_tokens <= request.num_tokens_with_spec

        req_blocks = self.req_blocks.get(request.request_id)
        if req_blocks is None:
            req_blocks = [[] for _ in self.groups]

        hit_blocks = new_computed_blocks or [[] for _ in self.groups]

        # --- dry run: can we afford it? -----------------------------------
        num_logical = cdiv(total_tokens, self.block_size)
        needed = sum(
            max(0, num_logical - len(req_blocks[gi]) - len(hit_blocks[gi]))
            for gi in range(len(self.groups))
        )
        if needed > self.pool.num_free_blocks:
            return False

        # --- commit --------------------------------------------------------
        for gi, group in enumerate(self.groups):
            blocks = req_blocks[gi]
            if hit_blocks[gi]:
                assert not blocks, "prefix-cache hits only apply at admission"
                self.pool.touch(hit_blocks[gi])
                blocks.extend(hit_blocks[gi])
                self.num_cached_blocks[request.request_id] = len(hit_blocks[gi])

            num_logical = cdiv(total_tokens, self.block_size)
            num_new = num_logical - len(blocks)
            if num_new > 0:
                blocks.extend(self.pool.allocate(num_new))

            # Recycle everything that scrolled out of a sliding window.
            first_live = self._first_live_block(group, num_computed)
            if first_live > 0:
                stale = [
                    b
                    for b in blocks[:first_live]
                    if b is not self.pool.null_block
                ]
                if stale:
                    self.pool.free(list(reversed(stale)))
                    for i in range(first_live):
                        blocks[i] = self.pool.null_block

        self.req_blocks[request.request_id] = req_blocks
        self.num_cached_blocks.setdefault(request.request_id, 0)
        return True

    def cache_blocks(self, request: Request) -> None:
        """Publish newly completed blocks to the prefix cache.

        Runs *after* the forward pass that filled them. Publishing earlier would
        expose blocks whose KV has not been written yet.
        """
        if not self.enable_prefix_caching:
            return
        gi = self.full_group_index
        if gi is None or any(g.kind != "full" for g in self.groups):
            return
        blocks = self.req_blocks.get(request.request_id)
        if blocks is None:
            return
        num_full = request.num_computed_tokens // self.block_size
        already = self.num_cached_blocks.get(request.request_id, 0)
        if num_full <= already:
            return
        hashes = extend_block_hashes(request, self.block_size)
        num_full = min(num_full, len(hashes), len(blocks[gi]))
        self.pool.cache_full_blocks(blocks[gi], hashes, already, num_full)
        self.num_cached_blocks[request.request_id] = num_full

    def free(self, request: Request) -> None:
        blocks = self.req_blocks.pop(request.request_id, None)
        self.num_cached_blocks.pop(request.request_id, None)
        if blocks is None:
            return
        for group_blocks in blocks:
            # Tail-first, so the shared prefix outlives the request-specific end.
            self.pool.free([b for b in reversed(group_blocks) if b is not self.pool.null_block])

    # ------------------------------------------------------------- accessors

    def block_ids(self, request_id: str, group_idx: int) -> list[int]:
        return [b.block_id for b in self.req_blocks[request_id][group_idx]]

    def first_live_block(self, request_id: str, group_idx: int, num_computed: int) -> int:
        return self._first_live_block(self.groups[group_idx], num_computed)

    def slot_indices(
        self, request_id: str, group_idx: int, start: int, end: int
    ) -> list[int]:
        """Physical slot index for each token position in ``[start, end)``."""
        blocks = self.req_blocks[request_id][group_idx]
        bs = self.block_size
        out = []
        for pos in range(start, end):
            block = blocks[pos // bs]
            assert block is not self.pool.null_block, (
                "writing to a recycled sliding-window block"
            )
            out.append(block.block_id * bs + pos % bs)
        return out
