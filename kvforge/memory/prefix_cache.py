"""Hash-chained automatic prefix caching.

A block is identified by the hash of *everything that produced its KV state*:

    h_i = H(h_{i-1}, tokens[i*B : (i+1)*B], extra_keys)

Chaining on ``h_{i-1}`` is what makes the key safe. Two requests may contain the
same 16 tokens in the middle of completely different prompts; their attention
states are different, so their blocks must not be shared. Chaining folds the
whole preceding context into the key, so equal hashes imply equal prefixes.

``extra_keys`` covers everything else the KV depends on but that the token ids do
not capture. The important case is multimodal input: an image and an audio clip
both expand to a run of identical placeholder token ids, so without the media
content hash in the key, a cached image block could be handed to an audio
request. See ``generate_block_hash_extra_keys``.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from kvforge.request import Request

#: Salt for the first block in a chain. Using a constant (rather than starting
#: from ``None``) keeps hashes stable across processes, which matters if the
#: cache is ever shared across workers or persisted.
INIT_HASH = 0x9E3779B97F4A7C15


def hash_block_tokens(
    parent_hash: int,
    token_ids: Sequence[int],
    extra_keys: tuple | None = None,
) -> int:
    """Hash one block given its parent's hash.

    We key the cache on a 64-bit digest rather than the token ids themselves,
    which is what production engines do: comparing full token tuples on every
    lookup would put an O(block_size) memcmp on the critical path. The lookup
    then trusts the digest, so its collision resistance *is* the cache's
    isolation between unrelated prefixes; a collision would serve one request KV
    computed from another's tokens.

    We therefore use BLAKE2b, not Python's builtin ``hash()``. ``hash()`` salts
    strings with ``PYTHONHASHSEED``, so any prompt with multimodal ``extra_keys``
    would hash differently in every process, silently breaking a shared or
    persisted cache; and its collisions are cheap to craft offline because the
    integer-tuple path is unsalted. BLAKE2b is deterministic across processes and
    cryptographically collision-resistant, which closes both. Inputs are
    length-delimited so that no regrouping of tokens or keys can alias.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(parent_hash.to_bytes(8, "little"))
    h.update(struct.pack("<Q", len(token_ids)))
    h.update(struct.pack(f"<{len(token_ids)}q", *token_ids))
    if extra_keys:
        for key in extra_keys:
            key_bytes = str(key).encode("utf-8")
            h.update(struct.pack("<Q", len(key_bytes)))
            h.update(key_bytes)
    return int.from_bytes(h.digest(), "little")


def generate_block_hash_extra_keys(
    request: "Request", start: int, end: int
) -> tuple | None:
    """Extra hash keys for the token range ``[start, end)`` of a request.

    Returns the content hashes of every multimodal span overlapping the range,
    or ``None`` for a pure-text block (so text-only requests pay nothing).
    """
    if not request.multi_modal_inputs:
        return None
    keys: list[str] = []
    for mm in request.multi_modal_inputs:
        mm_start, mm_end = mm.offset, mm.offset + mm.length
        if mm_start < end and start < mm_end:
            keys.append(mm.mm_hash)
    return tuple(keys) if keys else None


def hash_request_tokens(request: "Request", block_size: int) -> list[int]:
    """Hash every *full* block of a request's current token sequence.

    A partial trailing block is skipped: its KV state is not final, so sharing it
    would leak an incomplete block to another request.
    """
    token_ids = request.all_token_ids
    num_full = len(token_ids) // block_size
    hashes: list[int] = []
    parent = INIT_HASH
    for i in range(num_full):
        start, end = i * block_size, (i + 1) * block_size
        extra = generate_block_hash_extra_keys(request, start, end)
        parent = hash_block_tokens(parent, token_ids[start:end], extra)
        hashes.append(parent)
    return hashes


def extend_block_hashes(request: "Request", block_size: int) -> list[int]:
    """Incrementally extend ``request.block_hashes`` to cover new full blocks.

    Called once per scheduler step per request, so it must not rehash the whole
    sequence: it only hashes blocks that became full since the last call.
    """
    token_ids = request.all_token_ids
    num_full = len(token_ids) // block_size
    hashes = request.block_hashes
    parent = hashes[-1] if hashes else INIT_HASH
    for i in range(len(hashes), num_full):
        start, end = i * block_size, (i + 1) * block_size
        extra = generate_block_hash_extra_keys(request, start, end)
        parent = hash_block_tokens(parent, token_ids[start:end], extra)
        hashes.append(parent)
    return hashes
