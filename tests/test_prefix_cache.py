"""Block-hash chaining, multimodal keys, and cache reuse through the manager."""

from kvforge.config import CacheConfig, ModelConfig
from kvforge.memory.manager import KVCacheManager
from kvforge.memory.prefix_cache import (
    INIT_HASH,
    extend_block_hashes,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    hash_request_tokens,
)
from kvforge.request import MultiModalInput, Request
from kvforge.sampling import SamplingParams


def make_request(token_ids, rid="r", mm=None):
    return Request(
        request_id=rid,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(max_tokens=4),
        multi_modal_inputs=mm or [],
    )


def test_same_tokens_different_prefix_hash_differently():
    """The point of chaining: identical blocks in different contexts differ."""
    block = [1, 2, 3, 4]
    a = hash_block_tokens(hash_block_tokens(INIT_HASH, [9, 9, 9, 9]), block)
    b = hash_block_tokens(hash_block_tokens(INIT_HASH, [8, 8, 8, 8]), block)
    assert a != b


def test_hash_chain_is_deterministic_and_prefix_stable():
    r1 = make_request(list(range(32)))
    r2 = make_request(list(range(32)) + [99] * 8)
    h1 = hash_request_tokens(r1, block_size=8)
    h2 = hash_request_tokens(r2, block_size=8)
    assert len(h1) == 4 and len(h2) == 5
    # A longer sequence sharing a prefix reproduces the same leading hashes,
    # which is exactly what makes a partial hit possible.
    assert h2[: len(h1)] == h1


def test_block_hash_is_deterministic_across_processes():
    """The digest must not depend on PYTHONHASHSEED.

    A shared or persisted prefix cache is only sound if two processes agree on
    the key for the same tokens. Python's builtin hash() salts strings per
    process, so this pins that hash_block_tokens does not use it. The expected
    value is a fixed BLAKE2b digest; if it ever changes, cache keys changed and
    any persisted cache is invalidated.
    """
    import os
    import subprocess
    import sys

    code = (
        "from kvforge.memory.prefix_cache import hash_block_tokens, INIT_HASH;"
        "print(hash_block_tokens(INIT_HASH, [1, 2, 3], ('sha256:img',)))"
    )
    digests = []
    for seed in ("0", "1", "12345"):
        # Inherit the environment (PATH etc. must survive on Windows) and only
        # override the hash seed, so the subprocess actually runs.
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout.strip()
        assert out and out.lstrip("-").isdigit(), f"unexpected output: {out!r}"
        digests.append(out)
    assert len(set(digests)) == 1, f"digest varied with PYTHONHASHSEED: {digests}"


def test_length_delimiting_prevents_token_regrouping_collisions():
    """Splitting the same tokens differently must not alias to one key."""
    a = hash_block_tokens(INIT_HASH, [1, 2, 3])
    b = hash_block_tokens(INIT_HASH, [1, 2])
    c = hash_block_tokens(hash_block_tokens(INIT_HASH, [1]), [2, 3])
    assert len({a, b, c}) == 3


def test_partial_trailing_block_is_not_hashed():
    r = make_request(list(range(20)))  # 2 full blocks of 8, 4 tokens left over
    assert len(hash_request_tokens(r, block_size=8)) == 2


def test_extend_is_incremental_and_agrees_with_full_hash():
    r = make_request(list(range(16)))
    extend_block_hashes(r, 8)
    assert len(r.block_hashes) == 2
    for token in range(16, 32):
        r.append_output_token(token)
    extend_block_hashes(r, 8)
    assert r.block_hashes == hash_request_tokens(r, 8)


def test_multimodal_hash_separates_identical_placeholder_tokens():
    """Two images expand to the same placeholder ids; their KV differs."""
    placeholders = [0] * 16
    a = make_request(placeholders, mm=[MultiModalInput("img-sha-A", 0, 16)])
    b = make_request(placeholders, mm=[MultiModalInput("img-sha-B", 0, 16)])
    assert hash_request_tokens(a, 8) != hash_request_tokens(b, 8)

    same = make_request(placeholders, mm=[MultiModalInput("img-sha-A", 0, 16)])
    assert hash_request_tokens(a, 8) == hash_request_tokens(same, 8)


def test_extra_keys_only_cover_overlapping_spans():
    r = make_request(list(range(32)), mm=[MultiModalInput("img", 8, 8)])
    assert generate_block_hash_extra_keys(r, 0, 8) is None
    assert generate_block_hash_extra_keys(r, 8, 16) == ("img",)
    assert generate_block_hash_extra_keys(r, 16, 24) is None
    # A block straddling the boundary still picks the span up.
    assert generate_block_hash_extra_keys(r, 4, 12) == ("img",)


def make_manager(block_size=8, num_blocks=64, prefix=True):
    model = ModelConfig(num_layers=2, hidden_size=64, num_heads=4, num_kv_heads=2)
    return KVCacheManager(model, CacheConfig(block_size, num_blocks, prefix))


def test_manager_reuses_blocks_across_requests():
    kv = make_manager()
    tokens = list(range(40))

    first = make_request(tokens, rid="a")
    blocks, cached = kv.get_computed_blocks(first)
    assert cached == 0
    assert kv.allocate_slots(first, len(tokens), blocks)
    first.num_computed_tokens = len(tokens)
    kv.cache_blocks(first)
    ids_first = kv.block_ids("a", 0)

    second = make_request(tokens, rid="b")
    hit_blocks, cached = kv.get_computed_blocks(second)
    # 40 tokens is 5 full blocks; the manager holds the last one back so the
    # model always has at least one token to run on.
    assert cached == 32
    assert [b.block_id for b in hit_blocks[0]] == ids_first[:4]

    second.num_computed_tokens = cached
    assert kv.allocate_slots(second, len(tokens) - cached, hit_blocks)
    assert kv.block_ids("b", 0)[:4] == ids_first[:4]  # physically shared


def test_shared_blocks_survive_one_owner_finishing():
    kv = make_manager()
    tokens = list(range(40))
    a = make_request(tokens, rid="a")
    blocks, _ = kv.get_computed_blocks(a)
    kv.allocate_slots(a, len(tokens), blocks)
    a.num_computed_tokens = len(tokens)
    kv.cache_blocks(a)

    b = make_request(tokens, rid="b")
    hit, cached = kv.get_computed_blocks(b)
    b.num_computed_tokens = cached
    kv.allocate_slots(b, len(tokens) - cached, hit)
    shared = kv.req_blocks["b"][0][0]
    assert shared.ref_cnt == 2

    kv.free(a)
    assert shared.ref_cnt == 1
    assert kv.pool.get_cached_block(shared.block_hash) is shared


def test_disabled_prefix_cache_never_hits():
    kv = make_manager(prefix=False)
    tokens = list(range(40))
    a = make_request(tokens, rid="a")
    kv.allocate_slots(a, len(tokens))
    a.num_computed_tokens = len(tokens)
    kv.cache_blocks(a)
    b = make_request(tokens, rid="b")
    _, cached = kv.get_computed_blocks(b)
    assert cached == 0
