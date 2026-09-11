"""Input validation at the engine's API boundary.

These failures used to surface several frames deep in ``nn.Embedding`` as a bare
``IndexError: index out of range in self``, naming neither the offending token
nor the request. Catching them in ``add_request`` is the difference between a
five-second fix and a debugging session.
"""

import pytest

from kvforge import CacheConfig, EngineConfig, LLMEngine, ModelConfig, SamplingParams, SchedulerConfig

VOCAB = 64


def make_engine() -> LLMEngine:
    return LLMEngine(
        EngineConfig(
            model=ModelConfig(
                hidden_size=32, num_layers=1, num_heads=2, num_kv_heads=1,
                ffn_hidden_size=64, vocab_size=VOCAB, seed=0,
            ),
            cache=CacheConfig(block_size=8, num_blocks=32),
            scheduler=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=128),
        )
    )


def test_token_id_at_or_above_vocab_size_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match=r"token id 64 is outside"):
        engine.add_request([1, 2, VOCAB])


def test_error_names_the_token_and_the_vocabulary_bound():
    engine = make_engine()
    with pytest.raises(ValueError) as excinfo:
        engine.add_request([900])
    message = str(excinfo.value)
    assert "900" in message
    assert f"[0, {VOCAB})" in message
    assert "vocab_size" in message  # points at the thing to fix


def test_negative_token_id_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match=r"token id -1 is outside"):
        engine.add_request([1, -1, 2])


def test_empty_prompt_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="empty"):
        engine.add_request([])


def test_the_last_valid_token_id_is_accepted():
    """Boundary: vocab_size - 1 is a legal id, vocab_size is not."""
    engine = make_engine()
    engine.add_request([VOCAB - 1], SamplingParams(max_tokens=1, ignore_eos=True))
    (out,) = engine.run_to_completion()
    assert len(out.output_token_ids) == 1


def test_a_rejected_request_leaves_no_trace():
    """Validation runs before any state is touched, so a bad call is a no-op."""
    engine = make_engine()
    free_before = engine.kv.pool.num_free_blocks
    with pytest.raises(ValueError):
        engine.add_request([VOCAB + 5])
    assert engine.requests == {}
    assert not engine.scheduler.waiting
    assert engine.stats.prompt_tokens == 0
    assert engine.kv.pool.num_free_blocks == free_before


def test_generate_rejects_the_whole_batch_not_half_of_it():
    """One bad prompt must not leave earlier prompts queued in the engine."""
    engine = make_engine()
    with pytest.raises(ValueError, match="outside"):
        engine.generate([[1, 2], [3, VOCAB + 1]], SamplingParams(max_tokens=1))
    assert engine.requests == {}
    assert not engine.scheduler.waiting


def test_prompt_longer_than_max_model_len_is_rejected():
    engine = make_engine()  # max_model_len = 128
    with pytest.raises(ValueError, match="max_model_len is 128"):
        engine.add_request([1] * 129)


def test_prompt_larger_than_the_kv_pool_is_rejected_clearly():
    """The oversized-prompt case that used to crash with a bare KeyError.

    With 32 blocks of 8 slots and block 0 reserved, the pool holds 248 tokens;
    a 300-token prompt can never be scheduled and must be refused at the door,
    not dropped silently and then surfaced as ``KeyError: 'req-0'`` from inside
    ``generate``.
    """
    # max_model_len must be above the pool so the pool check is the one that fires.
    engine = LLMEngine(
        EngineConfig(
            model=ModelConfig(
                hidden_size=32, num_layers=1, num_heads=2, num_kv_heads=1,
                ffn_hidden_size=64, vocab_size=VOCAB, seed=0,
            ),
            cache=CacheConfig(block_size=8, num_blocks=32),
            scheduler=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=512, max_model_len=4096),
        )
    )
    with pytest.raises(ValueError, match="KV cache holds at most 248"):
        engine.generate([[1] * 300], SamplingParams(max_tokens=1))
    assert engine.requests == {}
    assert not engine.scheduler.waiting


def test_a_prompt_that_exactly_fills_the_pool_is_still_accepted():
    """The boundary must not over-reject: usable_slots tokens is legal."""
    engine = LLMEngine(
        EngineConfig(
            model=ModelConfig(
                hidden_size=32, num_layers=1, num_heads=2, num_kv_heads=1,
                ffn_hidden_size=64, vocab_size=VOCAB, seed=0,
            ),
            cache=CacheConfig(block_size=8, num_blocks=6),  # 5 usable blocks = 40 slots
            scheduler=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64, max_model_len=64),
        )
    )
    engine.add_request(list(range(40)), SamplingParams(max_tokens=1, ignore_eos=True))
    assert len(engine.scheduler.waiting) == 1
