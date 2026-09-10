"""MoE dispatch equivalence, and the memory behaviour of hybrid models."""

import torch

from kvforge import CacheConfig, EngineConfig, LLMEngine, ModelConfig, SamplingParams, SchedulerConfig
from kvforge.layers.moe import SparseMoEBlock


# ------------------------------------------------------------------- MoE


def test_grouped_dispatch_matches_dense_evaluation():
    """Sorting tokens by expert must not change a single output value."""
    torch.manual_seed(0)
    moe = SparseMoEBlock(hidden_size=32, ffn_hidden_size=64, num_experts=8, top_k=2)
    x = torch.randn(64, 32)
    torch.testing.assert_close(moe(x, mode="grouped"), moe(x, mode="dense"), atol=1e-5, rtol=1e-5)


def test_grouped_dispatch_handles_idle_experts():
    """With one token and top_k=1, seven of eight experts get nothing."""
    torch.manual_seed(1)
    moe = SparseMoEBlock(hidden_size=16, ffn_hidden_size=32, num_experts=8, top_k=1)
    x = torch.randn(1, 16)
    torch.testing.assert_close(moe(x, mode="grouped"), moe(x, mode="dense"), atol=1e-5, rtol=1e-5)


def test_router_weights_are_normalised_when_asked():
    torch.manual_seed(2)
    moe = SparseMoEBlock(16, 32, num_experts=4, top_k=2, renormalize=True)
    weights, expert_ids = moe.route(torch.randn(10, 16))
    assert weights.shape == (10, 2) and expert_ids.shape == (10, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones(10), atol=1e-6, rtol=1e-6)
    assert (expert_ids[:, 0] != expert_ids[:, 1]).all()  # top-k without replacement


def test_unnormalised_router_keeps_raw_softmax_mass():
    torch.manual_seed(3)
    moe = SparseMoEBlock(16, 32, num_experts=8, top_k=2, renormalize=False)
    weights, _ = moe.route(torch.randn(10, 16))
    assert (weights.sum(-1) < 1.0).all()  # top-2 of 8 cannot be the whole mass


# ---------------------------------------------------------------- hybrid


def hybrid_engine(pattern, window, num_blocks=512, block_size=16):
    return LLMEngine(
        EngineConfig(
            model=ModelConfig(
                hidden_size=64,
                num_layers=4,
                num_heads=4,
                num_kv_heads=2,
                ffn_hidden_size=128,
                vocab_size=128,
                attention_pattern=pattern,
                sliding_window=window,
                seed=99,
            ),
            cache=CacheConfig(block_size=block_size, num_blocks=num_blocks),
            scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=1024),
        )
    )


def test_layer_kinds_follow_the_declared_pattern():
    full = ModelConfig(num_layers=6, attention_pattern="full")
    assert full.layer_kinds() == ["full"] * 6
    hybrid = ModelConfig(num_layers=6, attention_pattern="hybrid:3")
    # Every third layer global, and the stack ends on a global layer.
    assert hybrid.layer_kinds() == ["sliding", "sliding", "full"] * 2


def test_hybrid_model_builds_two_kv_cache_groups():
    engine = hybrid_engine("hybrid:2", window=32)
    kinds = [g.kind for g in engine.kv.groups]
    assert kinds == ["full", "sliding"]
    assert engine.kv.groups[0].layer_ids == (1, 3)
    assert engine.kv.groups[1].layer_ids == (0, 2)
    assert engine.kv.groups[1].window == 32


def test_sliding_window_memory_stays_bounded_while_full_attention_grows():
    """The whole reason hybrid models exist, measured.

    A sliding-window group recycles blocks that scroll out of the window, so its
    live block count plateaus. A global group's grows with the sequence.
    """
    window, block_size = 32, 16
    sliding = hybrid_engine("sliding", window=window, block_size=block_size)
    full = hybrid_engine("full", window=window, block_size=block_size)

    prompt = [5] * 32
    params = SamplingParams(max_tokens=160, ignore_eos=True)
    sliding.generate([prompt], params)
    full.generate([prompt], params)

    # Peak utilisation is measured over the whole run, before any final free.
    sliding_peak = sliding.stats.peak_block_utilisation * (sliding.config.cache.num_blocks - 1)
    full_peak = full.stats.peak_block_utilisation * (full.config.cache.num_blocks - 1)

    # 192 tokens at block_size 16 is 12 blocks for global attention; the sliding
    # group only ever needs about window/block_size + 1 = 3.
    assert full_peak >= 12
    assert sliding_peak <= 4
    assert sliding_peak < full_peak / 3


def test_sliding_window_recycled_blocks_return_to_the_pool():
    engine = hybrid_engine("sliding", window=32, num_blocks=32, block_size=16)
    # Far more tokens than the pool could hold without recycling: 32 blocks of
    # 16 tokens is 512 token-slots, and this run needs 232 per layer group with
    # no recycling plus room for the prompt.
    engine.generate([[3] * 32], SamplingParams(max_tokens=200, ignore_eos=True))
    assert engine.stats.peak_block_utilisation < 0.25
    assert engine.scheduler.num_preemptions == 0
