"""Engine configuration objects.

The split mirrors production engines (vLLM / SGLang): the *model* shape, the
*cache* geometry, the *scheduler* policy and the *speculative decoding* policy
are independent concerns that get combined into a single ``EngineConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

AttentionKind = Literal["full", "sliding"]


@dataclass
class MoEConfig:
    """Mixture-of-experts settings for the FFN sub-layer."""

    num_experts: int = 8
    top_k: int = 2
    #: Layers whose FFN is replaced by an MoE block. ``None`` means every layer.
    layer_indices: tuple[int, ...] | None = None
    #: Normalise the top-k router probabilities so they sum to 1 (Mixtral does).
    renormalize: bool = True


@dataclass
class ModelConfig:
    """Shape of the decoder.

    ``attention_pattern`` controls *hybrid* models:

    ``"full"``
        Every layer is global attention (Llama, Qwen).
    ``"sliding"``
        Every layer is sliding-window attention (Mistral-v0.1).
    ``"hybrid:N"``
        Every ``N``-th layer is global, the rest are sliding window
        (Gemma-2/3, Ministral, Command-A).  This is the case that makes KV
        memory management interesting: the two layer families have different
        per-token memory costs and different lifetimes.
    """

    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int | None = None
    ffn_hidden_size: int = 512
    vocab_size: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    attention_pattern: str = "full"
    sliding_window: int = 256
    moe: MoEConfig | None = None
    dtype: torch.dtype = torch.float32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.head_dim is None:
            assert self.hidden_size % self.num_heads == 0
            self.head_dim = self.hidden_size // self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads (GQA)"
        )
        if self.attention_pattern.startswith("hybrid"):
            _, _, n = self.attention_pattern.partition(":")
            assert n.isdigit() and int(n) >= 1, "use 'hybrid:N', e.g. 'hybrid:4'"

    @property
    def num_query_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def layer_kinds(self) -> list[AttentionKind]:
        """Per-layer attention kind, in layer order."""
        if self.attention_pattern == "full":
            return ["full"] * self.num_layers
        if self.attention_pattern == "sliding":
            return ["sliding"] * self.num_layers
        every = int(self.attention_pattern.split(":")[1])
        # Convention (Gemma-3): the *last* layer of each group of ``every`` is
        # global, so a model always ends on a global layer.
        return [
            "full" if (i + 1) % every == 0 else "sliding"
            for i in range(self.num_layers)
        ]

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.moe is None:
            return False
        if self.moe.layer_indices is None:
            return True
        return layer_idx in self.moe.layer_indices


@dataclass
class CacheConfig:
    """KV cache geometry."""

    block_size: int = 16
    num_blocks: int = 512
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        assert self.block_size > 0
        # Block 0 is reserved as the "null block": sliding-window layers point
        # evicted logical blocks at it so block tables stay dense.
        assert self.num_blocks >= 2


@dataclass
class SchedulerConfig:
    """Continuous-batching policy."""

    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    #: When enabled a prefill may be split across steps so that decodes are not
    #: starved by one long prompt (chunked prefill / piggybacking).
    enable_chunked_prefill: bool = True

    def __post_init__(self) -> None:
        if not self.enable_chunked_prefill:
            assert self.max_num_batched_tokens >= self.max_model_len, (
                "without chunked prefill a prompt must fit in one batch"
            )


@dataclass
class SpeculativeConfig:
    """Speculative decoding policy."""

    method: Literal["ngram"] = "ngram"
    num_speculative_tokens: int = 4
    #: n-gram sizes to try when matching the suffix of the context, longest first.
    ngram_max: int = 4
    ngram_min: int = 2


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculative: SpeculativeConfig | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        hybrid = len(set(self.model.layer_kinds())) > 1
        if hybrid and self.cache.enable_prefix_caching:
            # See docs/DESIGN.md ("Hybrid models and prefix caching"): a full
            # attention hit is only reusable if the sliding-window groups still
            # hold the matching window, which we do not track yet.
            self.cache.enable_prefix_caching = False
