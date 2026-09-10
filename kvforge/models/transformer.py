"""A decoder-only transformer wired for paged, continuously batched inference.

Architecture is standard (pre-norm, RMSNorm, GQA, RoPE, SwiGLU) so that the
interesting part is the plumbing: every layer knows which KV cache group it
belongs to and reads its addressing from ``AttentionMetadata``. Adding a new
architecture means adding a layer type, not touching the runtime.

Weights are random by design. This project is about the inference runtime, and a
random-weight model of the right *shape* exercises every code path that a real
checkpoint would, without a multi-gigabyte download. Correctness is established
by parity against a dense reference implementation of the same weights, which is
a strictly stronger check than eyeballing generated text.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kvforge.config import ModelConfig
from kvforge.core.batch import AttentionMetadata
from kvforge.layers.attention import paged_attention, store_kv
from kvforge.layers.moe import SparseMoEBlock
from kvforge.layers.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=torch.float32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, dtype=torch.float32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PagedAttentionLayer(nn.Module):
    """Grouped-query attention reading and writing a paged KV cache."""

    def __init__(self, config: ModelConfig, layer_idx: int, group_idx: int, window: int | None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.group_idx = group_idx
        self.window = window
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        q_size = config.num_heads * config.head_dim
        kv_size = config.num_kv_heads * config.head_dim
        # Fused QKV: one GEMM instead of three. At decode batch sizes these
        # projections are memory-bound, so launch count matters more than FLOPs.
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=False, dtype=config.dtype)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False, dtype=config.dtype)
        self.q_size, self.kv_size = q_size, kv_size

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        qkv = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        q, k = rope(positions, q, k)

        group = meta.groups[self.group_idx]
        k_cache, v_cache = kv_cache
        store_kv(k, v, k_cache, v_cache, group.slot_mapping)

        attn = paged_attention(
            q,
            k_cache,
            v_cache,
            block_table=group.block_table,
            kv_offsets=group.kv_offsets,
            seq_lens=meta.seq_lens,
            query_start_loc=meta.query_start_loc,
            scale=self.scale,
            window=self.window,
        )
        return self.o_proj(attn.reshape(num_tokens, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, group_idx: int, window: int | None) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
        self.self_attn = PagedAttentionLayer(config, layer_idx, group_idx, window)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
        if config.is_moe_layer(layer_idx):
            assert config.moe is not None
            self.mlp: nn.Module = SparseMoEBlock(
                config.hidden_size,
                config.ffn_hidden_size,
                config.moe.num_experts,
                config.moe.top_k,
                config.moe.renormalize,
                config.dtype,
            )
        else:
            self.mlp = SwiGLU(config.hidden_size, config.ffn_hidden_size, config.dtype)

    def forward(self, hidden, positions, rope, kv_cache, meta):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, rope, kv_cache, meta)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class KVForgeTransformer(nn.Module):
    def __init__(self, config: ModelConfig, groups) -> None:
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.groups = groups
        kinds = config.layer_kinds()
        group_of_layer: dict[int, int] = {}
        window_of_layer: dict[int, int | None] = {}
        for gi, group in enumerate(groups):
            for layer_id in group.layer_ids:
                group_of_layer[layer_id] = gi
                window_of_layer[layer_id] = group.window

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, i, group_of_layer[i], window_of_layer[i])
                for i in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta, config.dtype
        )
        self.layer_kinds = kinds
        # Random init at ~1/sqrt(fan_in) keeps activations O(1) through the
        # stack; without it a 24-layer random model saturates into one token.
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=p.shape[-1] ** -0.5)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, positions, self.rope, kv_caches[i], meta)
        return self.norm(hidden)

    def compute_logits(self, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden[indices])
