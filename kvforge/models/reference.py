"""Dense, cache-free reference implementation of the same model.

Every optimisation in this repository - paging, block reuse, prefix caching,
chunked prefill, preemption, speculation - is supposed to be *invisible in the
output*. This module is how that claim gets tested: it recomputes the whole
sequence from scratch with textbook dense attention, sharing only the weights.

It is deliberately O(n^2) and holds no state. Its job is to be obviously
correct, not fast.
"""

from __future__ import annotations

import torch

from kvforge.layers.attention import dense_causal_attention
from kvforge.layers.moe import SparseMoEBlock
from kvforge.models.transformer import KVForgeTransformer


@torch.inference_mode()
def dense_forward(model: KVForgeTransformer, token_ids: list[int]) -> torch.Tensor:
    """Run the full sequence with no KV cache. Returns hidden states ``[T, H]``."""
    device = model.embed_tokens.weight.device
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.arange(len(token_ids), dtype=torch.long, device=device)
    hidden = model.embed_tokens(ids)
    num_tokens = hidden.shape[0]

    for layer in model.layers:
        attn = layer.self_attn
        x = layer.input_layernorm(hidden)
        qkv = attn.qkv_proj(x)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = q.view(num_tokens, attn.num_heads, attn.head_dim)
        k = k.view(num_tokens, attn.num_kv_heads, attn.head_dim)
        v = v.view(num_tokens, attn.num_kv_heads, attn.head_dim)
        q, k = model.rope(positions, q, k)
        o = dense_causal_attention(q, k, v, scale=attn.scale, window=attn.window)
        hidden = hidden + attn.o_proj(o.reshape(num_tokens, -1))

        y = layer.post_attention_layernorm(hidden)
        if isinstance(layer.mlp, SparseMoEBlock):
            y = layer.mlp(y, mode="dense")
        else:
            y = layer.mlp(y)
        hidden = hidden + y

    return model.norm(hidden)


@torch.inference_mode()
def dense_generate(
    model: KVForgeTransformer, prompt_token_ids: list[int], max_tokens: int
) -> list[int]:
    """Greedy generation with no cache at all: recompute everything, every step."""
    ids = list(prompt_token_ids)
    out: list[int] = []
    for _ in range(max_tokens):
        hidden = dense_forward(model, ids)
        logits = model.lm_head(hidden[-1])
        token = int(logits.argmax())
        ids.append(token)
        out.append(token)
    return out
