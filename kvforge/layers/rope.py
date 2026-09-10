"""Rotary position embeddings.

Positions are passed in explicitly rather than derived from the tensor layout.
Under continuous batching a token's position in the flat batch has nothing to do
with its position in its own sequence: token 0 of the batch might be the 900th
token of a resumed prefill. Every position-dependent op in a serving engine has
to take positions as data.
"""

from __future__ import annotations

import torch


class RotaryEmbedding:
    def __init__(
        self,
        head_dim: int,
        max_position: int,
        theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        pos = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def __call__(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to ``[num_tokens, num_heads, head_dim]`` q and k."""
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        q = query * cos + self._rotate_half(query) * sin
        k = key * cos + self._rotate_half(key) * sin
        return q, k
