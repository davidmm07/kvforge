"""Mixture-of-experts FFN with a grouped (sorted) dispatch path.

The naive way to run top-k MoE is to evaluate every expert on every token and
mask. It is trivially correct and trivially wasteful: with 8 experts and top-2
it does 4x the necessary FLOPs, and the ratio grows with the expert count.

The grouped path sorts the ``num_tokens * top_k`` (token, expert) pairs by
expert so that each expert's inputs are contiguous, then runs one GEMM per
expert over just its own rows. That is the CPU/PyTorch shape of what a fused
grouped-GEMM (or the MegaBlocks block-sparse formulation) does on GPU: the
sorting and the segment offsets are the same, only the inner GEMM differs.

``benchmarks/bench_moe.py`` measures the gap, and ``tests/test_moe.py`` pins the
two paths to the same numerics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SparseMoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        top_k: int,
        renormalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize

        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        # Experts are stacked into single tensors so a dispatch is a slice, not
        # a Python attribute lookup per expert.
        scale = hidden_size**-0.5
        self.w_gate = nn.Parameter(torch.randn(num_experts, hidden_size, ffn_hidden_size, dtype=dtype) * scale)
        self.w_up = nn.Parameter(torch.randn(num_experts, hidden_size, ffn_hidden_size, dtype=dtype) * scale)
        self.w_down = nn.Parameter(torch.randn(num_experts, ffn_hidden_size, hidden_size, dtype=dtype) * (ffn_hidden_size**-0.5))

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k router. Returns ``(weights, expert_ids)`` of shape ``[T, k]``."""
        logits = self.gate(x)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        weights, expert_ids = probs.topk(self.top_k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(x.dtype), expert_ids

    def _expert_mlp(self, x: torch.Tensor, e: int) -> torch.Tensor:
        return (F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]

    def forward(self, x: torch.Tensor, mode: str = "grouped") -> torch.Tensor:
        weights, expert_ids = self.route(x)
        if mode == "dense":
            return self._forward_dense(x, weights, expert_ids)
        return self._forward_grouped(x, weights, expert_ids)

    def _forward_grouped(
        self, x: torch.Tensor, weights: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        flat_experts = expert_ids.reshape(-1)
        flat_weights = weights.reshape(-1)

        # Sort the (token, expert) pairs by expert so each expert owns one
        # contiguous segment; `counts` gives the segment boundaries.
        order = torch.argsort(flat_experts, stable=True)
        token_of_pair = order // self.top_k
        counts = torch.bincount(flat_experts, minlength=self.num_experts).tolist()

        out = torch.zeros_like(x)
        start = 0
        for e, count in enumerate(counts):
            if count == 0:
                continue  # expert received no tokens this step
            end = start + count
            pairs = order[start:end]
            rows = token_of_pair[start:end]
            y = self._expert_mlp(x[rows], e)
            out.index_add_(0, rows, y * flat_weights[pairs].unsqueeze(1))
            start = end
        assert start == num_tokens * self.top_k
        return out

    def _forward_dense(
        self, x: torch.Tensor, weights: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Every expert on every token, then mask. Baseline / parity oracle."""
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            y = self._expert_mlp(x, e)
            # Weight for expert e per token: 0 if it was not selected.
            w = (weights * (expert_ids == e)).sum(dim=-1, keepdim=True)
            out = out + y * w
        return out
