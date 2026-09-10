"""Sampling parameters and a batched sampler.

The sampler is deliberately vectorised over the batch. Per-request Python loops
in the sampling path are a classic source of decode-time overhead in serving
engines because they run once per token per request, so they sit directly on the
inter-token latency critical path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables
    max_tokens: int = 16
    seed: int | None = None
    ignore_eos: bool = False
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def apply_top_k_top_p(
    logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Mask logits outside the top-k / top-p nucleus.

    ``logits`` is ``[batch, vocab]``, ``top_k`` and ``top_p`` are ``[batch]``.
    Both filters share a single sort, which is the expensive part.
    """
    need_k = bool((top_k > 0).any())
    need_p = bool((top_p < 1.0).any())
    if not need_k and not need_p:
        return logits

    # Ascending sort: the tokens to discard form a prefix under both filters.
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
    vocab = logits.shape[-1]

    if need_k:
        k = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab))
        k = k.clamp(max=vocab)
        rank = torch.arange(vocab, device=logits.device).unsqueeze(0)
        drop_k = rank < (vocab - k).unsqueeze(1)
        sorted_logits = sorted_logits.masked_fill(drop_k, -float("inf"))

    if need_p:
        probs = sorted_logits.softmax(dim=-1)
        cumsum = probs.cumsum(dim=-1)
        drop_p = cumsum <= (1.0 - top_p).unsqueeze(1)
        drop_p[..., -1] = False  # never drop the most likely token
        sorted_logits = sorted_logits.masked_fill(drop_p, -float("inf"))

    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


class Sampler:
    """Turns ``[batch, vocab]`` logits into token ids plus their probabilities.

    The probabilities are returned alongside the tokens because speculative
    decoding needs the target model's distribution to run rejection sampling
    against the draft distribution.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def _tensorise(self, params: list[SamplingParams]) -> tuple[torch.Tensor, ...]:
        temp = torch.tensor(
            [p.temperature for p in params], dtype=torch.float32, device=self.device
        )
        top_k = torch.tensor(
            [p.top_k for p in params], dtype=torch.long, device=self.device
        )
        top_p = torch.tensor(
            [p.top_p for p in params], dtype=torch.float32, device=self.device
        )
        return temp, top_k, top_p

    def compute_probs(
        self, logits: torch.Tensor, params: list[SamplingParams]
    ) -> torch.Tensor:
        temp, top_k, top_p = self._tensorise(params)
        logits = logits.float()
        greedy = temp <= 0
        safe_temp = torch.where(greedy, torch.ones_like(temp), temp)
        logits = logits / safe_temp.unsqueeze(1)
        logits = apply_top_k_top_p(logits, top_k, top_p)
        probs = logits.softmax(dim=-1)
        # A greedy row gets a point mass on its argmax. Rejection sampling then
        # degenerates to an equality check, so one code path serves both modes.
        if bool(greedy.any()):
            onehot = torch.zeros_like(probs)
            onehot.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            probs = torch.where(greedy.unsqueeze(1), onehot, probs)
        return probs

    def sample_from_probs(
        self, probs: torch.Tensor, generators: list[torch.Generator | None]
    ) -> torch.Tensor:
        out = torch.empty(probs.shape[0], dtype=torch.long, device=probs.device)
        # One multinomial call covers every row on the default generator; only
        # explicitly seeded requests pay for a per-row call.
        default_rows = [i for i, g in enumerate(generators) if g is None]
        if default_rows:
            idx = torch.tensor(default_rows, device=probs.device)
            out[idx] = torch.multinomial(probs[idx], 1).squeeze(1)
        for i, gen in enumerate(generators):
            if gen is not None:
                out[i] = torch.multinomial(probs[i], 1, generator=gen).squeeze(0)
        return out

    def __call__(
        self,
        logits: torch.Tensor,
        params: list[SamplingParams],
        generators: list[torch.Generator | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.compute_probs(logits, params)
        if generators is None:
            generators = [None] * len(params)
        tokens = self.sample_from_probs(probs, generators)
        return tokens, probs
