"""Distribution-preserving verification of draft tokens.

Speculative decoding is only interesting because it is *lossless*: the sequence
of tokens it emits is distributed exactly as if every token had been sampled
from the target model one at a time (Leviathan et al., 2023; Chen et al., 2023).

For each draft token ``x`` with draft probability ``q(x)`` and target probability
``p(x)``:

* accept with probability ``min(1, p(x) / q(x))``;
* on rejection, resample from the normalised residual ``(p - q)_+`` and stop.

If every draft is accepted, one *bonus* token is sampled from the target
distribution at the last position - which is free, because the target model
already computed those logits while verifying. That bonus is where the speedup
comes from: ``k`` drafts cost one forward pass and can yield ``k + 1`` tokens.

The n-gram proposer has no distribution of its own, so ``q`` is a point mass on
the drafted token. Acceptance then simplifies to accepting with probability
``p(x)``, and the residual is ``p`` with ``x`` zeroed out. Greedy decoding falls
out of the same formula: ``p`` is one-hot on the argmax, so a draft is accepted
iff it equals the argmax, and a rejection resamples the argmax deterministically.
One code path, both modes, no special casing.
"""

from __future__ import annotations

import torch


def rejection_sample(
    target_probs: torch.Tensor,
    draft_token_ids: list[int],
    draft_probs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Verify drafts against the target distribution.

    Args:
        target_probs: ``[k + 1, vocab]`` target distributions, one per draft
            position plus one for the bonus token.
        draft_token_ids: the ``k`` proposed tokens.
        draft_probs: ``[k, vocab]`` draft distributions, or ``None`` for a
            proposer that emits point masses (n-gram, prompt lookup).

    Returns:
        Between 1 and ``k + 1`` token ids. The last one always comes from the
        target model, so progress is guaranteed even when every draft is wrong.
    """
    k = len(draft_token_ids)
    assert target_probs.shape[0] == k + 1, "need one distribution per draft plus a bonus"
    accepted: list[int] = []

    for i, draft_id in enumerate(draft_token_ids):
        p = target_probs[i]
        p_x = float(p[draft_id])
        q_x = 1.0 if draft_probs is None else float(draft_probs[i, draft_id])
        threshold = min(1.0, p_x / q_x) if q_x > 0 else 0.0
        u = float(torch.rand(1, generator=generator))
        if u < threshold:
            accepted.append(draft_id)
            continue

        # Rejected: resample from the residual so the overall distribution of
        # the emitted token is still exactly p.
        if draft_probs is None:
            residual = p.clone()
            residual[draft_id] = 0.0
        else:
            residual = (p - draft_probs[i]).clamp_min(0.0)
        total = float(residual.sum())
        if total <= 0.0:
            # p is entirely concentrated on the rejected token: p == q there, so
            # accepting is the distribution-preserving choice.
            accepted.append(draft_id)
            continue
        token = int(torch.multinomial(residual / total, 1, generator=generator))
        return accepted + [token]

    bonus = int(torch.multinomial(target_probs[k], 1, generator=generator))
    return accepted + [bonus]
