"""N-gram (prompt-lookup) speculative proposer.

Draft tokens come from the request's own context instead of a draft model: find
the most recent earlier occurrence of the last ``n`` tokens and propose whatever
followed it. No extra weights, no extra forward pass, and on the workloads where
the output copies the input - RAG, code editing, summarisation, agentic tool
loops that echo state - the acceptance rate is high.

The proposal is only a *guess*. Correctness comes from the verification step
(``kvforge.spec.rejection``), which is distribution-preserving regardless of how
bad the guess is. A poor proposer costs throughput, never quality.
"""

from __future__ import annotations

import torch


class NgramProposer:
    def __init__(self, min_n: int = 2, max_n: int = 4, num_speculative_tokens: int = 4) -> None:
        assert 1 <= min_n <= max_n
        self.min_n = min_n
        self.max_n = max_n
        self.k = num_speculative_tokens

    def propose(self, token_ids: list[int]) -> list[int]:
        """Propose up to ``k`` continuation tokens, longest matching n-gram first."""
        length = len(token_ids)
        if length < self.min_n + 1:
            return []
        tokens = torch.tensor(token_ids, dtype=torch.long)
        for n in range(min(self.max_n, length - 1), self.min_n - 1, -1):
            suffix = tokens[-n:]
            # All length-n windows except the trailing suffix itself.
            windows = tokens.unfold(0, n, 1)[:-1]
            if windows.numel() == 0:
                continue
            matches = (windows == suffix).all(dim=1).nonzero(as_tuple=True)[0]
            if matches.numel() == 0:
                continue
            # Most recent match: locality beats the first occurrence in practice.
            start = int(matches[-1]) + n
            draft = token_ids[start : start + self.k]
            if draft:
                return draft
        return []
