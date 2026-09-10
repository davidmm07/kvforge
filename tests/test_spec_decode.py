"""Speculative decoding: proposal, and the losslessness of verification."""

import torch

from kvforge.spec.ngram import NgramProposer
from kvforge.spec.rejection import rejection_sample


# ----------------------------------------------------------------- proposer


def test_ngram_proposes_the_continuation_of_the_latest_match():
    proposer = NgramProposer(min_n=2, max_n=4, num_speculative_tokens=3)
    tokens = [1, 2, 3, 4, 5, 9, 9, 1, 2]
    # The suffix [1, 2] appeared at index 0 and was followed by 3, 4, 5.
    assert proposer.propose(tokens) == [3, 4, 5]


def test_ngram_prefers_the_longest_matching_ngram():
    proposer = NgramProposer(min_n=2, max_n=4, num_speculative_tokens=2)
    # Suffix [7, 8, 1, 2]: the 4-gram matches at index 0 (-> 50, 51), while the
    # shorter 2-gram [1, 2] also matches later (-> 60). Longest wins.
    tokens = [7, 8, 1, 2, 50, 51, 1, 2, 60, 61, 7, 8, 1, 2]
    assert proposer.propose(tokens) == [50, 51]


def test_ngram_returns_nothing_without_a_match():
    proposer = NgramProposer(min_n=2, max_n=4, num_speculative_tokens=4)
    assert proposer.propose([1, 2, 3, 4, 5]) == []
    assert proposer.propose([1]) == []


def test_ngram_never_proposes_more_than_k():
    proposer = NgramProposer(min_n=2, max_n=3, num_speculative_tokens=2)
    assert len(proposer.propose([1, 2, 3, 4, 5, 6, 1, 2])) == 2


# -------------------------------------------------------------- verification


def one_hot(index, vocab=8):
    p = torch.zeros(vocab)
    p[index] = 1.0
    return p


def test_greedy_accepts_a_correct_draft_and_takes_the_bonus():
    """All drafts match the argmax: k drafts + 1 bonus from one forward pass."""
    target = torch.stack([one_hot(3), one_hot(4), one_hot(5)])
    assert rejection_sample(target, [3, 4]) == [3, 4, 5]


def test_greedy_rejects_at_the_first_mismatch_and_corrects_it():
    target = torch.stack([one_hot(3, 16), one_hot(9, 16), one_hot(5, 16)])
    # Second draft is wrong, so it is replaced by the target's token and the
    # rest of the drafts are discarded.
    assert rejection_sample(target, [3, 7]) == [3, 9]


def test_a_token_is_always_emitted_even_when_every_draft_is_wrong():
    target = torch.stack([one_hot(1, 16), one_hot(2, 16), one_hot(3, 16)])
    out = rejection_sample(target, [11, 12])
    assert out == [1]  # forward progress is guaranteed


def test_rejection_sampling_preserves_the_target_distribution():
    """The statistical heart of speculative decoding.

    Emitted tokens must be distributed exactly as the target model's own
    samples, no matter what the draft proposes. Anything else silently changes
    what the served model produces.
    """
    torch.manual_seed(0)
    vocab = 6
    target_dist = torch.tensor([0.30, 0.25, 0.20, 0.13, 0.07, 0.05])
    draft_token = 4  # a deliberately unlikely proposal

    gen = torch.Generator().manual_seed(7)
    counts = torch.zeros(vocab)
    trials = 40_000
    for _ in range(trials):
        # One draft position; the bonus row is irrelevant when it is rejected,
        # but must be present, so reuse the same distribution.
        target = torch.stack([target_dist, target_dist])
        out = rejection_sample(target, [draft_token], generator=gen)
        counts[out[0]] += 1

    empirical = counts / trials
    tv_distance = 0.5 * (empirical - target_dist).abs().sum()
    assert tv_distance < 0.01, f"distribution drifted: TV={tv_distance:.4f}"


def test_rejection_sampling_with_an_explicit_draft_distribution():
    """The general (draft-model) case, not just point-mass proposers."""
    torch.manual_seed(1)
    vocab = 5
    target_dist = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
    draft_dist = torch.tensor([0.1, 0.1, 0.3, 0.3, 0.2])

    gen = torch.Generator().manual_seed(3)
    counts = torch.zeros(vocab)
    trials = 40_000
    for _ in range(trials):
        # Sample the draft token from the draft distribution, as a real draft
        # model would; only then is the acceptance test unbiased.
        draft_token = int(torch.multinomial(draft_dist, 1, generator=gen))
        target = torch.stack([target_dist, target_dist])
        out = rejection_sample(
            target, [draft_token], draft_probs=draft_dist.unsqueeze(0), generator=gen
        )
        counts[out[0]] += 1

    empirical = counts / trials
    tv_distance = 0.5 * (empirical - target_dist).abs().sum()
    assert tv_distance < 0.015, f"distribution drifted: TV={tv_distance:.4f}"


def test_acceptance_count_never_exceeds_the_draft_length():
    torch.manual_seed(2)
    target = torch.rand(5, 12).softmax(-1)
    for _ in range(50):
        out = rejection_sample(target, [1, 2, 3, 4], generator=torch.Generator().manual_seed(_))
        assert 1 <= len(out) <= 5
