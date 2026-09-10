"""Flat, device-ready description of one forward pass.

Everything the model needs about the batch lives here, already tensorised. The
model never touches ``Request`` objects: keeping Python-side scheduling state out
of the forward pass is what lets the same graph be captured / compiled once and
replayed for every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GroupMetadata:
    """Per-KV-cache-group addressing for one step."""

    #: ``[num_seqs, max_blocks]`` physical block ids. For sliding-window groups
    #: this is already compacted to the live window.
    block_table: torch.Tensor
    #: ``[num_seqs]`` absolute token position of ``block_table[:, 0]`` slot 0.
    #: Always 0 for global groups; advances with the window otherwise.
    kv_offsets: torch.Tensor
    #: ``[num_tokens]`` flat destination slot for every token's K/V.
    slot_mapping: torch.Tensor
    window: int | None = None


@dataclass
class AttentionMetadata:
    #: ``[num_seqs + 1]`` prefix sums of per-sequence query lengths.
    query_start_loc: torch.Tensor
    #: ``[num_seqs]`` context length after this step, per sequence.
    seq_lens: torch.Tensor
    groups: list[GroupMetadata]
    num_tokens: int
    num_seqs: int


@dataclass
class ModelInput:
    input_ids: torch.Tensor  # [num_tokens]
    positions: torch.Tensor  # [num_tokens]
    #: Indices into the flat token batch whose logits we actually need. A
    #: prefill chunk that is not the last chunk of its prompt contributes no
    #: logits at all, so this is usually far smaller than ``num_tokens``.
    logits_indices: torch.Tensor
    attn_metadata: AttentionMetadata
