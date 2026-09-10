"""Turns scheduler decisions into tensors and runs the forward pass.

This is the layer where a serving engine either wins or loses on decode latency.
At batch size 1, decode is memory-bound and the actual GEMMs take microseconds,
so any per-request Python work in this file lands directly on inter-token
latency. The build below is therefore linear in *scheduled tokens*, never in
sequence length: slot mappings and block tables are sliced, not rebuilt.
"""

from __future__ import annotations

import torch

from kvforge.config import EngineConfig
from kvforge.core.batch import AttentionMetadata, GroupMetadata, ModelInput
from kvforge.core.scheduler import SchedulerOutput
from kvforge.memory.manager import KVCacheManager, cdiv
from kvforge.models.transformer import KVForgeTransformer


class ModelRunner:
    def __init__(self, config: EngineConfig, kv_manager: KVCacheManager) -> None:
        self.config = config
        self.kv = kv_manager
        self.device = config.device
        self.block_size = config.cache.block_size
        self.model = KVForgeTransformer(config.model, kv_manager.groups).to(config.device)
        self.model.eval()
        self.kv_caches = self._allocate_kv_caches()

    # ------------------------------------------------------------------ setup

    def _allocate_kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        m, c = self.config.model, self.config.cache
        shape = (c.num_blocks, c.block_size, m.num_kv_heads, m.head_dim)
        caches = []
        for _ in range(m.num_layers):
            k = torch.zeros(shape, dtype=m.dtype, device=self.device)
            v = torch.zeros(shape, dtype=m.dtype, device=self.device)
            caches.append((k, v))
        return caches

    def kv_cache_bytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.kv_caches)

    # ------------------------------------------------------------------ input

    def prepare_input(self, out: SchedulerOutput) -> tuple[ModelInput, list[tuple[str, int]]]:
        """Build the flat batch. Returns the input and the per-request logits plan."""
        input_ids: list[int] = []
        positions: list[int] = []
        query_start_loc: list[int] = [0]
        seq_lens: list[int] = []
        logits_indices: list[int] = []
        logits_plan: list[tuple[str, int]] = []

        num_groups = self.kv.num_groups
        slot_mappings: list[list[int]] = [[] for _ in range(num_groups)]
        block_tables: list[list[list[int]]] = [[] for _ in range(num_groups)]
        kv_offsets: list[list[int]] = [[] for _ in range(num_groups)]

        for req in out.scheduled:
            rid = req.request_id
            num_new = out.num_scheduled_tokens[rid]
            start = req.num_computed_tokens
            end = start + num_new

            # Token ids for [start, end), which may straddle the boundary
            # between committed tokens and speculative drafts.
            committed = req.all_token_ids
            if end <= len(committed):
                chunk = committed[start:end]
            else:
                chunk = committed[start:] + req.spec_token_ids[: end - len(committed)]
            assert len(chunk) == num_new
            input_ids.extend(chunk)
            positions.extend(range(start, end))

            token_offset = query_start_loc[-1]
            query_start_loc.append(token_offset + num_new)
            seq_lens.append(end)

            # Logits are only needed where a token will actually be sampled: the
            # last position of a completed prompt, or every draft position of a
            # speculative verify. A mid-prompt chunk needs none, which is what
            # makes chunked prefill cheap on the lm_head.
            if end == req.num_tokens_with_spec:
                num_logits = 1 + out.num_draft_tokens.get(rid, 0)
                logits_indices.extend(range(token_offset + num_new - num_logits, token_offset + num_new))
                logits_plan.append((rid, num_logits))

            for gi in range(num_groups):
                slot_mappings[gi].extend(self.kv.slot_indices(rid, gi, start, end))
                first_live = self.kv.first_live_block(rid, gi, start)
                ids = self.kv.block_ids(rid, gi)[first_live : cdiv(end, self.block_size)]
                block_tables[gi].append(ids)
                kv_offsets[gi].append(first_live * self.block_size)

        dev = self.device
        groups_meta = []
        for gi, group in enumerate(self.kv.groups):
            width = max((len(t) for t in block_tables[gi]), default=1)
            padded = [t + [0] * (width - len(t)) for t in block_tables[gi]]
            groups_meta.append(
                GroupMetadata(
                    block_table=torch.tensor(padded, dtype=torch.long, device=dev),
                    kv_offsets=torch.tensor(kv_offsets[gi], dtype=torch.long, device=dev),
                    slot_mapping=torch.tensor(slot_mappings[gi], dtype=torch.long, device=dev),
                    window=group.window,
                )
            )

        meta = AttentionMetadata(
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.long, device=dev),
            seq_lens=torch.tensor(seq_lens, dtype=torch.long, device=dev),
            groups=groups_meta,
            num_tokens=len(input_ids),
            num_seqs=len(seq_lens),
        )
        model_input = ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=dev),
            positions=torch.tensor(positions, dtype=torch.long, device=dev),
            logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=dev),
            attn_metadata=meta,
        )
        return model_input, logits_plan

    # ---------------------------------------------------------------- execute

    @torch.inference_mode()
    def execute(self, model_input: ModelInput) -> torch.Tensor:
        hidden = self.model(
            model_input.input_ids,
            model_input.positions,
            self.kv_caches,
            model_input.attn_metadata,
        )
        if model_input.logits_indices.numel() == 0:
            return hidden.new_zeros((0, self.config.model.vocab_size))
        return self.model.compute_logits(hidden, model_input.logits_indices)
