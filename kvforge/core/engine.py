"""The engine loop: schedule, run, sample, verify, repeat."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import torch

from kvforge.config import EngineConfig
from kvforge.core.model_runner import ModelRunner
from kvforge.core.scheduler import Scheduler
from kvforge.memory.manager import KVCacheManager
from kvforge.request import MultiModalInput, Request, RequestStatus
from kvforge.sampling import Sampler, SamplingParams
from kvforge.spec.ngram import NgramProposer
from kvforge.spec.rejection import rejection_sample


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    finish_reason: str
    num_cached_tokens: int = 0
    num_preemptions: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    ttft: float = 0.0
    latency: float = 0.0


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    generated_tokens: int = 0
    preemptions: int = 0
    elapsed: float = 0.0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_hit_tokens: int = 0
    prompt_tokens: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    peak_block_utilisation: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def output_throughput(self) -> float:
        return self.generated_tokens / self.elapsed if self.elapsed else 0.0

    @property
    def mean_batch_size(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0

    @property
    def prefix_cache_token_hit_rate(self) -> float:
        return self.prefix_cache_hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Generated tokens per model forward pass. > 1 means speculation paid off."""
        return self.generated_tokens / self.steps if self.steps else 0.0


class LLMEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.kv = KVCacheManager(config.model, config.cache)
        self.runner = ModelRunner(config, self.kv)
        self.scheduler = Scheduler(config.scheduler, self.kv)
        self.sampler = Sampler(config.device)
        self.proposer = (
            NgramProposer(
                config.speculative.ngram_min,
                config.speculative.ngram_max,
                config.speculative.num_speculative_tokens,
            )
            if config.speculative is not None
            else None
        )

        self.requests: dict[str, Request] = {}
        self.generators: dict[str, torch.Generator] = {}
        self._t_arrival: dict[str, float] = {}
        self._t_first: dict[str, float] = {}
        self._counter = itertools.count()
        self.stats = EngineStats()

    # ------------------------------------------------------------------ input

    def _validate_prompt(self, prompt_token_ids: list[int]) -> None:
        """Reject bad prompts at the API boundary, where the context still exists.

        Every check here guards a failure that would otherwise surface deep in
        the runtime as a confusing error. An out-of-range token id is only
        noticed by ``nn.Embedding``, several frames down, as a bare
        ``IndexError: index out of range in self`` that names neither the token
        nor the request. A prompt too large to schedule is worse: the request is
        silently dropped and ``generate`` dies on a ``KeyError`` for its own id.
        These scans cost nothing next to prefill and turn both into errors that
        say what to fix.
        """
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids is empty; a request needs at least one token")

        vocab_size = self.config.model.vocab_size
        lowest, highest = min(prompt_token_ids), max(prompt_token_ids)
        if lowest < 0 or highest >= vocab_size:
            bad = highest if highest >= vocab_size else lowest
            raise ValueError(
                f"token id {bad} is outside the model's vocabulary "
                f"[0, {vocab_size}); check that the tokenizer and "
                f"ModelConfig.vocab_size agree"
            )

        num_tokens = len(prompt_token_ids)
        max_len = self.config.scheduler.max_model_len
        if num_tokens > max_len:
            raise ValueError(
                f"prompt has {num_tokens} tokens but max_model_len is {max_len}; "
                f"leave room for at least one generated token"
            )
        # Block 0 is the reserved null block, so it never holds real KV.
        usable_slots = (self.config.cache.num_blocks - 1) * self.config.cache.block_size
        if num_tokens > usable_slots:
            raise ValueError(
                f"prompt has {num_tokens} tokens but the KV cache holds at most "
                f"{usable_slots} ({self.config.cache.num_blocks - 1} blocks x "
                f"{self.config.cache.block_size}); it could never be scheduled. "
                f"Raise CacheConfig.num_blocks or shorten the prompt"
            )

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        multi_modal_inputs: list[MultiModalInput] | None = None,
    ) -> str:
        self._validate_prompt(prompt_token_ids)
        rid = request_id or f"req-{next(self._counter)}"
        params = sampling_params or SamplingParams()
        req = Request(
            request_id=rid,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=params,
            multi_modal_inputs=multi_modal_inputs or [],
        )
        if params.seed is not None:
            gen = torch.Generator(device=self.config.device)
            gen.manual_seed(params.seed)
            self.generators[rid] = gen
        self.requests[rid] = req
        self._t_arrival[rid] = time.perf_counter()
        self.stats.prompt_tokens += len(prompt_token_ids)
        self.scheduler.add_request(req)
        return rid

    # ------------------------------------------------------------------- step

    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        if not sched:
            return []

        t0 = time.perf_counter()
        model_input, logits_plan = self.runner.prepare_input(sched)
        logits = self.runner.execute(model_input)
        sampled = self._sample(logits, logits_plan)
        finished = self.scheduler.update_from_output(sched, sampled)
        self._record_step(sched, sampled, time.perf_counter() - t0)

        self._propose_drafts()
        return [self._make_output(req) for req in finished]

    def _sample(
        self, logits: torch.Tensor, plan: list[tuple[str, int]]
    ) -> dict[str, list[int]]:
        sampled: dict[str, list[int]] = {}
        # Requests needing a single token are sampled together: one softmax and
        # one multinomial for the whole decode batch instead of N of each.
        simple_rows, simple_ids = [], []
        spec_jobs = []
        offset = 0
        for rid, num_logits in plan:
            rows = logits[offset : offset + num_logits]
            offset += num_logits
            if num_logits == 1:
                simple_rows.append(rows[0])
                simple_ids.append(rid)
            else:
                spec_jobs.append((rid, rows))

        if simple_rows:
            batch = torch.stack(simple_rows)
            params = [self.requests[r].sampling_params for r in simple_ids]
            gens = [self.generators.get(r) for r in simple_ids]
            tokens, _ = self.sampler(batch, params, gens)
            for rid, token in zip(simple_ids, tokens.tolist()):
                sampled[rid] = [token]

        for rid, rows in spec_jobs:
            req = self.requests[rid]
            probs = self.sampler.compute_probs(rows, [req.sampling_params] * rows.shape[0])
            sampled[rid] = rejection_sample(
                probs, req.spec_token_ids, generator=self.generators.get(rid)
            )
        return sampled

    def _propose_drafts(self) -> None:
        if self.proposer is None:
            return
        for req in self.scheduler.running:
            # Only sequences in a pure decode state: a request still working
            # through a chunked prefill has no "next token" to speculate past.
            if req.num_computed_tokens != req.num_tokens - 1:
                continue
            budget = req.sampling_params.max_tokens - req.num_output_tokens - 1
            budget = min(budget, self.config.scheduler.max_model_len - req.num_tokens - 1)
            if budget <= 0:
                continue
            req.spec_token_ids = self.proposer.propose(req.all_token_ids)[:budget]

    def _record_step(self, sched, sampled: dict[str, list[int]], dt: float) -> None:
        s = self.stats
        s.steps += 1
        s.elapsed += dt
        s.prefill_tokens += sched.num_prefill_tokens
        s.decode_tokens += sched.num_decode_tokens
        s.generated_tokens += sum(len(v) for v in sampled.values())
        s.batch_sizes.append(len(sched.scheduled))
        s.peak_block_utilisation = max(s.peak_block_utilisation, self.kv.pool.utilisation())
        s.preemptions = self.scheduler.num_preemptions
        now = time.perf_counter()
        for rid in sampled:
            self._t_first.setdefault(rid, now)

    def _make_output(self, req: Request) -> RequestOutput:
        rid = req.request_id
        now = time.perf_counter()
        reason = "length" if req.status == RequestStatus.FINISHED_LENGTH else "stop"
        self.stats.prefix_cache_hit_tokens += req.num_cached_tokens
        self.stats.draft_tokens += req.num_draft_tokens
        self.stats.accepted_tokens += req.num_accepted_tokens
        self.stats.prefix_cache_queries = self.kv.pool.stats.queries
        self.stats.prefix_cache_hits = self.kv.pool.stats.hits
        return RequestOutput(
            request_id=rid,
            prompt_token_ids=req.prompt_token_ids,
            output_token_ids=list(req.output_token_ids),
            finish_reason=reason,
            num_cached_tokens=req.num_cached_tokens,
            num_preemptions=req.num_preemptions,
            num_draft_tokens=req.num_draft_tokens,
            num_accepted_tokens=req.num_accepted_tokens,
            ttft=self._t_first.get(rid, now) - self._t_arrival[rid],
            latency=now - self._t_arrival[rid],
        )

    # ------------------------------------------------------------------ drive

    def run_to_completion(self, max_steps: int = 1_000_000) -> list[RequestOutput]:
        outputs: list[RequestOutput] = []
        for _ in range(max_steps):
            if not self.scheduler.has_work:
                break
            outputs.extend(self.step())
        return outputs

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[RequestOutput]:
        """Offline batch entry point, ordered to match ``prompts``."""
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        # Validate the whole batch before admitting any of it, so one bad prompt
        # cannot leave the engine holding half a batch it will never finish.
        for prompt in prompts:
            self._validate_prompt(prompt)
        ids = [
            self.add_request(p, params) for p, params in zip(prompts, sampling_params)
        ]
        outputs = {o.request_id: o for o in self.run_to_completion()}
        return [outputs[i] for i in ids]
