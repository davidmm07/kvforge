"""Continuous batching scheduler with chunked prefill and preemption.

One step admits as many tokens as the budget allows, mixing three kinds of work
in a single forward pass:

* **decodes** - one query token per running sequence,
* **prefill chunks** - a slice of a prompt, possibly resumed from a previous step,
* **speculative verifies** - ``1 + k`` query tokens for a sequence with drafts.

Two policies matter here and both are about tail latency:

*Chunked prefill.* Without it, one 8k-token prompt occupies a whole step and
every running decode stalls for its duration. Splitting the prompt into budget
sized chunks lets decodes ride along in the same batch, trading a little prefill
throughput for a much flatter inter-token latency distribution.

*Preemption by recomputation.* The KV cache is finite and admission is
optimistic, so a running sequence can fail to get a block. Rather than failing
the request we evict the most recently admitted one, return its blocks, and put
it back at the front of the queue. Its tokens are still there, only its KV is
gone, so it resumes as a (prefix-cached, therefore usually cheap) prefill.
Evicting the newest request keeps the oldest ones making progress, which avoids
the livelock where everyone is repeatedly preempted just before finishing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from kvforge.config import SchedulerConfig
from kvforge.memory.manager import KVCacheManager
from kvforge.request import Request, RequestStatus


@dataclass
class SchedulerOutput:
    #: Requests in batch order: running first (stable), then newly admitted.
    scheduled: list[Request] = field(default_factory=list)
    #: request_id -> number of query tokens scheduled this step.
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    #: request_id -> number of draft tokens included in those query tokens.
    num_draft_tokens: dict[str, int] = field(default_factory=dict)
    #: request_ids preempted this step.
    preempted: list[str] = field(default_factory=list)
    total_tokens: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    def __bool__(self) -> bool:
        return bool(self.scheduled)


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        kv_manager: KVCacheManager,
        num_lookahead_tokens: int = 0,
    ) -> None:
        self.config = config
        self.kv = kv_manager
        self.num_lookahead_tokens = num_lookahead_tokens

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.step_count = 0
        self.num_preemptions = 0

    # ------------------------------------------------------------------ queue

    def add_request(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt_last(self, output: SchedulerOutput) -> Request | None:
        """Evict the most recently admitted running request."""
        if not self.running:
            return None
        victim = self.running.pop()
        self.kv.free(victim)
        victim.reset_for_recompute()
        self.waiting.appendleft(victim)
        output.preempted.append(victim.request_id)
        self.num_preemptions += 1
        return victim

    # --------------------------------------------------------------- schedule

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens

        # --- 1. running requests: decodes and resumed prefill chunks --------
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            num_new = req.num_tokens_with_spec - req.num_computed_tokens
            num_new = min(num_new, budget)
            if num_new <= 0:
                i += 1
                continue

            scheduled_ok = True
            while not self.kv.allocate_slots(req, num_new):
                victim = self._preempt_last(out)
                if victim is None or victim is req:
                    scheduled_ok = False
                    break
            if not scheduled_ok:
                break

            out.scheduled.append(req)
            out.num_scheduled_tokens[req.request_id] = num_new
            out.num_draft_tokens[req.request_id] = len(req.spec_token_ids)
            budget -= num_new
            if req.num_computed_tokens + num_new <= req.num_prompt_tokens:
                out.num_prefill_tokens += num_new
            else:
                out.num_decode_tokens += num_new
            i += 1

        # --- 2. waiting requests: new prefills -----------------------------
        num_seqs = len(self.running)
        while self.waiting and budget > 0 and num_seqs < self.config.max_num_seqs:
            req = self.waiting[0]

            computed_blocks, num_cached = self.kv.get_computed_blocks(req)
            num_new = req.num_tokens - (req.num_computed_tokens + num_cached)
            if num_new <= 0:
                # Fully cached prompts are impossible (the manager holds back the
                # last block), so this means the request is malformed.
                self.waiting.popleft()
                req.status = RequestStatus.FINISHED_ABORTED
                self.finished.append(req)
                continue

            if not self.config.enable_chunked_prefill and num_new > budget:
                break  # wait for a step with room for the whole prompt
            num_new = min(num_new, budget)

            req.num_computed_tokens += num_cached
            if not self.kv.allocate_slots(req, num_new, computed_blocks):
                req.num_computed_tokens -= num_cached
                break

            req.num_cached_tokens = num_cached
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            out.scheduled.append(req)
            out.num_scheduled_tokens[req.request_id] = num_new
            out.num_prefill_tokens += num_new
            budget -= num_new
            num_seqs += 1

        out.total_tokens = sum(out.num_scheduled_tokens.values())
        self.step_count += 1
        return out

    # ----------------------------------------------------------------- update

    def update_from_output(
        self,
        out: SchedulerOutput,
        sampled: dict[str, list[int]],
    ) -> list[Request]:
        """Advance request state after a forward pass.

        ``sampled`` maps request_id to the tokens accepted this step (one for a
        plain decode, one to ``k + 1`` with speculative decoding, none for a
        prefill chunk that did not reach the end of the prompt).
        """
        finished_now: list[Request] = []
        for req in out.scheduled:
            rid = req.request_id
            num_scheduled = out.num_scheduled_tokens[rid]
            num_draft = out.num_draft_tokens.get(rid, 0)
            tokens = sampled.get(rid, [])

            if not tokens:
                # Mid-prompt chunk: KV advanced, no token produced.
                req.num_computed_tokens += num_scheduled
            else:
                # ``tokens`` is the accepted draft prefix plus one token that
                # always comes from the target model, so len(tokens) - 1 drafts
                # were accepted. KV written for rejected draft positions is
                # discarded by simply not advancing past it; those slots get
                # overwritten next step.
                num_accepted = len(tokens) - 1
                req.num_computed_tokens += num_scheduled - (num_draft - num_accepted)
                req.num_draft_tokens += num_draft
                req.num_accepted_tokens += num_accepted
                req.spec_token_ids.clear()
                for token_id in tokens:
                    req.append_output_token(token_id)
                    if req.check_stop(self.config.max_model_len):
                        finished_now.append(req)
                        break

            self.kv.cache_blocks(req)

        if finished_now:
            done = {r.request_id for r in finished_now}
            self.running = [r for r in self.running if r.request_id not in done]
            for req in finished_now:
                self.kv.free(req)
                self.finished.append(req)
        return finished_now
