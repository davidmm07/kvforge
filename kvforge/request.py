"""Per-request state carried through the scheduler and the model runner."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from kvforge.sampling import SamplingParams


class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status >= RequestStatus.FINISHED_STOPPED


@dataclass
class MultiModalInput:
    """A placeholder span in the prompt filled by an encoder's embeddings.

    Only the *identity* of the span matters to the runtime: it becomes an extra
    key in the prefix-cache block hash so that two prompts with identical
    placeholder tokens but different images can never share a cached block.
    """

    #: Content hash of the decoded media (image / audio / video frames).
    mm_hash: str
    #: Token offset of the placeholder span in the prompt.
    offset: int
    #: Number of placeholder tokens the span occupies.
    length: int


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_step: int = 0
    multi_modal_inputs: list[MultiModalInput] = field(default_factory=list)

    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    #: Tokens whose KV entries are already in the cache. Prefix-cache hits and
    #: chunks of a chunked prefill both advance this without a sampled token.
    num_computed_tokens: int = 0
    #: Draft tokens proposed for the next step by the speculative proposer.
    spec_token_ids: list[int] = field(default_factory=list)
    #: Block-hash chain, extended lazily as the sequence grows.
    block_hashes: list[int] = field(default_factory=list)

    # Bookkeeping for reporting.
    num_preemptions: int = 0
    num_cached_tokens: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0

    def __post_init__(self) -> None:
        self._all_token_ids = list(self.prompt_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """Length of the sequence, i.e. how many KV slots it needs."""
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self._all_token_ids

    @property
    def num_tokens_with_spec(self) -> int:
        return self.num_tokens + len(self.spec_token_ids)

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        self._all_token_ids.append(token_id)

    def truncate_to(self, num_tokens: int) -> None:
        """Drop tokens past ``num_tokens`` (used for rejected draft tokens)."""
        assert num_tokens <= self.num_tokens
        drop = self.num_tokens - num_tokens
        if drop == 0:
            return
        assert drop <= len(self.output_token_ids)
        del self._all_token_ids[num_tokens:]
        del self.output_token_ids[len(self.output_token_ids) - drop :]

    def reset_for_recompute(self) -> None:
        """Preemption by recomputation: the KV is gone, the tokens are not."""
        self.num_computed_tokens = 0
        self.spec_token_ids.clear()
        self.status = RequestStatus.PREEMPTED
        self.num_preemptions += 1

    def check_stop(self, max_model_len: int) -> bool:
        params = self.sampling_params
        if self.num_output_tokens >= params.max_tokens:
            self.status = RequestStatus.FINISHED_LENGTH
            return True
        if self.num_tokens >= max_model_len:
            self.status = RequestStatus.FINISHED_LENGTH
            return True
        if not params.ignore_eos and self.output_token_ids:
            last = self.output_token_ids[-1]
            if params.eos_token_id is not None and last == params.eos_token_id:
                self.status = RequestStatus.FINISHED_STOPPED
                return True
            if last in params.stop_token_ids:
                self.status = RequestStatus.FINISHED_STOPPED
                return True
        return False
