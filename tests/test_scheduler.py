"""Scheduling policy: budgets, chunking, admission limits, preemption."""

from kvforge.config import CacheConfig, ModelConfig, SchedulerConfig
from kvforge.core.scheduler import Scheduler
from kvforge.memory.manager import KVCacheManager
from kvforge.request import Request, RequestStatus
from kvforge.sampling import SamplingParams


def build(max_num_batched_tokens=64, max_num_seqs=4, num_blocks=64, chunked=True, block_size=8):
    model = ModelConfig(num_layers=2, hidden_size=64, num_heads=4, num_kv_heads=2)
    kv = KVCacheManager(model, CacheConfig(block_size, num_blocks, enable_prefix_caching=False))
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        # Without chunked prefill a prompt must fit in one batch, so the model
        # length cannot exceed the token budget.
        max_model_len=512 if chunked else max_num_batched_tokens,
        enable_chunked_prefill=chunked,
    )
    return Scheduler(config, kv), kv


def add(scheduler, rid, num_tokens, max_tokens=8):
    req = Request(
        request_id=rid,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
    scheduler.add_request(req)
    return req


def test_token_budget_is_respected():
    scheduler, _ = build(max_num_batched_tokens=64)
    add(scheduler, "a", 50)
    add(scheduler, "b", 50)
    out = scheduler.schedule()
    assert out.total_tokens <= 64
    assert out.num_scheduled_tokens["a"] == 50
    assert out.num_scheduled_tokens["b"] == 14  # chunked to fit the budget


def test_chunked_prefill_splits_a_long_prompt_across_steps():
    scheduler, _ = build(max_num_batched_tokens=16)
    req = add(scheduler, "a", 40)
    chunks = []
    for _ in range(3):
        out = scheduler.schedule()
        chunks.append(out.num_scheduled_tokens["a"])
        scheduler.update_from_output(out, {})
    assert chunks == [16, 16, 8]
    assert req.num_computed_tokens == 40


def test_without_chunked_prefill_a_prompt_waits_for_a_whole_step():
    scheduler, _ = build(max_num_batched_tokens=64, chunked=False)
    add(scheduler, "a", 50)
    add(scheduler, "b", 50)
    out = scheduler.schedule()
    # 'b' does not fit in the remaining 14 tokens and cannot be split, so it waits.
    assert list(out.num_scheduled_tokens) == ["a"]
    assert len(scheduler.waiting) == 1


def test_decodes_are_prioritised_over_new_prefills():
    """Running sequences get their token before anyone new is admitted."""
    scheduler, _ = build(max_num_batched_tokens=20)
    add(scheduler, "a", 10)
    out = scheduler.schedule()
    scheduler.update_from_output(out, {"a": [1]})

    add(scheduler, "b", 40)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["a"] == 1  # decode first
    assert out.num_scheduled_tokens["b"] == 19  # then whatever budget is left


def test_max_num_seqs_caps_concurrency():
    scheduler, _ = build(max_num_batched_tokens=512, max_num_seqs=2)
    for i in range(5):
        add(scheduler, f"r{i}", 10)
    out = scheduler.schedule()
    assert len(out.scheduled) == 2
    assert len(scheduler.waiting) == 3


def test_preemption_evicts_the_newest_and_requeues_it_at_the_front():
    # 6 usable blocks of 8 tokens: three 16-token prompts fit exactly, and the
    # first decode step then has nowhere to grow.
    scheduler, kv = build(max_num_batched_tokens=512, max_num_seqs=8, num_blocks=7)
    for i in range(4):
        add(scheduler, f"r{i}", 16)

    out = scheduler.schedule()
    admitted = [r.request_id for r in out.scheduled]
    assert len(admitted) < 4, "expected the pool to run out of blocks"

    # Drive decodes until memory pressure forces an eviction.
    preempted = []
    for _ in range(12):
        scheduler.update_from_output(out, {r.request_id: [1] for r in out.scheduled})
        out = scheduler.schedule()
        preempted.extend(out.preempted)
        if not out:
            break

    assert preempted, "no preemption happened under a deliberately tiny cache"
    victim_id = preempted[0]
    victim = next(r for r in scheduler.finished + list(scheduler.waiting) + scheduler.running
                  if r.request_id == victim_id)
    assert victim.num_preemptions >= 1


def test_preempted_request_keeps_its_tokens_and_loses_only_its_kv():
    scheduler, kv = build(max_num_batched_tokens=512, max_num_seqs=8, num_blocks=7)
    req = add(scheduler, "a", 16)
    out = scheduler.schedule()
    scheduler.update_from_output(out, {"a": [42]})
    assert req.num_computed_tokens == 16

    tokens_before = list(req.all_token_ids)
    scheduler._preempt_last(out)
    assert req.status == RequestStatus.PREEMPTED
    assert req.num_computed_tokens == 0
    assert req.all_token_ids == tokens_before  # tokens survive, KV does not
    assert "a" not in kv.req_blocks
    assert scheduler.waiting[0] is req  # front of the queue, not the back


def test_finished_requests_release_their_blocks():
    scheduler, kv = build()
    add(scheduler, "a", 16, max_tokens=1)
    out = scheduler.schedule()
    free_before = kv.pool.num_free_blocks
    finished = scheduler.update_from_output(out, {"a": [5]})
    assert [r.request_id for r in finished] == ["a"]
    assert kv.pool.num_free_blocks > free_before
    assert "a" not in kv.req_blocks
