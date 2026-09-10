"""Shared setup for the benchmarks.

The model is small and runs on CPU. That is a deliberate choice: every number
reported here is a *runtime* property - how many tokens got recomputed, how many
forward passes a generation took, how full the KV pool got - and those are
hardware independent. Wall-clock figures are reported too, but the wall-clock
story on CPU with a reference attention kernel is not the interesting one; the
work-avoided figures are.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvforge import (  # noqa: E402
    CacheConfig,
    EngineConfig,
    ModelConfig,
    MoEConfig,
    SchedulerConfig,
    SpeculativeConfig,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BENCH_MODEL = dict(
    hidden_size=384,
    num_layers=6,
    num_heads=6,
    num_kv_heads=2,
    head_dim=64,
    ffn_hidden_size=1024,
    vocab_size=2048,
    seed=7,
)


def build_config(
    *,
    model: dict | None = None,
    block_size: int = 16,
    num_blocks: int = 2048,
    prefix_caching: bool = True,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 2048,
    max_model_len: int = 4096,
    chunked_prefill: bool = True,
    speculative: SpeculativeConfig | None = None,
) -> EngineConfig:
    return EngineConfig(
        model=ModelConfig(**(BENCH_MODEL | (model or {}))),
        cache=CacheConfig(block_size, num_blocks, prefix_caching),
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            enable_chunked_prefill=chunked_prefill,
        ),
        speculative=speculative,
    )


def random_prompt(rng: random.Random, length: int, vocab: int = 2048) -> list[int]:
    return [rng.randrange(vocab) for _ in range(length)]


def chat_workload(
    num_requests: int,
    system_prompt_len: int = 256,
    user_len_range: tuple[int, int] = (16, 64),
    seed: int = 0,
    num_conversations: int = 4,
) -> list[list[int]]:
    """Requests sharing a long system prompt, as a served chat endpoint sees.

    ``num_conversations`` distinct system prompts, each reused by several
    requests: the shape that makes prefix caching worth having.
    """
    rng = random.Random(seed)
    systems = [random_prompt(rng, system_prompt_len) for _ in range(num_conversations)]
    prompts = []
    for i in range(num_requests):
        user = random_prompt(rng, rng.randrange(*user_len_range))
        prompts.append(systems[i % num_conversations] + user)
    return prompts


def repetitive_workload(num_requests: int, seed: int = 0) -> list[list[int]]:
    """Prompts with heavy internal repetition, where n-gram drafting fires.

    Stands in for RAG / code-editing / agentic traffic, where the output copies
    long spans of the input.
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(num_requests):
        motif = random_prompt(rng, rng.randrange(8, 16))
        prompts.append((motif * 12)[:160])
    return prompts


def save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a markdown table. ``columns`` is a list of (key, header)."""
    head = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(k, "")) for k, _ in columns) + " |" for row in rows
    ]
    return "\n".join([head, rule, *body])
