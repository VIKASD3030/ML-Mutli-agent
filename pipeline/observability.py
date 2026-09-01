"""
pipeline/observability.py

Shared wrapper around every agent's LLM call. Every agent should call
timed_llm_call() instead of self.client.beta.chat.completions.parse()
directly — this is what makes cost/latency tracking automatic and
consistent across all 5 agents, rather than each one reimplementing it.

COST TABLE NOTE: the rates below are illustrative placeholders, not
guaranteed-current pricing — check OpenAI's actual current pricing page
and update PRICING_PER_1K_TOKENS before trusting estimated_cost_usd for
anything beyond rough relative comparison between runs.
"""

from __future__ import annotations

import time
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from schemas.trace_entry import TraceEntry

T = TypeVar("T", bound=BaseModel)

# {model: (cost per 1K prompt tokens, cost per 1K completion tokens)} in USD.
# PLACEHOLDER VALUES — verify against OpenAI's current pricing before
# relying on estimated_cost_usd for real budgeting decisions.

PRICING_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
}
DEFAULT_PRICING = (0.0,0.0)

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = PRICING_PER_1K_TOKENS.get(model, DEFAULT_PRICING)
    return (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate

def timed_llm_call(
    client: OpenAI,
    agent_name: str,
    model: str,
    messages: list[dict],
    response_format: Type[T],
    temperature: float = 0,
) -> tuple[T, TraceEntry]:
    """Wraps client.beta.chat.completions.parse() with timing + usage
    tracking. Returns the PARSED object (same as calling .parse()
    yourself) plus a TraceEntry describing the call — agents append the
    entry to pipeline_run.trace themselves (this function doesn't take
    pipeline_run, keeping it usable standalone/in tests, same philosophy
    as every tool wrapper's optional cache/run_id params).
    """

    start =time.monotonic()
    completion = client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=messages,
        response_format=response_format
    )

    latency_ms = (time.monotonic() - start) * 1000

    usage = completion.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    trace_entry = TraceEntry(
        agent= agent_name,
        model = model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=round(latency_ms, 2),
        estimated_cost_usd=round(_estimate_cost(model, prompt_tokens, completion_tokens), 6),
    )

    parsed = completion.choices[0].message.parsed
    return parsed, trace_entry