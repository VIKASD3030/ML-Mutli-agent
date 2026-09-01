"""
schemas/trace_entry.py

Structured record of ONE LLM API call — the thing PipelineRun.history's
free-text log lines can't answer questions from. Every agent's LLM call
produces exactly one of these.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

class TraceEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float