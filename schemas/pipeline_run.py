"""
PipelineRun — the single persisted state object for one end-to-end run.

Each agent reads only the field(s) relevant to its stage (the Data Agent
never sees tuning_result, for instance), even though the whole object lives
in one place. In Phase 1 (this skeleton) it's just an in-memory/JSON-
serializable object; Phase 2+ would back this with a Postgres row.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from schemas.trace_entry import TraceEntry

from pydantic import BaseModel, Field

from schemas.data_profile import DataProfile
from schemas.eda_report import EDAReport
from schemas.evaluation_report import EvaluationReport
from schemas.problem_spec import ProblemSpec
from schemas.tuning_result import TuningResult

RunStatus = Literal[
    "created",
    "data_ready",
    "features_ready",
    "tuned",
    "evaluated",
    "done",
    "escalated",
]


class HistoryEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: str
    event: str


class PipelineRun(BaseModel):
    run_id: str
    status: RunStatus = "created"
    loop_count: int = 0

    problem_spec: ProblemSpec
    data_profile: Optional[DataProfile] = None
    eda_report: Optional[EDAReport] = None
    tuning_result: Optional[TuningResult] = None
    evaluation_report: Optional[EvaluationReport] = None
    trace: list[TraceEntry] = Field(default_factory=list)

    history: list[HistoryEntry] = Field(default_factory=list)

    def log(self, stage: str, event: str) -> None:
        self.history.append(HistoryEntry(stage=stage, event=event))

    def record_call(self, entry: TraceEntry) -> None:
        self.trace.append(entry)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.estimated_cost_usd for e in self.trace), 6)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.trace)