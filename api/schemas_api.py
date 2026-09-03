"""
api/schemas_api.py

Request/response models for the HTTP API — kept as its OWN top-level
types, separate from schemas/pipeline_run.py's PipelineRun, because the
two evolve for different reasons: internal schemas change when the
pipeline needs a fix (recall `converged`, `data_quality_warnings` — both
added mid-project to fix real bugs, not to serve any API consumer). The
API's own shape shouldn't silently change every time that happens.

Judgment call worth being explicit about: nested fields WITHIN RunDetail
(DataProfile, EDAReport, etc.) reuse the real internal types directly,
rather than being re-declared as API-only duplicates. That's a deliberate
asymmetry — duplicating every nested field of 5 schemas just to keep them
"API-owned" would be pure busywork with no real benefit, and FastAPI gets
accurate OpenAPI docs for free by pointing at the real types. The
separation matters most at the TOP level (RunDetail is its own class,
not a raw PipelineRun re-export) and especially for REQUESTS, where
deliberate validation (see RunCreateRequest below) actually matters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

from schemas.data_profile import DataProfile
from schemas.eda_report import EDAReport
from schemas.evaluation_report import EvaluationReport
from schemas.problem_spec import ProblemSpec
from schemas.pipeline_run import PipelineRun, HistoryEntry
from schemas.trace_entry import TraceEntry
from schemas.tuning_result import TuningResult

class RunCreateRequest(BaseModel):
    problem_spec: Optional[ProblemSpec] = None
    context: Optional[str] = None
    file_path: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_input_mode(self) -> "RunCreateRequest":
        has_spec = self.problem_spec is not None
        has_context = self.context is not None and self.context.strip() != ""
        if has_spec == has_context:
            raise ValueError(
                "Provide exactly one of: problem_spec (structured) or "
                "context (free text, optionally with file_path)."
            )
        return self

class ClarificationResponse(BaseModel):
    status: Literal["needs_clarification"] = "needs_clarification"
    missing_fields: list[str]
    question_for_user: str

class RunCreatedResponse(BaseModel):
    run_id: str
    status: Literal["queued"] = "queued"

class UploadResponse(BaseModel):
    file_path: str
    filename: str
    size_bytes: int

class RunSummary(BaseModel):
    """GET /runs — deliberately lightweight, no nested reports. A list
    endpoint returning full 26-table detail per row would get slow fast
    once there's real run history."""

    run_id: str
    status: str
    task_type: Optional[str] = None
    pass_fail: Optional[str] = None
    metric_value: Optional[float] = None
    loop_count: int
    created_at: datetime
    updated_at: datetime

class RunDetail(BaseModel):
    """GET /runs/{run_id} — full detail, backs the live-run view."""

    run_id: str
    status: str
    loop_count: int
    problem_spec: ProblemSpec
    data_profile: Optional[DataProfile] = None
    eda_report: Optional[EDAReport] = None
    tuning_result: Optional[TuningResult] = None
    evaluation_report: Optional[EvaluationReport] = None
    history: list[HistoryEntry] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0


    @classmethod
    def from_pipeline_run(cls, pipeline_run: PipelineRun) -> "RunDetail":
        return cls(
            run_id=pipeline_run.run_id, status=pipeline_run.status,
            loop_count=pipeline_run.loop_count, problem_spec=pipeline_run.problem_spec,
            data_profile=pipeline_run.data_profile, eda_report=pipeline_run.eda_report,
            tuning_result=pipeline_run.tuning_result,
            evaluation_report=pipeline_run.evaluation_report,
            history=pipeline_run.history, trace=pipeline_run.trace,
            total_cost_usd=pipeline_run.total_cost_usd, total_tokens=pipeline_run.total_tokens,
        )