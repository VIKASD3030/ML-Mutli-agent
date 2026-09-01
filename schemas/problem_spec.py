"""
ProblemSpec — produced by the Requirement Gathering Agent (Agent 1).

This is the single source of truth for "what are we trying to do." No other
agent is allowed to modify it; every downstream stage reads from it.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

TaskType = Literal["classification", "regression"]

# A deliberately small, explicit set of supported metrics per task type.
# Keeping this closed (rather than a free-text string) means every downstream
# agent can branch on it safely instead of parsing arbitrary text.
CLASSIFICATION_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}
REGRESSION_METRICS = {"rmse", "mae", "r2"}


class PipelineConstraints(BaseModel):
    """Optional constraints that later stages must respect."""

    max_training_seconds: Optional[int] = Field(
        default=None, description="Wall-clock budget for the training stage."
    )
    max_tuning_trials: Optional[int] = Field(
        default=50, description="Upper bound on hyperparameter search trials."
    )
    interpretability_required: bool = Field(
        default=False,
        description="If true, the Training Agent should prefer interpretable "
        "model families (e.g., linear/tree) over opaque ones.",
    )
    max_loop_backs: int = Field(
        default=3,
        description="How many times the Orchestrator may route control "
        "backward before escalating to the user.",
    )


class ProblemSpec(BaseModel):
    """Structured, machine-actionable version of the user's request."""

    task_type: TaskType
    target_column: str = Field(..., min_length=1)
    success_metric: str = Field(
        ..., description="Metric name; must be valid for the given task_type."
    )
    metric_threshold: float = Field(
        ..., description="Minimum (or maximum, for error metrics) acceptable "
        "value of success_metric for the pipeline to report pass."
    )
    data_source: str = Field(
        ..., description="Path, URI, or table name where the raw data lives."
    )
    constraints: PipelineConstraints = Field(default_factory=PipelineConstraints)
    notes: Optional[str] = Field(
        default=None, description="Free-text context carried along for humans; "
        "never parsed by downstream agents."
    )

    @field_validator("success_metric")
    @classmethod
    def _metric_matches_task(cls, v: str, info) -> str:
        task_type = info.data.get("task_type")
        valid = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
        if task_type is not None and v not in valid:
            raise ValueError(
                f"'{v}' is not a valid metric for task_type='{task_type}'. "
                f"Expected one of: {sorted(valid)}"
            )
        return v


class ClarificationNeeded(BaseModel):
    """Returned instead of a ProblemSpec when the request is too ambiguous."""

    status: Literal["needs_clarification"] = "needs_clarification"
    missing_fields: list[str]
    question_for_user: str
