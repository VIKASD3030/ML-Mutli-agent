"""
DataProfile — produced by the Data Agent (Agent 2).

Captures what the Data Agent found and did, so downstream agents (and humans
reviewing a run) can see the cleaning decisions without re-deriving them.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    missing_pct: float = Field(..., ge=0.0, le=100.0)
    is_target: bool = False


class CleaningAction(BaseModel):
    column: str
    action: str  # e.g. "median_imputation", "dropped_duplicate_rows", "type_coerced"
    rationale: str


class DataProfile(BaseModel):
    row_count: int = Field(..., ge=0)
    column_count: int = Field(..., ge=0)
    column_schema: list[ColumnSchema]
    class_balance: Optional[dict[str, float]] = Field(
        default=None,
        description="Class label -> proportion, only populated for classification tasks.",
    )
    outlier_flags: list[str] = Field(
        default_factory=list,
        description="Column names flagged for outlier concerns.",
    )
    cleaning_actions_taken: list[CleaningAction] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking issues surfaced to the Orchestrator/user, "
        "e.g. severe class imbalance.",
    )
    blocking_issue: Optional[str] = Field(
        default=None,
        description="If set, the pipeline halts here — data is unusable as-is "
        "(e.g. >50% missing on the target column).",
    )

    @property
    def is_usable(self) -> bool:
        return self.blocking_issue is None
