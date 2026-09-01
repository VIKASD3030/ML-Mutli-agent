"""
EvaluationReport — produced by the Model Training & Testing Agent (Agent 5).

`failure_analysis.recommended_next_step` is what the Orchestrator's
deterministic state machine reads to decide where to route control on a
failed run. The *content* of that recommendation is agent-produced; the
*routing* itself stays deterministic code (see pipeline/run_pipeline.py).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RecommendedNextStep = Literal[
    "revisit_features",
    "expand_hyperparam_search",
    "insufficient_data",
    "bad_spec",
]


class FailureAnalysis(BaseModel):
    summary: str
    recommended_next_step: RecommendedNextStep


class EvaluationReport(BaseModel):
    test_metrics: dict[str, float]
    confusion_matrix: Optional[list[list[int]]] = Field(
        default=None, description="Classification only."
    )
    residual_summary: Optional[dict[str, float]] = Field(
        default=None, description="Regression only, e.g. mean/std of residuals."
    )
    metric_optimized: str
    metric_value: float
    metric_threshold: float
    pass_fail: Literal["pass", "fail"]
    failure_analysis: Optional[FailureAnalysis] = None
    model_artifact_path: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.pass_fail == "pass"
