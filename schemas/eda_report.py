"""
EDAReport — produced by the Feature Engineering / EDA Agent (Agent 3).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureDecision(BaseModel):
    name: str
    rationale: str


class EDAReport(BaseModel):
    correlation_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Feature name -> correlation with target (absolute value).",
    )
    engineered_features: list[FeatureDecision] = Field(default_factory=list)
    dropped_features: list[FeatureDecision] = Field(default_factory=list)
    leakage_warnings: list[str] = Field(
        default_factory=list,
        description="Features suspiciously predictive of the target — "
        "high-priority signal, never silently dropped by the Orchestrator.",
    )
    target_distribution_notes: str = ""
    final_feature_names: list[str] = Field(default_factory=list)

    @property
    def has_leakage_risk(self) -> bool:
        return len(self.leakage_warnings) > 0
