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
    data_quality_warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking data problems that will not stop feature "
        "engineering but may break a later stage — e.g. missing values that "
        "survive into derived features and that a RandomForest cannot consume. "
        "Deliberately SEPARATE from leakage_warnings: leakage is a question of "
        "whether a result can be trusted, this is a question of whether the run "
        "will complete at all. Conflating them would mean an agent weighing "
        "'this model may be too good to be true' against 'training is about to "
        "crash' on the same axis.",
    )
    target_distribution_notes: str = ""
    final_feature_names: list[str] = Field(default_factory=list)

    @property
    def has_leakage_risk(self) -> bool:
        return len(self.leakage_warnings) > 0

    @property
    def has_data_quality_risk(self) -> bool:
        return len(self.data_quality_warnings) > 0
