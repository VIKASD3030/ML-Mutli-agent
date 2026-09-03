"""
db/models.py

SQLAlchemy ORM tables — the fully relational persistence layer for
PipelineRun and everything it contains. Every class here has a "Record"
suffix specifically to avoid colliding with the Pydantic schemas of the
same conceptual name in schemas/ — db/sync.py needs to import BOTH the
Pydantic class and the ORM class for the same concept simultaneously.

This file is built incrementally, one PipelineRun field at a time,
matching the order fields actually get populated during a real run:
Run -> ProblemSpec -> DataProfile -> EDAReport -> TuningResult ->
EvaluationReport -> history/trace. This chunk covers Run, ProblemSpec,
and PipelineConstraints only — more tables follow in later chunks.
"""

from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class RunRecord(Base):
    """The root table. Every other table hangs off this one via
    run_id — same role schemas/pipeline_run.py's PipelineRun plays as
    the root Pydantic object everything else nests under."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="created")
    loop_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    # One-to-one relationships to each stage's data — uselist=False makes
    # SQLAlchemy return a single object (or None) instead of a list, since
    # a run has exactly one ProblemSpec, exactly one DataProfile, etc.
    # cascade="all, delete-orphan" means deleting a Run deletes everything
    # that hangs off it too — no orphaned rows left behind.
    problem_spec: Mapped["ProblemSpecRecord"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    data_profile: Mapped["DataProfileRecord"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    eda_report: Mapped["EDAReportRecord"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    tuning_result: Mapped["TuningResultRecord"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    evaluation_report: Mapped["EvaluationReportRecord"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    history_entries: Mapped[list["HistoryEntryRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    trace_entries: Mapped[list["TraceEntryRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ProblemSpecRecord(Base):
    """Mirrors schemas/problem_spec.py's ProblemSpec — one row per run,
    linked via a foreign key back to RunRecord."""

    __tablename__ = "problem_specs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # unique=True enforces the one-to-one relationship at the database
    # level, not just in Python — two ProblemSpecRecords can never point
    # at the same run_id, matching that a PipelineRun has exactly one
    # ProblemSpec, always.
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), unique=True)

    task_type: Mapped[str] = mapped_column(String)
    target_column: Mapped[str] = mapped_column(String)
    success_metric: Mapped[str] = mapped_column(String)
    metric_threshold: Mapped[float] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)

    run: Mapped["RunRecord"] = relationship(back_populates="problem_spec")
    constraints: Mapped["PipelineConstraintsRecord"] = relationship(
        back_populates="problem_spec", uselist=False, cascade="all, delete-orphan"
    )


class PipelineConstraintsRecord(Base):
    """Mirrors ProblemSpec's nested PipelineConstraints object — its own
    table, not columns on ProblemSpecRecord directly, because the
    Pydantic side already models it as a distinct nested object with its
    own identity. Keeping the same nesting shape on the DB side means
    db/sync.py's translation logic can mirror the Pydantic structure
    field-for-field instead of flattening it."""

    __tablename__ = "pipeline_constraints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_spec_id: Mapped[int] = mapped_column(ForeignKey("problem_specs.id"), unique=True)

    max_training_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_tuning_trials: Mapped[int | None] = mapped_column(Integer, nullable=True)
    interpretability_required: Mapped[bool] = mapped_column(Boolean, default=False)
    max_loop_backs: Mapped[int] = mapped_column(Integer, default=3)

    problem_spec: Mapped["ProblemSpecRecord"] = relationship(back_populates="constraints")

# =====================================================================
# DataProfile group — mirrors schemas/data_profile.py
# =====================================================================

class DataProfileRecord(Base):
    __tablename__ = "data_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), unique=True)

    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    # blocking_issue stays a plain nullable column here, NOT its own
    # table — it's a single scalar per profile (at most one), unlike
    # cleaning_actions_taken or warnings, which are genuine collections.
    blocking_issue: Mapped[str | None] = mapped_column(String, nullable=True)

    run: Mapped["RunRecord"] = relationship(back_populates="data_profile")
    column_schemas: Mapped[list["ColumnSchemaRecord"]] = relationship(
        back_populates="data_profile", cascade="all, delete-orphan"
    )
    cleaning_actions: Mapped[list["CleaningActionRecord"]] = relationship(
        back_populates="data_profile", cascade="all, delete-orphan"
    )
    class_balance_entries: Mapped[list["ClassBalanceEntryRecord"]] = relationship(
        back_populates="data_profile", cascade="all, delete-orphan"
    )
    warnings: Mapped[list["DataProfileWarningRecord"]] = relationship(
        back_populates="data_profile", cascade="all, delete-orphan"
    )
    outlier_flags: Mapped[list["OutlierFlagRecord"]] = relationship(
        back_populates="data_profile", cascade="all, delete-orphan"
    )


class ColumnSchemaRecord(Base):
    __tablename__ = "column_schemas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_profile_id: Mapped[int] = mapped_column(ForeignKey("data_profiles.id"))

    name: Mapped[str] = mapped_column(String)
    dtype: Mapped[str] = mapped_column(String)
    missing_pct: Mapped[float] = mapped_column(Float)
    is_target: Mapped[bool] = mapped_column(Boolean)

    data_profile: Mapped["DataProfileRecord"] = relationship(back_populates="column_schemas")


class CleaningActionRecord(Base):
    __tablename__ = "cleaning_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_profile_id: Mapped[int] = mapped_column(ForeignKey("data_profiles.id"))

    column: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    rationale: Mapped[str] = mapped_column(String)

    data_profile: Mapped["DataProfileRecord"] = relationship(back_populates="cleaning_actions")


class ClassBalanceEntryRecord(Base):
    """One row per class label. DataProfile.class_balance is a
    dict[str, float] ("0" -> 0.3726, "1" -> 0.6274) — a dict, same
    dict-becomes-rows translation as everywhere else in a fully
    normalized design."""

    __tablename__ = "class_balance_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_profile_id: Mapped[int] = mapped_column(ForeignKey("data_profiles.id"))

    class_label: Mapped[str] = mapped_column(String)
    proportion: Mapped[float] = mapped_column(Float)

    data_profile: Mapped["DataProfileRecord"] = relationship(back_populates="class_balance_entries")


class DataProfileWarningRecord(Base):
    __tablename__ = "data_profile_warnings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_profile_id: Mapped[int] = mapped_column(ForeignKey("data_profiles.id"))
    warning_text: Mapped[str] = mapped_column(String)

    data_profile: Mapped["DataProfileRecord"] = relationship(back_populates="warnings")


class OutlierFlagRecord(Base):
    """NEW — missed in the earlier illustrative table list.
    DataProfile.outlier_flags is a list[str] of column names flagged
    for outlier concerns — a genuine collection, needs its own table
    same as every other list field."""

    __tablename__ = "outlier_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_profile_id: Mapped[int] = mapped_column(ForeignKey("data_profiles.id"))
    column_name: Mapped[str] = mapped_column(String)

    data_profile: Mapped["DataProfileRecord"] = relationship(back_populates="outlier_flags")


# =====================================================================
# EDAReport group — mirrors schemas/eda_report.py
# =====================================================================

class EDAReportRecord(Base):
    __tablename__ = "eda_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), unique=True)
    target_distribution_notes: Mapped[str] = mapped_column(String)

    run: Mapped["RunRecord"] = relationship(back_populates="eda_report")
    correlations: Mapped[list["FeatureCorrelationRecord"]] = relationship(
        back_populates="eda_report", cascade="all, delete-orphan"
    )
    feature_decisions: Mapped[list["FeatureDecisionRecord"]] = relationship(
        back_populates="eda_report", cascade="all, delete-orphan"
    )
    leakage_warnings: Mapped[list["LeakageWarningRecord"]] = relationship(
        back_populates="eda_report", cascade="all, delete-orphan"
    )
    data_quality_warnings: Mapped[list["DataQualityWarningRecord"]] = relationship(
        back_populates="eda_report", cascade="all, delete-orphan"
    )
    final_feature_names: Mapped[list["FinalFeatureNameRecord"]] = relationship(
        back_populates="eda_report", cascade="all, delete-orphan"
    )


class FeatureCorrelationRecord(Base):
    __tablename__ = "feature_correlations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eda_report_id: Mapped[int] = mapped_column(ForeignKey("eda_reports.id"))
    feature_name: Mapped[str] = mapped_column(String)
    correlation: Mapped[float] = mapped_column(Float)

    eda_report: Mapped["EDAReportRecord"] = relationship(back_populates="correlations")


class FeatureDecisionRecord(Base):
    """Covers BOTH engineered_features and dropped_features from
    EDAReport — same underlying shape (name + rationale) in the Pydantic
    schema (both use FeatureDecision), so one table with a `kind`
    discriminator column ("engineered" | "dropped") instead of two
    near-identical tables. This is a deliberate exception to "everything
    gets its own table" — the two Pydantic FIELDS are genuinely the same
    TYPE, just used in two different list slots, so a discriminator
    column is the correct normalization, not a shortcut."""

    __tablename__ = "feature_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eda_report_id: Mapped[int] = mapped_column(ForeignKey("eda_reports.id"))
    kind: Mapped[str] = mapped_column(String)  # "engineered" | "dropped"
    name: Mapped[str] = mapped_column(String)
    rationale: Mapped[str] = mapped_column(String)

    eda_report: Mapped["EDAReportRecord"] = relationship(back_populates="feature_decisions")


class LeakageWarningRecord(Base):
    __tablename__ = "leakage_warnings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eda_report_id: Mapped[int] = mapped_column(ForeignKey("eda_reports.id"))
    warning_text: Mapped[str] = mapped_column(String)

    eda_report: Mapped["EDAReportRecord"] = relationship(back_populates="leakage_warnings")


class DataQualityWarningRecord(Base):
    """Mirrors EDAReport.data_quality_warnings — the field added during
    the NaT-handling fix. Deliberately its own table, not merged with
    LeakageWarningRecord, for the exact reason that field was kept
    separate from leakage_warnings in the Pydantic schema: "can this
    result be trusted" vs "will this run complete at all" are different
    concerns that shouldn't be mixed in one list, in Python OR in the DB."""

    __tablename__ = "data_quality_warnings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eda_report_id: Mapped[int] = mapped_column(ForeignKey("eda_reports.id"))
    warning_text: Mapped[str] = mapped_column(String)

    eda_report: Mapped["EDAReportRecord"] = relationship(back_populates="data_quality_warnings")


class FinalFeatureNameRecord(Base):
    __tablename__ = "final_feature_names"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eda_report_id: Mapped[int] = mapped_column(ForeignKey("eda_reports.id"))
    feature_name: Mapped[str] = mapped_column(String)

    eda_report: Mapped["EDAReportRecord"] = relationship(back_populates="final_feature_names")


# =====================================================================
# TuningResult group — mirrors schemas/tuning_result.py
# =====================================================================

class TuningResultRecord(Base):
    __tablename__ = "tuning_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), unique=True)

    best_cv_score: Mapped[float] = mapped_column(Float)
    metric_optimized: Mapped[str] = mapped_column(String)
    search_budget_used: Mapped[int] = mapped_column(Integer)
    search_budget_total: Mapped[int] = mapped_column(Integer)
    convergence_notes: Mapped[str] = mapped_column(String)
    # This is the exact field whose meaning we spent real debugging time
    # on — was structurally always False in production until fixed to
    # reflect the real plateau heuristic. Persisting it correctly here
    # matters for the same reason it mattered in training_stage.py.
    converged: Mapped[bool] = mapped_column(Boolean)

    run: Mapped["RunRecord"] = relationship(back_populates="tuning_result")
    best_params: Mapped[list["TuningBestParamRecord"]] = relationship(
        back_populates="tuning_result", cascade="all, delete-orphan"
    )
    trial_records: Mapped[list["TrialRecordRow"]] = relationship(
        back_populates="tuning_result", cascade="all, delete-orphan"
    )


class TuningBestParamRecord(Base):
    """best_params is dict[str, Any] — hyperparameter names to values.
    param_value is stored as a STRING (str(value)) here, not a typed
    column — a real, honest limitation of full normalization: Any means
    genuinely heterogeneous types (int, float, occasionally str for some
    model families), and a single relational column can't type-vary per
    row. Reading this back means re-parsing the string, which the
    Pydantic side never had to do. Flagging this explicitly rather than
    pretending it's lossless."""

    __tablename__ = "tuning_best_params"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tuning_result_id: Mapped[int] = mapped_column(ForeignKey("tuning_results.id"))
    param_name: Mapped[str] = mapped_column(String)
    param_value: Mapped[str] = mapped_column(String)

    tuning_result: Mapped["TuningResultRecord"] = relationship(back_populates="best_params")


class TrialRecordRow(Base):
    """Named *Row*, not *Record*, specifically to avoid colliding with
    the "Record" suffix convention — schemas/tuning_result.py already
    has a class literally named TrialRecord, so TrialRecordRecord would
    be both confusing and silly."""

    __tablename__ = "trial_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tuning_result_id: Mapped[int] = mapped_column(ForeignKey("tuning_results.id"))
    trial_number: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)

    tuning_result: Mapped["TuningResultRecord"] = relationship(back_populates="trial_records")
    params: Mapped[list["TrialParamRecord"]] = relationship(
        back_populates="trial_record", cascade="all, delete-orphan"
    )


class TrialParamRecord(Base):
    """NEW — missed earlier. Each TrialRecord ALSO has its own
    params: dict[str, Any] (the specific hyperparameters tried in that
    one trial, distinct from TuningResult's overall best_params). This
    is a dict nested inside a list item — to stay genuinely relational,
    it needs its own table, one level deeper than TuningBestParamRecord.
    Same string-storage caveat as TuningBestParamRecord applies here too."""

    __tablename__ = "trial_record_params"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_record_id: Mapped[int] = mapped_column(ForeignKey("trial_records.id"))
    param_name: Mapped[str] = mapped_column(String)
    param_value: Mapped[str] = mapped_column(String)

    trial_record: Mapped["TrialRecordRow"] = relationship(back_populates="params")


# =====================================================================
# EvaluationReport group — mirrors schemas/evaluation_report.py
# =====================================================================

class EvaluationReportRecord(Base):
    __tablename__ = "evaluation_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), unique=True)

    metric_optimized: Mapped[str] = mapped_column(String)
    metric_value: Mapped[float] = mapped_column(Float)
    metric_threshold: Mapped[float] = mapped_column(Float)
    pass_fail: Mapped[str] = mapped_column(String)
    model_artifact_path: Mapped[str | None] = mapped_column(String, nullable=True)

    run: Mapped["RunRecord"] = relationship(back_populates="evaluation_report")
    test_metrics: Mapped[list["TestMetricEntryRecord"]] = relationship(
        back_populates="evaluation_report", cascade="all, delete-orphan"
    )
    confusion_matrix_cells: Mapped[list["ConfusionMatrixCellRecord"]] = relationship(
        back_populates="evaluation_report", cascade="all, delete-orphan"
    )
    residual_summary_entries: Mapped[list["ResidualSummaryEntryRecord"]] = relationship(
        back_populates="evaluation_report", cascade="all, delete-orphan"
    )
    # failure_analysis is OPTIONAL and one-to-one — a passing run has
    # none at all, a failing run has exactly one. uselist=False still
    # applies; the Python-side value is just None when there isn't one.
    failure_analysis: Mapped["FailureAnalysisRecord"] = relationship(
        back_populates="evaluation_report", uselist=False, cascade="all, delete-orphan"
    )


class TestMetricEntryRecord(Base):
    __tablename__ = "test_metric_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_report_id: Mapped[int] = mapped_column(ForeignKey("evaluation_reports.id"))
    metric_name: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)

    evaluation_report: Mapped["EvaluationReportRecord"] = relationship(back_populates="test_metrics")


class ConfusionMatrixCellRecord(Base):
    """confusion_matrix is list[list[int]] — a 2D matrix, decomposed
    into one row per cell with explicit coordinates so it can be
    reassembled into the original 2D shape on read."""

    __tablename__ = "confusion_matrix_cells"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_report_id: Mapped[int] = mapped_column(ForeignKey("evaluation_reports.id"))
    row_idx: Mapped[int] = mapped_column(Integer)
    col_idx: Mapped[int] = mapped_column(Integer)
    count: Mapped[int] = mapped_column(Integer)

    evaluation_report: Mapped["EvaluationReportRecord"] = relationship(
        back_populates="confusion_matrix_cells"
    )


class ResidualSummaryEntryRecord(Base):
    """NEW — missed earlier. EvaluationReport.residual_summary is
    Optional[dict[str, float]] (mean/std of residuals), populated only
    for REGRESSION tasks — will simply have zero rows for every
    classification run, which is correct, not a gap."""

    __tablename__ = "residual_summary_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_report_id: Mapped[int] = mapped_column(ForeignKey("evaluation_reports.id"))
    key: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)

    evaluation_report: Mapped["EvaluationReportRecord"] = relationship(
        back_populates="residual_summary_entries"
    )


class FailureAnalysisRecord(Base):
    __tablename__ = "failure_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_report_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_reports.id"), unique=True
    )
    summary: Mapped[str] = mapped_column(String)
    recommended_next_step: Mapped[str] = mapped_column(String)

    evaluation_report: Mapped["EvaluationReportRecord"] = relationship(
        back_populates="failure_analysis"
    )


# =====================================================================
# History and Trace — mirror schemas/pipeline_run.py's HistoryEntry
# and schemas/trace_entry.py's TraceEntry. Both attach DIRECTLY to
# RunRecord (not through an intermediate report table) since both are
# top-level lists on PipelineRun itself, not nested inside a stage report.
# =====================================================================

class HistoryEntryRecord(Base):
    __tablename__ = "history_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    stage: Mapped[str] = mapped_column(String)
    event: Mapped[str] = mapped_column(String)

    run: Mapped["RunRecord"] = relationship(back_populates="history_entries")


class TraceEntryRecord(Base):
    __tablename__ = "trace_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    agent: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    estimated_cost_usd: Mapped[float] = mapped_column(Float)

    run: Mapped["RunRecord"] = relationship(back_populates="trace_entries")