"""
db/sync.py

The bridge between the Pydantic world (schemas/pipeline_run.py's
PipelineRun, used everywhere in agents/pipeline/tools) and the fully
relational SQLAlchemy world (db/models.py's 26 tables).

Two functions:
- persist_run_state(session, pipeline_run) — WRITE. Called after every
  agent step, same place pipeline_run.log()/record_call() already get
  called. Full-replace strategy: every stage report present on the
  Pydantic object gets its OLD child rows deleted and NEW ones inserted,
  via SQLAlchemy's cascade="all, delete-orphan" reacting to reassigning
  a relationship attribute. This is simpler and more robust than
  diffing old-vs-new — the honest cost is that child row IDs churn on
  every sync (a fresh autoincrement id each time), which is fine since
  nothing external references those ids, only run_id.
- load_run_state(session, run_id) — READ. Reconstructs a real
  PipelineRun object from the full row tree — the exact reverse
  translation, including reassembling dicts from row-lists and the 2D
  confusion_matrix from individual cell rows.

KNOWN LIMITATION, flagged rather than hidden: TuningResult.best_params
and TrialRecord.params are dict[str, Any] on the Pydantic side, but a
relational column can't type-vary per row — param_value is stored as
str(value) and best-effort re-parsed as int/float on read via
_parse_param_value(). This is lossy only in the theoretical case of a
genuinely string-valued hyperparameter, which never occurs in this
project's actual RandomForest search space (n_estimators, max_depth,
min_samples_split are always int).

PERFORMANCE NOTE, also flagged rather than optimized away: load_run_state
does not use SQLAlchemy's selectinload/joinedload, so each relationship
access below triggers its own lazy-loaded query (N+1 queries). Acceptable
for this project's read frequency (one run detail view at a time); would
be worth adding if this ever needs to list many runs' full detail at once.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from db.models import (
    ClassBalanceEntryRecord,
    CleaningActionRecord,
    ColumnSchemaRecord,
    ConfusionMatrixCellRecord,
    DataProfileRecord,
    DataProfileWarningRecord,
    DataQualityWarningRecord,
    EDAReportRecord,
    EvaluationReportRecord,
    FailureAnalysisRecord,
    FeatureCorrelationRecord,
    FeatureDecisionRecord,
    FinalFeatureNameRecord,
    HistoryEntryRecord,
    LeakageWarningRecord,
    OutlierFlagRecord,
    PipelineConstraintsRecord,
    ProblemSpecRecord,
    ResidualSummaryEntryRecord,
    RunRecord,
    TestMetricEntryRecord,
    TraceEntryRecord,
    TrialParamRecord,
    TrialRecordRow,
    TuningBestParamRecord,
    TuningResultRecord,
)
from schemas.data_profile import CleaningAction, ColumnSchema, DataProfile
from schemas.eda_report import EDAReport, FeatureDecision
from schemas.evaluation_report import EvaluationReport, FailureAnalysis
from schemas.pipeline_run import HistoryEntry, PipelineRun
from schemas.problem_spec import PipelineConstraints, ProblemSpec
from schemas.trace_entry import TraceEntry
from schemas.tuning_result import TrialRecord, TuningResult

# =====================================================================
# WRITE PATH — Pydantic PipelineRun -> ORM row tree
# =====================================================================


def persist_run_state(session: Session, pipeline_run: PipelineRun) -> None:
    """Upsert the full state of a PipelineRun into the database. Safe to
    call repeatedly during a single run — each call fully replaces every
    section that's currently populated on the Pydantic object."""

    run_record = session.get(RunRecord, pipeline_run.run_id)
    if run_record is None:
        run_record = RunRecord(run_id=pipeline_run.run_id)
        session.add(run_record)

    run_record.status = pipeline_run.status
    run_record.loop_count = pipeline_run.loop_count

    # ProblemSpec is always present from the first call — and unlike the
    # other four stage reports, it can genuinely CHANGE mid-run (e.g.
    # constraints.max_tuning_trials growing on an expand_hyperparam_search
    # loop-back), so it's resynced unconditionally every call, not gated
    # behind an `is not None` check like the others below.
    if run_record.problem_spec is not None:
        session.delete(run_record.problem_spec)
        session.flush()
    run_record.problem_spec = _build_problem_spec_record(pipeline_run.problem_spec)

    if pipeline_run.data_profile is not None:
        if run_record.data_profile is not None:
            session.delete(run_record.data_profile)
            session.flush()
        run_record.data_profile = _build_data_profile_record(pipeline_run.data_profile)

    if pipeline_run.eda_report is not None:
        if run_record.eda_report is not None:
            session.delete(run_record.eda_report)
            session.flush()
        run_record.eda_report = _build_eda_report_record(pipeline_run.eda_report)

    if pipeline_run.tuning_result is not None:
        if run_record.tuning_result is not None:
            session.delete(run_record.tuning_result)
            session.flush()
        run_record.tuning_result = _build_tuning_result_record(pipeline_run.tuning_result)

    if pipeline_run.evaluation_report is not None:
        if run_record.evaluation_report is not None:
            session.delete(run_record.evaluation_report)
            session.flush()
        run_record.evaluation_report = _build_evaluation_report_record(
            pipeline_run.evaluation_report
        )

    # history/trace are append-only lists that GROW across a run's
    # lifetime — reassigning the whole collection each call (rather than
    # trying to append only the new entries) keeps this function simple
    # and correct at the cost of some redundant delete+insert work on
    # already-synced entries. Given this project's actual scale (tens of
    # entries per run, not thousands), that cost is negligible.
    run_record.history_entries = [
        HistoryEntryRecord(timestamp=e.timestamp, stage=e.stage, event=e.event)
        for e in pipeline_run.history
    ]
    run_record.trace_entries = [
        TraceEntryRecord(
            timestamp=e.timestamp,
            agent=e.agent,
            model=e.model,
            prompt_tokens=e.prompt_tokens,
            completion_tokens=e.completion_tokens,
            total_tokens=e.total_tokens,
            latency_ms=e.latency_ms,
            estimated_cost_usd=e.estimated_cost_usd,
        )
        for e in pipeline_run.trace
    ]

    session.commit()


def _build_problem_spec_record(spec: ProblemSpec) -> ProblemSpecRecord:
    return ProblemSpecRecord(
        task_type=spec.task_type,
        target_column=spec.target_column,
        success_metric=spec.success_metric,
        metric_threshold=spec.metric_threshold,
        data_source=spec.data_source,
        notes=spec.notes,
        constraints=PipelineConstraintsRecord(
            max_training_seconds=spec.constraints.max_training_seconds,
            max_tuning_trials=spec.constraints.max_tuning_trials,
            interpretability_required=spec.constraints.interpretability_required,
            max_loop_backs=spec.constraints.max_loop_backs,
        ),
    )


def _build_data_profile_record(profile: DataProfile) -> DataProfileRecord:
    return DataProfileRecord(
        row_count=profile.row_count,
        column_count=profile.column_count,
        blocking_issue=profile.blocking_issue,
        column_schemas=[
            ColumnSchemaRecord(
                name=c.name, dtype=c.dtype, missing_pct=c.missing_pct, is_target=c.is_target
            )
            for c in profile.column_schema
        ],
        cleaning_actions=[
            CleaningActionRecord(column=a.column, action=a.action, rationale=a.rationale)
            for a in profile.cleaning_actions_taken
        ],
        class_balance_entries=[
            ClassBalanceEntryRecord(class_label=label, proportion=prop)
            for label, prop in (profile.class_balance or {}).items()
        ],
        warnings=[DataProfileWarningRecord(warning_text=w) for w in profile.warnings],
        outlier_flags=[OutlierFlagRecord(column_name=c) for c in profile.outlier_flags],
    )


def _build_eda_report_record(report: EDAReport) -> EDAReportRecord:
    # engineered_features and dropped_features share ONE table
    # (FeatureDecisionRecord) with a `kind` discriminator — matching the
    # design note in db/models.py: both are the same Pydantic TYPE
    # (FeatureDecision), just populated into two different list slots.
    feature_decisions = [
        FeatureDecisionRecord(kind="engineered", name=f.name, rationale=f.rationale)
        for f in report.engineered_features
    ] + [
        FeatureDecisionRecord(kind="dropped", name=f.name, rationale=f.rationale)
        for f in report.dropped_features
    ]

    return EDAReportRecord(
        target_distribution_notes=report.target_distribution_notes,
        correlations=[
            FeatureCorrelationRecord(feature_name=name, correlation=corr)
            for name, corr in report.correlation_summary.items()
        ],
        feature_decisions=feature_decisions,
        leakage_warnings=[LeakageWarningRecord(warning_text=w) for w in report.leakage_warnings],
        data_quality_warnings=[
            DataQualityWarningRecord(warning_text=w) for w in report.data_quality_warnings
        ],
        final_feature_names=[
            FinalFeatureNameRecord(feature_name=n) for n in report.final_feature_names
        ],
    )


def _build_tuning_result_record(result: TuningResult) -> TuningResultRecord:
    return TuningResultRecord(
        best_cv_score=result.best_cv_score,
        metric_optimized=result.metric_optimized,
        search_budget_used=result.search_budget_used,
        search_budget_total=result.search_budget_total,
        convergence_notes=result.convergence_notes,
        converged=result.converged,
        best_params=[
            TuningBestParamRecord(param_name=k, param_value=str(v))
            for k, v in result.best_params.items()
        ],
        trial_records=[
            TrialRecordRow(
                trial_number=t.trial_number,
                score=t.score,
                params=[
                    TrialParamRecord(param_name=k, param_value=str(v))
                    for k, v in t.params.items()
                ],
            )
            for t in result.search_history
        ],
    )


def _build_evaluation_report_record(report: EvaluationReport) -> EvaluationReportRecord:
    confusion_cells = []
    if report.confusion_matrix is not None:
        for row_idx, row in enumerate(report.confusion_matrix):
            for col_idx, count in enumerate(row):
                confusion_cells.append(
                    ConfusionMatrixCellRecord(row_idx=row_idx, col_idx=col_idx, count=count)
                )

    residual_entries = []
    if report.residual_summary is not None:
        residual_entries = [
            ResidualSummaryEntryRecord(key=k, value=v)
            for k, v in report.residual_summary.items()
        ]

    failure_analysis_record = None
    if report.failure_analysis is not None:
        failure_analysis_record = FailureAnalysisRecord(
            summary=report.failure_analysis.summary,
            recommended_next_step=report.failure_analysis.recommended_next_step,
        )

    return EvaluationReportRecord(
        metric_optimized=report.metric_optimized,
        metric_value=report.metric_value,
        metric_threshold=report.metric_threshold,
        pass_fail=report.pass_fail,
        model_artifact_path=report.model_artifact_path,
        test_metrics=[
            TestMetricEntryRecord(metric_name=k, value=v) for k, v in report.test_metrics.items()
        ],
        confusion_matrix_cells=confusion_cells,
        residual_summary_entries=residual_entries,
        failure_analysis=failure_analysis_record,
    )


# =====================================================================
# READ PATH — ORM row tree -> Pydantic PipelineRun
# =====================================================================


def load_run_state(session: Session, run_id: str) -> PipelineRun | None:
    """Reconstruct a real PipelineRun from the database. Returns None if
    no run with this run_id exists — mirrors a dict.get()-style contract
    rather than raising, since "run not found" is an expected, routine
    outcome for an API endpoint to handle (e.g. return 404), not an
    exceptional one."""

    run_record = session.get(RunRecord, run_id)
    if run_record is None:
        return None

    pipeline_run = PipelineRun(
        run_id=run_record.run_id,
        status=run_record.status,
        loop_count=run_record.loop_count,
        problem_spec=_rebuild_problem_spec(run_record.problem_spec),
    )

    if run_record.data_profile is not None:
        pipeline_run.data_profile = _rebuild_data_profile(run_record.data_profile)
    if run_record.eda_report is not None:
        pipeline_run.eda_report = _rebuild_eda_report(run_record.eda_report)
    if run_record.tuning_result is not None:
        pipeline_run.tuning_result = _rebuild_tuning_result(run_record.tuning_result)
    if run_record.evaluation_report is not None:
        pipeline_run.evaluation_report = _rebuild_evaluation_report(run_record.evaluation_report)

    # Sort by timestamp, not insertion order — the semantically correct
    # thing to sort a timeline by, robust even if row ids ever got
    # reused or inserted out of order for some reason.
    pipeline_run.history = [
        HistoryEntry(timestamp=e.timestamp, stage=e.stage, event=e.event)
        for e in sorted(run_record.history_entries, key=lambda e: e.timestamp)
    ]
    pipeline_run.trace = [
        TraceEntry(
            timestamp=e.timestamp,
            agent=e.agent,
            model=e.model,
            prompt_tokens=e.prompt_tokens,
            completion_tokens=e.completion_tokens,
            total_tokens=e.total_tokens,
            latency_ms=e.latency_ms,
            estimated_cost_usd=e.estimated_cost_usd,
        )
        for e in sorted(run_record.trace_entries, key=lambda e: e.timestamp)
    ]

    return pipeline_run


def _parse_param_value(raw: str) -> int | float | str:
    """Best-effort numeric coercion for hyperparameter values stored as
    strings (see module docstring's KNOWN LIMITATION). Correct for every
    value this project's actual search space produces (n_estimators,
    max_depth, min_samples_split are always int); falls back to the raw
    string for the theoretical case of a genuinely string-valued param."""
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _rebuild_problem_spec(record: ProblemSpecRecord) -> ProblemSpec:
    constraints = PipelineConstraints(
        max_training_seconds=record.constraints.max_training_seconds,
        max_tuning_trials=record.constraints.max_tuning_trials,
        interpretability_required=record.constraints.interpretability_required,
        max_loop_backs=record.constraints.max_loop_backs,
    )
    return ProblemSpec(
        task_type=record.task_type,
        target_column=record.target_column,
        success_metric=record.success_metric,
        metric_threshold=record.metric_threshold,
        data_source=record.data_source,
        constraints=constraints,
        notes=record.notes,
    )


def _rebuild_data_profile(record: DataProfileRecord) -> DataProfile:
    return DataProfile(
        row_count=record.row_count,
        column_count=record.column_count,
        column_schema=[
            ColumnSchema(name=c.name, dtype=c.dtype, missing_pct=c.missing_pct, is_target=c.is_target)
            for c in record.column_schemas
        ],
        class_balance=(
            {e.class_label: e.proportion for e in record.class_balance_entries} or None
        ),
        outlier_flags=[o.column_name for o in record.outlier_flags],
        cleaning_actions_taken=[
            CleaningAction(column=a.column, action=a.action, rationale=a.rationale)
            for a in record.cleaning_actions
        ],
        warnings=[w.warning_text for w in record.warnings],
        blocking_issue=record.blocking_issue,
    )


def _rebuild_eda_report(record: EDAReportRecord) -> EDAReport:
    engineered = [
        FeatureDecision(name=f.name, rationale=f.rationale)
        for f in record.feature_decisions
        if f.kind == "engineered"
    ]
    dropped = [
        FeatureDecision(name=f.name, rationale=f.rationale)
        for f in record.feature_decisions
        if f.kind == "dropped"
    ]
    return EDAReport(
        correlation_summary={c.feature_name: c.correlation for c in record.correlations},
        engineered_features=engineered,
        dropped_features=dropped,
        leakage_warnings=[w.warning_text for w in record.leakage_warnings],
        data_quality_warnings=[w.warning_text for w in record.data_quality_warnings],
        target_distribution_notes=record.target_distribution_notes,
        final_feature_names=[n.feature_name for n in record.final_feature_names],
    )


def _rebuild_tuning_result(record: TuningResultRecord) -> TuningResult:
    return TuningResult(
        best_params={p.param_name: _parse_param_value(p.param_value) for p in record.best_params},
        best_cv_score=record.best_cv_score,
        metric_optimized=record.metric_optimized,
        search_history=[
            TrialRecord(
                trial_number=t.trial_number,
                params={p.param_name: _parse_param_value(p.param_value) for p in t.params},
                score=t.score,
            )
            for t in sorted(record.trial_records, key=lambda t: t.trial_number)
        ],
        search_budget_used=record.search_budget_used,
        search_budget_total=record.search_budget_total,
        convergence_notes=record.convergence_notes,
        converged=record.converged,
    )


def _rebuild_evaluation_report(record: EvaluationReportRecord) -> EvaluationReport:
    confusion_matrix = None
    if record.confusion_matrix_cells:
        # Reconstruct the 2D shape from individual (row_idx, col_idx,
        # count) rows — the exact reverse of _build_evaluation_report_
        # record()'s enumerate(row)/enumerate(matrix) flattening.
        max_row = max(c.row_idx for c in record.confusion_matrix_cells)
        max_col = max(c.col_idx for c in record.confusion_matrix_cells)
        matrix = [[0] * (max_col + 1) for _ in range(max_row + 1)]
        for c in record.confusion_matrix_cells:
            matrix[c.row_idx][c.col_idx] = c.count
        confusion_matrix = matrix

    residual_summary = None
    if record.residual_summary_entries:
        residual_summary = {e.key: e.value for e in record.residual_summary_entries}

    failure_analysis = None
    if record.failure_analysis is not None:
        failure_analysis = FailureAnalysis(
            summary=record.failure_analysis.summary,
            recommended_next_step=record.failure_analysis.recommended_next_step,
        )

    return EvaluationReport(
        test_metrics={m.metric_name: m.value for m in record.test_metrics},
        confusion_matrix=confusion_matrix,
        residual_summary=residual_summary,
        metric_optimized=record.metric_optimized,
        metric_value=record.metric_value,
        metric_threshold=record.metric_threshold,
        pass_fail=record.pass_fail,
        failure_analysis=failure_analysis,
        model_artifact_path=record.model_artifact_path,
    )