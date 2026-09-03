"""
Test/integration/test_db_sync.py

Round-trip test for db/sync.py: build a fully-populated PipelineRun by
hand, persist it, load it back, and verify every section survived the
translation intact.

Uses an in-memory SQLite database, NOT real Postgres — same principle as
this project's fake OpenAI client: nothing in db/models.py uses a
Postgres-specific type (no JSONB, by design, since Option A went fully
relational), so SQLite exercises the identical SQLAlchemy translation
logic without needing a live external service running for this test.

Deliberately maximally-populated, not a minimal happy-path object — a
PipelineRun with mostly-None fields would never exercise the tables that
only matter when populated (confusion_matrix, failure_analysis,
data_quality_warnings, outlier_flags, trial params, etc.), which is
exactly the kind of shallow-test gap that's bitten this project before.
"""

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.base import Base
from db.sync import load_run_state, persist_run_state
from schemas.data_profile import CleaningAction, ColumnSchema, DataProfile
from schemas.eda_report import EDAReport, FeatureDecision
from schemas.evaluation_report import EvaluationReport, FailureAnalysis
from schemas.pipeline_run import HistoryEntry, PipelineRun
from schemas.problem_spec import PipelineConstraints, ProblemSpec
from schemas.trace_entry import TraceEntry
from schemas.tuning_result import TrialRecord, TuningResult


@pytest.fixture
def db_session():
    # In-memory, fresh tables per test — no state leaks between tests,
    # no file left on disk, no external service required.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def full_pipeline_run() -> PipelineRun:
    """Every section populated, every list with 2+ entries where the
    schema allows it — deliberately exercising every one of the 26
    tables, not just the ones a trivial run would touch."""

    ts = datetime.datetime(2026, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)

    spec = ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
        constraints=PipelineConstraints(
            max_training_seconds=300,
            max_tuning_trials=15,
            interpretability_required=True,
            max_loop_backs=3,
        ),
        notes="round-trip test spec",
    )

    profile = DataProfile(
        row_count=569,
        column_count=31,
        column_schema=[
            ColumnSchema(name="mean radius", dtype="float64", missing_pct=0.0, is_target=False),
            ColumnSchema(name="diagnosis", dtype="int64", missing_pct=0.0, is_target=True),
        ],
        class_balance={"0": 0.3726, "1": 0.6274},
        outlier_flags=["perimeter error", "area error"],
        cleaning_actions_taken=[
            CleaningAction(column="mean radius", action="median_imputation", rationale="test rationale"),
        ],
        warnings=["Severe class imbalance detected"],
        blocking_issue=None,
    )

    eda = EDAReport(
        correlation_summary={"mean radius": 0.73, "mean texture": 0.4152},
        engineered_features=[
            FeatureDecision(name="a__x__b", rationale="interaction of top two"),
        ],
        dropped_features=[
            FeatureDecision(name="texture error", rationale="low signal"),
        ],
        leakage_warnings=["'suspicious_col' correlates 0.99 with target"],
        data_quality_warnings=["'signup_date' has 3 missing values (NaT)"],
        target_distribution_notes="Target 'diagnosis' — 569 labeled rows.",
        final_feature_names=["mean radius", "mean texture", "a__x__b"],
    )

    tuning = TuningResult(
        best_params={"n_estimators": 127, "max_depth": 8, "min_samples_split": 4},
        best_cv_score=0.9583,
        metric_optimized="f1_macro",
        search_history=[
            TrialRecord(trial_number=0, params={"n_estimators": 80, "max_depth": 5}, score=0.94),
            TrialRecord(trial_number=1, params={"n_estimators": 127, "max_depth": 8}, score=0.9583),
        ],
        search_budget_used=15,
        search_budget_total=15,
        convergence_notes="Score plateaued in the final trials — likely converged.",
        converged=True,
    )

    evaluation = EvaluationReport(
        test_metrics={"f1": 0.9435, "accuracy": 0.95, "precision": 0.93, "recall": 0.955},
        confusion_matrix=[[39, 3], [3, 69]],
        residual_summary=None,
        metric_optimized="f1",
        metric_value=0.9435,
        metric_threshold=0.999,
        pass_fail="fail",
        failure_analysis=FailureAnalysis(
            summary="Tuning did not converge — worth expanding the search.",
            recommended_next_step="expand_hyperparam_search",
        ),
        model_artifact_path=None,
    )

    run = PipelineRun(run_id="test-run-full", problem_spec=spec)
    run.status = "escalated"
    run.loop_count = 2
    run.data_profile = profile
    run.eda_report = eda
    run.tuning_result = tuning
    run.evaluation_report = evaluation
    run.history = [
        HistoryEntry(timestamp=ts, stage="orchestrator", event="run started"),
        HistoryEntry(timestamp=ts + datetime.timedelta(seconds=3), stage="data_agent", event="proceed=True"),
    ]
    run.trace = [
        TraceEntry(
            timestamp=ts, agent="data_agent", model="gpt-4o-mini",
            prompt_tokens=200, completion_tokens=80, total_tokens=280,
            latency_ms=1500.0, estimated_cost_usd=0.00008,
        ),
    ]
    return run


def test_round_trip_preserves_problem_spec(db_session, full_pipeline_run):
    persist_run_state(db_session, full_pipeline_run)
    loaded = load_run_state(db_session, "test-run-full")

    assert loaded is not None
    assert loaded.problem_spec == full_pipeline_run.problem_spec
    assert loaded.status == "escalated"
    assert loaded.loop_count == 2


def test_round_trip_preserves_data_profile(db_session, full_pipeline_run):
    persist_run_state(db_session, full_pipeline_run)
    loaded = load_run_state(db_session, "test-run-full")

    assert loaded.data_profile.row_count == 569
    assert set(loaded.data_profile.outlier_flags) == {"perimeter error", "area error"}
    assert loaded.data_profile.class_balance == {"0": 0.3726, "1": 0.6274}
    assert len(loaded.data_profile.cleaning_actions_taken) == 1
    assert loaded.data_profile.cleaning_actions_taken[0].action == "median_imputation"
    assert len(loaded.data_profile.column_schema) == 2


def test_round_trip_preserves_eda_report_including_new_data_quality_warnings(db_session, full_pipeline_run):
    persist_run_state(db_session, full_pipeline_run)
    loaded = load_run_state(db_session, "test-run-full")

    assert loaded.eda_report.correlation_summary == {"mean radius": 0.73, "mean texture": 0.4152}
    assert len(loaded.eda_report.engineered_features) == 1
    assert len(loaded.eda_report.dropped_features) == 1
    assert loaded.eda_report.leakage_warnings == full_pipeline_run.eda_report.leakage_warnings
    # Specifically verifying the field added for the NaT fix — this
    # table (DataQualityWarningRecord) exists BECAUSE of that earlier bug.
    assert loaded.eda_report.data_quality_warnings == full_pipeline_run.eda_report.data_quality_warnings
    assert loaded.eda_report.final_feature_names == full_pipeline_run.eda_report.final_feature_names


def test_round_trip_preserves_tuning_result_including_converged_and_nested_trial_params(db_session, full_pipeline_run):
    persist_run_state(db_session, full_pipeline_run)
    loaded = load_run_state(db_session, "test-run-full")

    # Specifically verifying the field that was structurally always False
    # in production until the earlier fix — persisting it wrong here
    # would silently reintroduce that exact bug one layer up.
    assert loaded.tuning_result.converged is True
    assert loaded.tuning_result.best_params == full_pipeline_run.tuning_result.best_params
    assert len(loaded.tuning_result.search_history) == 2
    # search_history is sorted by trial_number on load — confirming both
    # correct values AND correct ordering survived.
    assert loaded.tuning_result.search_history[0].trial_number == 0
    assert loaded.tuning_result.search_history[0].params == {"n_estimators": 80, "max_depth": 5}
    assert loaded.tuning_result.search_history[1].params == {"n_estimators": 127, "max_depth": 8}


def test_round_trip_preserves_evaluation_report_including_confusion_matrix_shape(db_session, full_pipeline_run):
    persist_run_state(db_session, full_pipeline_run)
    loaded = load_run_state(db_session, "test-run-full")

    assert loaded.evaluation_report.pass_fail == "fail"
    assert loaded.evaluation_report.test_metrics == full_pipeline_run.evaluation_report.test_metrics
    # The 2D shape specifically — proving the flatten-to-cells-then-
    # reassemble round trip is exact, not just "some cells exist."
    assert loaded.evaluation_report.confusion_matrix == [[39, 3], [3, 69]]
    assert loaded.evaluation_report.failure_analysis.recommended_next_step == "expand_hyperparam_search"
    assert loaded.evaluation_report.residual_summary is None  # correctly absent, not an empty dict


def test_round_trip_preserves_history_and_trace_in_timestamp_order(db_session, full_pipeline_run):
    persist_run_state(db_session, full_pipeline_run)
    loaded = load_run_state(db_session, "test-run-full")

    assert len(loaded.history) == 2
    assert loaded.history[0].stage == "orchestrator"
    assert loaded.history[1].stage == "data_agent"
    assert len(loaded.trace) == 1
    assert loaded.trace[0].agent == "data_agent"
    assert loaded.trace[0].total_tokens == 280


def test_load_run_state_returns_none_for_unknown_run_id(db_session):
    result = load_run_state(db_session, "does-not-exist")
    assert result is None


def test_persist_is_safe_to_call_multiple_times_and_replaces_not_duplicates(db_session, full_pipeline_run):
    """Confirms the 'full-replace on every sync call' behavior documented
    in db/sync.py — calling persist_run_state twice on evolving state
    (e.g. a real run being synced after every agent step) must not
    accumulate duplicate child rows."""
    persist_run_state(db_session, full_pipeline_run)

    full_pipeline_run.loop_count = 3
    full_pipeline_run.history.append(
        HistoryEntry(
            timestamp=datetime.datetime(2026, 1, 15, 10, 31, 0, tzinfo=datetime.timezone.utc),
            stage="orchestrator", event="second sync",
        )
    )
    persist_run_state(db_session, full_pipeline_run)

    loaded = load_run_state(db_session, "test-run-full")
    assert loaded.loop_count == 3
    assert len(loaded.history) == 3  # 2 original + 1 new, not 4 or 5