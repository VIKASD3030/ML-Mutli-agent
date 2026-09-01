"""
Test/unit/test_training_tools.py

UNIT tests — no LLM anywhere in this chain, so sklearn runs for real.

train_and_evaluate_tool() is the most expensive call in the project: its
own module docstring warns that it runs load, clean, engineer, tune AND
train internally. Two optional escape hatches exist to avoid that cost —
the X/y cache, and a pre-computed tuning_result — and BOTH are the kind of
optimisation that fails silently. Skip them by accident and every number
the tool returns is still correct; the run just quietly pays for a second
full hyperparameter search. That is precisely the shape of bug a test has
to catch, because no assertion on the RESULT will ever notice it.

So the tests below verify the tuning skip two independent ways:

  1. PARAMS FIDELITY (no mocking) — hand in a TuningResult whose
     best_params are impossible for the real search to produce
     (n_estimators=7, far outside the 50-300 range tuning_stage.py
     samples from) and confirm the tool's output matches a direct
     train_and_evaluate() call using those exact params. If a fresh
     search had run, different params would have trained the model.

  2. CALL DETECTION (monkeypatch) — replace tune_hyperparameters with a
     counter. This is not mocking behaviour under test; the behaviour
     under test is literally "is this collaborator called or not", and a
     counter is the only direct way to observe it. Test 1 stands alone
     without it; the two together pin both the fact and the mechanism.

The pass/fail tests deliberately use the REAL breast-cancer features
rather than a synthetic set, because pass_fail is a judgement about
realistic model performance and a trivially separable toy set would score
1.0 and make the 'fail' case unreachable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.run_cache import RunCache
from pipeline.training_stage import train_and_evaluate
from schemas.evaluation_report import EvaluationReport
from schemas.problem_spec import PipelineConstraints, ProblemSpec
from schemas.tuning_result import TuningResult
from tools.data_tools import PipelineBlockedError
from tools.feature_tools import engineer_features_tool
from tools.training_tools import train_and_evaluate_tool
import tools.training_tools as training_tools

UNLOADABLE_SOURCE = "builtin:this_source_does_not_exist"


def _distinct_tuning_result(n_estimators: int = 7) -> TuningResult:
    """A TuningResult the real search could never return. tuning_stage.py
    samples n_estimators from suggest_int(50, 300), so 7 is outside the
    space by construction — which is what makes it usable as a fingerprint
    for 'these params, and not freshly-searched ones, trained the model'.
    Small on purpose too: 7 trees fit almost instantly."""
    return TuningResult(
        best_params={"n_estimators": n_estimators, "max_depth": 3, "min_samples_split": 2},
        best_cv_score=0.5,
        metric_optimized="f1_macro",
        search_budget_used=10,
        search_budget_total=10,
        convergence_notes="hand-built for testing",
        # converged=False so the failure branch routes to
        # expand_hyperparam_search — the same route these tests saw before
        # `converged` became a real field (the old derived property was
        # used < total, i.e. 10 < 10, i.e. always False). Convergence is
        # not what this file is testing; keeping the pre-existing route
        # means the pass/fail tests below still exercise exactly the
        # behaviour they were written for.
        converged=False,
    )


@pytest.fixture(scope="module")
def real_features() -> tuple[pd.DataFrame, pd.Series]:
    """The genuine engineered X/y for breast_cancer, computed once for the
    whole module. Built through the real feature tool so these tests train
    on exactly what the pipeline would hand the training stage."""
    cache = RunCache()
    run_id = "module-features"
    engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )
    return cache.get(run_id, "X"), cache.get(run_id, "y")


@pytest.fixture
def cache_with_features(real_features) -> tuple[RunCache, str]:
    """A fresh cache per test, pre-loaded with the shared real features —
    the state engineer_features_tool() leaves behind in a live run."""
    X, y = real_features
    cache = RunCache()
    run_id = "run-training-tools"
    cache.set(run_id, "X", X)
    cache.set(run_id, "y", y)
    return cache, run_id


def _spec(metric_threshold: float) -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=metric_threshold,
        data_source="builtin:breast_cancer",
        constraints=PipelineConstraints(max_tuning_trials=2),
    )


# --------------------------------------------------------------------------
# pass / fail
# --------------------------------------------------------------------------

def test_realistic_threshold_returns_pass_with_no_failure_analysis(cache_with_features):
    """A reachable threshold must produce pass_fail='pass' AND leave
    failure_analysis as None. The second half matters as much as the
    first: run_pipeline.py only reads failure_analysis on the failure
    branch, so a stray populated value on a passing run would be a
    latent trap for any future code that checks the field first."""
    cache, run_id = cache_with_features

    report = train_and_evaluate_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.90,
        cache=cache,
        run_id=run_id,
        tuning_result=_distinct_tuning_result(n_estimators=50),
    )

    assert isinstance(report, EvaluationReport)
    assert report.pass_fail == "pass"
    assert report.passed is True
    assert report.failure_analysis is None
    assert report.metric_value >= report.metric_threshold
    # Classification => confusion matrix present, residuals absent.
    assert report.confusion_matrix is not None
    assert report.residual_summary is None


def test_unreachable_threshold_returns_fail_with_populated_failure_analysis(cache_with_features):
    """0.999 f1 is not reachable on this dataset, so the tool must report
    a failure AND attach the rule-based failure_analysis the orchestrator
    routes on. recommended_next_step is asserted only as 'one of the legal
    values' — which specific branch fires is training_stage.py's decision
    and is pinned in test_training_stage.py, not here."""
    cache, run_id = cache_with_features

    report = train_and_evaluate_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.999,
        cache=cache,
        run_id=run_id,
        tuning_result=_distinct_tuning_result(n_estimators=50),
    )

    assert report.pass_fail == "fail"
    assert report.passed is False
    assert report.failure_analysis is not None
    assert report.failure_analysis.summary
    assert report.failure_analysis.recommended_next_step in {
        "revisit_features",
        "expand_hyperparam_search",
        "insufficient_data",
        "bad_spec",
    }


# --------------------------------------------------------------------------
# The tuning skip — proof #1, params fidelity, no mocking
# --------------------------------------------------------------------------

def test_supplied_tuning_result_actually_trains_the_model(cache_with_features, real_features):
    """The regression test for the doubled-Optuna bug, proved WITHOUT any
    patching. The tool is handed best_params the real search cannot
    produce (n_estimators=7). We then run train_and_evaluate() directly
    with that same TuningResult and the same X/y — both paths are fully
    deterministic (train_test_split and the forest are both random_state
    =42), so an exact match on metric_value proves the tool trained on
    the SUPPLIED params. Had it re-run the search, the model would have
    had 50-300 trees and the numbers would not line up."""
    cache, run_id = cache_with_features
    X, y = real_features
    tuning = _distinct_tuning_result(n_estimators=7)

    from_tool = train_and_evaluate_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.90,
        cache=cache,
        run_id=run_id,
        tuning_result=tuning,
    )

    _model, expected = train_and_evaluate(X, y, _spec(0.90), tuning)

    assert from_tool.metric_value == expected.metric_value
    assert from_tool.test_metrics == expected.test_metrics
    assert from_tool.confusion_matrix == expected.confusion_matrix


def test_supplied_tuning_result_skips_the_expensive_recompute(cache_with_features):
    """The other half of the shortcut: with X/y cached AND a tuning_result
    supplied, the tool should touch neither load_data() nor the search.
    data_source points at an unloadable source, so completing at all is
    proof that the entire load/clean/engineer/tune prefix was skipped."""
    cache, run_id = cache_with_features

    report = train_and_evaluate_tool(
        data_source=UNLOADABLE_SOURCE,  # would raise ValueError if loaded
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.90,
        cache=cache,
        run_id=run_id,
        tuning_result=_distinct_tuning_result(),
    )

    assert isinstance(report, EvaluationReport)
    assert report.pass_fail in {"pass", "fail"}


# --------------------------------------------------------------------------
# The tuning skip — proof #2, direct call detection
# --------------------------------------------------------------------------

@pytest.fixture
def tuning_spy(monkeypatch):
    """Counts calls to tune_hyperparameters and forwards to the real
    implementation, so behaviour is unchanged and only observability is
    added. See the module docstring for why a counter is the right tool
    for this specific assertion."""
    real = training_tools.tune_hyperparameters
    calls: list[int] = []

    def counting(X, y, spec):
        calls.append(1)
        return real(X, y, spec)

    monkeypatch.setattr(training_tools, "tune_hyperparameters", counting)
    return calls


def test_tune_hyperparameters_is_never_called_when_tuning_result_supplied(
    cache_with_features, tuning_spy
):
    """Direct statement of the bug that was fixed: passing a tuning_result
    must result in ZERO searches. Before the fix, the pipeline paid for
    the Tuning Agent's search and then immediately paid for an identical
    one inside the Training Agent."""
    cache, run_id = cache_with_features

    train_and_evaluate_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.90,
        cache=cache,
        run_id=run_id,
        tuning_result=_distinct_tuning_result(),
    )

    assert len(tuning_spy) == 0


def test_omitted_tuning_result_falls_back_to_running_tuning(cache_with_features, tuning_spy):
    """The fallback must still work — the skip is an optimisation, not a
    new requirement. Called standalone (as an LLM would), the tool has no
    prior TuningResult and must search exactly once."""
    cache, run_id = cache_with_features

    report = train_and_evaluate_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.90,
        max_tuning_trials=2,
        cache=cache,
        run_id=run_id,
        # tuning_result deliberately omitted
    )

    assert len(tuning_spy) == 1
    assert isinstance(report, EvaluationReport)


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------

def test_cache_miss_recomputes_everything_and_still_works():
    """No cache, no tuning_result — the full expensive path. Kept to two
    trials so the suite stays quick, but it must genuinely run end to end,
    because this is what a bare LLM tool call does."""
    report = train_and_evaluate_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=0.90,
        max_tuning_trials=2,
    )

    assert isinstance(report, EvaluationReport)
    assert report.metric_optimized == "f1"
    assert report.metric_threshold == 0.90


def test_cache_hit_requires_both_x_and_y(real_features):
    """Half a hit must fall through to the recompute branch, which against
    an unloadable source surfaces as ValueError rather than reaching
    train_and_evaluate() with y=None."""
    X, _y = real_features
    cache = RunCache()
    run_id = "run-half-cached"
    cache.set(run_id, "X", X)  # deliberately no "y"

    with pytest.raises(ValueError):
        train_and_evaluate_tool(
            data_source=UNLOADABLE_SOURCE,
            target_column="diagnosis",
            task_type="classification",
            success_metric="f1",
            metric_threshold=0.90,
            cache=cache,
            run_id=run_id,
            tuning_result=_distinct_tuning_result(),
        )


# --------------------------------------------------------------------------
# Validation and escalation
# --------------------------------------------------------------------------

def test_metric_must_match_task_type():
    """success_metric is the first REAL (non-dummy) metric argument in the
    tool chain — the tool's own docstring makes a point of it. ProblemSpec's
    _metric_matches_task validator has to reject a regression metric on a
    classification task, and it must do so before any training happens."""
    with pytest.raises(Exception) as exc_info:
        train_and_evaluate_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
            success_metric="rmse",  # regression metric, wrong task
            metric_threshold=0.90,
            max_tuning_trials=2,
        )

    assert "rmse" in str(exc_info.value)


def test_blocking_issue_raises_with_training_agent_stage(monkeypatch):
    """Escalation contract, with this tool's own stage string. As
    elsewhere, load_data is substituted only to supply unusable INPUT —
    see test_data_tools.py's module docstring."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(training_tools, "load_data", lambda spec: df_without_target)

    with pytest.raises(PipelineBlockedError) as exc_info:
        train_and_evaluate_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
            success_metric="f1",
            metric_threshold=0.90,
            max_tuning_trials=2,
        )

    assert exc_info.value.stage == "training_agent"
    assert "diagnosis" in exc_info.value.reason
