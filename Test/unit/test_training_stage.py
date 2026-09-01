"""
tests/unit/test_training_stage.py

Uses a REAL, fixed dataset (no LLM) but a HAND-CONSTRUCTED TuningResult
for each test — this is deliberate: train_test_split and
RandomForestClassifier both use random_state=42 internally, so given a
fixed X/y and a fixed TuningResult, train_and_evaluate() is FULLY
deterministic. That lets us test the exact failure_analysis branching
logic (revisit_features vs expand_hyperparam_search vs
insufficient_data) precisely, without Optuna's own non-determinism ever
entering the picture.
"""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from pipeline.training_stage import train_and_evaluate
from schemas.problem_spec import ProblemSpec
from schemas.tuning_result import TuningResult


def _small_dataset() -> tuple[pd.DataFrame, pd.Series]:
    raw = load_breast_cancer(as_frame=True)
    X = raw.frame.drop(columns=["target"])
    y = raw.frame["target"]
    return X, y


def _unlearnable_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A dataset with NO relationship between features and target, so any
    model scores around chance (f1 ~= 0.5) no matter how well tuned.

    Needed because the revisit_features branch requires
    gap > 0.1 * threshold, and gap is (threshold - metric_value). On
    breast_cancer a tuned forest scores ~0.96, so even a threshold of
    1.0 leaves a gap of only ~0.04 against a 0.10 cutoff — that branch is
    arithmetically UNREACHABLE on data the model can actually learn. The
    branch means "tuning is done and we are still nowhere near", which by
    definition needs a model that genuinely fails.

    Deterministic by construction (no RNG): the features cycle on a
    period coprime with the alternating target, so they carry no usable
    signal while still having real variance."""
    n = 300
    y = pd.Series([i % 2 for i in range(n)], name="target")
    X = pd.DataFrame({
        "f0": [float(i % 7) for i in range(n)],
        "f1": [float((i * 3) % 11) for i in range(n)],
        "f2": [float((i * 5) % 13) for i in range(n)],
    })
    return X, y


def _spec(metric_threshold: float) -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="target",
        success_metric="f1",
        metric_threshold=metric_threshold,
        data_source="builtin:breast_cancer",
    )


def _tuning_result(*, converged: bool) -> TuningResult:
    """Hand-built TuningResult — real, usable RandomForest params, with
    `converged` stated directly.

    This helper used to fake convergence by juggling
    search_budget_used/total (25 vs 10) because `converged` was a derived
    property reading those two numbers. That indirection was itself a
    symptom of the bug: in production Optuna always uses its whole budget,
    so used < total never held and the property was permanently False.
    `converged` is now a real field set from the plateau heuristic, so the
    helper states the fact plainly and the budget numbers go back to
    describing an ordinary fully-used budget."""
    return TuningResult(
        best_params={"n_estimators": 100, "max_depth": 8, "min_samples_split": 4},
        best_cv_score=0.95,
        # Optuna's own scoring name, unrelated to EvaluationReport's
        # test_metrics keys — those are named for ProblemSpec.success_metric.
        metric_optimized="f1_macro",
        search_history=[],
        search_budget_used=10,
        search_budget_total=10,
        convergence_notes="test fixture",
        converged=converged,
    )


def test_train_and_evaluate_passes_with_achievable_threshold():
    X, y = _small_dataset()
    spec = _spec(metric_threshold=0.50)  # trivially achievable on this dataset

    _model, report = train_and_evaluate(X, y, spec, _tuning_result(converged=True))

    assert report.pass_fail == "pass"
    assert report.failure_analysis is None
    assert report.confusion_matrix is not None
    assert report.residual_summary is None  # classification, not regression


def test_train_and_evaluate_fails_when_search_did_not_converge():
    """THE rule: not converged -> expand_hyperparam_search, regardless of
    how close the score is to the threshold — this is the first branch
    checked in training_stage.py's if/elif chain."""
    X, y = _small_dataset()
    spec = _spec(metric_threshold=0.999)  # unreachable

    _model, report = train_and_evaluate(X, y, spec, _tuning_result(converged=False))

    assert report.pass_fail == "fail"
    assert report.failure_analysis is not None
    assert report.failure_analysis.recommended_next_step == "expand_hyperparam_search"


def test_train_and_evaluate_recommends_revisit_features_on_wide_converged_gap():
    """converged=True + a WIDE gap -> the rule concludes the FEATURES lack
    signal, not the hyperparameters.

    Uses the unlearnable dataset deliberately: this branch fires only when
    gap > 0.1 * threshold, which cannot happen while the model is scoring
    ~0.96. Signal-free features scoring near chance are exactly the
    situation the branch is written for, so this is the honest way to
    reach it — see _unlearnable_dataset()."""
    X, y = _unlearnable_dataset()
    spec = _spec(metric_threshold=0.90)

    _model, report = train_and_evaluate(X, y, spec, _tuning_result(converged=True))

    assert report.pass_fail == "fail"
    # Confirm the premise the branch depends on actually holds here,
    # so a future data change cannot make this pass for the wrong reason.
    gap = spec.metric_threshold - report.metric_value
    assert gap > 0.1 * spec.metric_threshold
    assert report.failure_analysis.recommended_next_step == "revisit_features"


def test_train_and_evaluate_recommends_insufficient_data_on_narrow_converged_gap():
    """converged=True + a threshold only SLIGHTLY above what's achievable
    -> the rule concludes more data is needed, not a pipeline change."""
    X, y = _small_dataset()
    # Real F1 on this dataset with these params is typically ~0.94-0.96 —
    # 0.965 is a narrow, plausible-but-unreached gap from a converged search.
    spec = _spec(metric_threshold=0.965)

    _model, report = train_and_evaluate(X, y, spec, _tuning_result(converged=True))

    # NOTE: this test is sensitive to the exact metric_value train_and_evaluate
    # produces for this fixed dataset/params/random_state — if it fails,
    # print report.metric_value and adjust metric_threshold's gap accordingly
    # rather than assuming the branching logic itself is wrong.
    assert report.pass_fail == "fail"
    assert report.failure_analysis.recommended_next_step == "insufficient_data"

# --------------------------------------------------------------------------
# Metric resolution — regression tests for the silent wrong-metric bug
# --------------------------------------------------------------------------

def _spec_with_metric(success_metric: str, metric_threshold: float = 0.0) -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="target",
        success_metric=success_metric,
        metric_threshold=metric_threshold,
        data_source="builtin:breast_cancer",
    )


def test_classification_reports_every_metric_the_validator_allows():
    """test_metrics must be keyed by the SAME names ProblemSpec accepts for
    classification. It used to hold exactly one entry, "f1_macro", which no
    valid success_metric could ever match — so every lookup missed and fell
    back to f1."""
    X, y = _small_dataset()

    _model, report = train_and_evaluate(
        X, y, _spec_with_metric("f1"), _tuning_result(converged=False)
    )

    assert set(report.test_metrics) == {"f1", "accuracy", "precision", "recall", "roc_auc"}
    assert "f1_macro" not in report.test_metrics
    assert all(isinstance(v, float) for v in report.test_metrics.values())


def test_the_reported_metrics_are_genuinely_different_numbers():
    """The sharpest statement of the bug. Before the fix, asking for
    accuracy, precision, recall or roc_auc returned the F1 score while
    metric_optimized still named what was requested — so every one of
    these values would have been identical. Distinct values are what
    proves each metric is really being computed.

    Note recall is NOT asserted to differ from f1: on this particular
    split the macro recall and macro F1 coincide exactly, which is a real
    property of the data, not a symptom. accuracy and roc_auc cannot
    coincide with f1 here (accuracy weights the classes differently, and
    roc_auc is computed from predict_proba rather than the hard labels),
    so those are the ones that carry the assertion."""
    X, y = _small_dataset()

    _model, report = train_and_evaluate(
        X, y, _spec_with_metric("f1"), _tuning_result(converged=False)
    )

    values = report.test_metrics
    assert values["accuracy"] != values["f1"]
    assert values["roc_auc"] != values["f1"]
    # roc_auc comes from predicted probabilities, not thresholded labels,
    # so it should be meaningfully higher than the label-based scores.
    assert values["roc_auc"] > values["f1"]
    # And at least three genuinely distinct values overall — a fallback
    # that returned f1 for everything would collapse this to one.
    assert len(set(values.values())) >= 3


@pytest.mark.parametrize(
    "success_metric", ["f1", "accuracy", "precision", "recall", "roc_auc"]
)
def test_metric_value_matches_the_requested_metric(success_metric):
    """metric_value must be the metric that metric_optimized names — for
    every metric the validator allows, not just f1. This is the assertion
    that would have caught the original bug: pass/fail was being decided
    on the F1 score while the report claimed to be measuring something
    else."""
    X, y = _small_dataset()

    _model, report = train_and_evaluate(
        X, y, _spec_with_metric(success_metric), _tuning_result(converged=False)
    )

    assert report.metric_optimized == success_metric
    assert report.metric_value == round(report.test_metrics[success_metric], 4)


def test_unknown_metric_raises_instead_of_falling_back():
    """The defensive raise. ProblemSpec's validator should make this
    unreachable, so the spec is mutated AFTER construction to get past it
    (pydantic does not validate on assignment by default).

    The old code would have silently returned the f1 score here and
    reported a pass/fail verdict against it. A loud failure is the correct
    behaviour: if validation is ever bypassed, the run must stop rather
    than produce a confidently wrong number."""
    X, y = _small_dataset()
    spec = _spec_with_metric("f1")
    spec.success_metric = "not_a_real_metric"  # bypasses the field validator

    with pytest.raises(ValueError) as exc_info:
        train_and_evaluate(X, y, spec, _tuning_result(converged=False))

    message = str(exc_info.value)
    assert "not_a_real_metric" in message
    # The message must name what IS available, so the failure is actionable.
    assert "f1" in message and "accuracy" in message


def test_regression_branch_also_uses_explicit_lookup():
    """The regression branch had the same silent-fallback pattern. All
    three of its keys happen to match today, so the bug never bit here —
    but the pattern should not survive in one branch just because it is
    currently harmless."""
    n = 120
    X = pd.DataFrame({"feat": [float(i) for i in range(n)]})
    y = pd.Series([2.0 * i + 1.0 for i in range(n)], name="target")
    spec = ProblemSpec(
        task_type="regression",
        target_column="target",
        success_metric="mae",
        metric_threshold=1e9,  # trivially satisfied; not what is under test
        data_source="builtin:breast_cancer",
    )

    _model, report = train_and_evaluate(X, y, spec, _tuning_result(converged=False))

    assert set(report.test_metrics) == {"rmse", "mae", "r2"}
    assert report.metric_optimized == "mae"
    assert report.metric_value == round(report.test_metrics["mae"], 4)
    assert report.confusion_matrix is None
    assert report.residual_summary is not None


def test_regression_unknown_metric_raises():
    """Same defensive raise on the regression side."""
    n = 60
    X = pd.DataFrame({"feat": [float(i) for i in range(n)]})
    y = pd.Series([1.5 * i for i in range(n)], name="target")
    spec = ProblemSpec(
        task_type="regression",
        target_column="target",
        success_metric="rmse",
        metric_threshold=1e9,
        data_source="builtin:breast_cancer",
    )
    spec.success_metric = "f1"  # a classification metric, bypassing validation

    with pytest.raises(ValueError) as exc_info:
        train_and_evaluate(X, y, spec, _tuning_result(converged=False))

    assert "f1" in str(exc_info.value)
    assert "rmse" in str(exc_info.value)
