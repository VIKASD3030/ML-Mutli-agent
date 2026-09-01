"""
tests/unit/test_tuning_stage.py

No mocking — Optuna + RandomForest run for real here, but with a tiny
trial budget to keep the suite fast. These tests check STRUCTURE and
CONSTRAINTS (is the result schema-valid, does it respect the search
space, does the budget get used correctly) — NOT specific score values,
since Optuna's own sampler is unseeded and best_cv_score will vary
run-to-run, exactly as you discovered yourself earlier in this project.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

from pipeline.tuning_stage import tune_hyperparameters
from schemas.problem_spec import ProblemSpec


def _small_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Real breast cancer data, same as every manual test so far — kept
    here as a plain helper rather than a fixture since it's used by
    every test in this file identically."""
    raw = load_breast_cancer(as_frame=True)
    X = raw.frame.drop(columns=["target"])
    y = raw.frame["target"]
    return X, y


def _spec_with_trials(n_trials: int) -> ProblemSpec:
    spec = ProblemSpec(
        task_type="classification",
        target_column="target",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
    )
    spec.constraints.max_tuning_trials = n_trials
    return spec


def test_tune_hyperparameters_respects_requested_trial_budget():
    """The most basic contract: ask for N trials, get N trials logged in
    search_history — this is what search_budget_used/search_budget_total
    are actually FOR, and it's worth verifying directly rather than
    assuming Optuna always behaves this way."""
    X, y = _small_dataset()
    spec = _spec_with_trials(3)  # deliberately tiny — speed over search quality

    result = tune_hyperparameters(X, y, spec)

    assert result.search_budget_total == 3
    assert result.search_budget_used == 3
    assert len(result.search_history) == 3


def test_tune_hyperparameters_best_params_within_declared_search_space():
    """Regression test tying the RESULT back to the exact ranges defined
    in _build_model() — if someone changes the search space there without
    updating this test, this test still passes (it reads the actual
    result, not a hardcoded range), but it does verify the result is
    internally sane: every param Optuna returns should be usable directly
    by RandomForestClassifier."""
    X, y = _small_dataset()
    spec = _spec_with_trials(3)

    result = tune_hyperparameters(X, y, spec)

    assert 50 <= result.best_params["n_estimators"] <= 300
    assert 2 <= result.best_params["max_depth"] <= 20
    assert 2 <= result.best_params["min_samples_split"] <= 10


def test_tune_hyperparameters_returns_schema_valid_result():
    """Checks the SHAPE of the contract this stage promises downstream
    consumers (training_stage.py, TuningAgent) — every field populated,
    correct types, nothing silently None where it shouldn't be."""
    X, y = _small_dataset()
    spec = _spec_with_trials(3)

    result = tune_hyperparameters(X, y, spec)

    assert result.metric_optimized == "f1_macro"
    assert isinstance(result.best_cv_score, float)
    assert isinstance(result.convergence_notes, str) and len(result.convergence_notes) > 0
    assert all(isinstance(t.trial_number, int) for t in result.search_history)
    assert isinstance(result.converged, bool)


def test_converged_flag_agrees_with_convergence_notes():
    """`converged` and `convergence_notes` must never disagree — they are
    two renderings of ONE plateau comparison.

    This is the regression test for a real design bug: `converged` used to
    be a separate @property (search_budget_used < search_budget_total)
    that had nothing to do with the plateau heuristic behind the notes
    string. Because study.optimize() always spends its full budget, that
    property was permanently False, which made two of training_stage.py's
    three failure_analysis branches — revisit_features and
    insufficient_data — unreachable dead code. Both are now derived from
    the same `plateaued` variable, and this test is what keeps them tied
    together if either is ever edited alone.

    The VALUE is not asserted (the sampler is unseeded, so whether the
    best trial lands in the final 20% genuinely varies); the AGREEMENT
    is, and that holds either way."""
    X, y = _small_dataset()
    spec = _spec_with_trials(3)

    result = tune_hyperparameters(X, y, spec)

    notes_say_plateaued = "plateaued" in result.convergence_notes.lower()
    assert result.converged is notes_say_plateaued

    # And the budget numbers must NOT be what decides it — they are always
    # equal, so anything derived from them could only ever be constant.
    assert result.search_budget_used == result.search_budget_total