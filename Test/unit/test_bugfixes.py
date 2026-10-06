"""
Test/unit/test_bugfixes.py

One test group per bug in the "ML Pipeline Agent — Final Upgrade Plan"
(section "Bugs to fix before anything else"). Each group was run against the
unfixed code first and failed there. No LLM, no API calls.

1. Test-set leak: tuning CV, feature selection and imputation saw every row;
   the train/test split happened later.
2. Convergence check inverted.
3. Tuning ignored the requested metric.
4. Optuna sampler / CV folds unseeded.
5. No .gitignore; .env, uploads/, __pycache__, _t.csv tracked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline.tuning_stage as tuning_stage
import tools.training_tools as training_tools
import tools.tuning_tools as tuning_tools
from pipeline.data_stage import split_and_clean
from pipeline.feature_stage import apply_features, fit_features
from pipeline.metrics import cv_scorer, cv_splitter
from pipeline.run_cache import RunCache
from pipeline.tuning_stage import tune_hyperparameters
from schemas.data_profile import DataProfile
from schemas.problem_spec import ProblemSpec
from tools.feature_tools import engineer_features_tool
from tools.training_tools import train_and_evaluate_tool
from tools.tuning_tools import tune_model_tool

REPO = Path(__file__).resolve().parents[2]

N_BUILTIN = 569                      # sklearn breast cancer rows
N_TEST = 114                         # ceil(0.2 * 569)
N_TRAIN = N_BUILTIN - N_TEST         # 455


def _spec(**overrides) -> ProblemSpec:
    base = dict(
        task_type="classification",
        target_column="y",
        success_metric="f1",
        metric_threshold=0.5,
        data_source="builtin:breast_cancer",
    )
    base.update(overrides)
    return ProblemSpec(**base)


def _profile(n: int) -> DataProfile:
    return DataProfile(row_count=n, column_count=1, column_schema=[])


def _xy(n: int = 60, task: str = "classification"):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] > 0).astype(int) if task == "classification" else X["a"] * 2 + X["b"])
    return X, y


# ==========================================================================
# Bug 1 — test-set leak
# ==========================================================================

def test_imputation_uses_train_statistics_not_all_rows():
    n = 100
    ids = np.arange(n)
    x = (ids.astype(float) + 1) ** 2          # skewed, so train and all-row medians differ
    x[::7] = np.nan
    raw = pd.DataFrame({"row_id": ids, "x": x, "y": ids % 2})

    train_df, test_df = split_and_clean(raw, _spec())

    assert len(train_df) + len(test_df) == n
    assert set(train_df["row_id"]).isdisjoint(set(test_df["row_id"]))

    observed = raw.dropna(subset=["x"])
    train_median = observed[observed["row_id"].isin(train_df["row_id"])]["x"].median()
    assert train_median != observed["x"].median(), "fixture must make the medians differ"

    for frame in (train_df, test_df):
        assert frame["x"].notna().all()
        was_missing = raw.set_index("row_id").loc[frame["row_id"], "x"].isna().to_numpy()
        assert (frame.loc[was_missing, "x"] == train_median).all()


def test_split_is_stratified_and_deterministic():
    raw = pd.DataFrame({"a": np.arange(200.0), "y": [0] * 180 + [1] * 20})

    train_1, test_1 = split_and_clean(raw, _spec())
    train_2, test_2 = split_and_clean(raw, _spec())

    assert test_1["y"].sum() == 4                      # 20% of the 20 positives
    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(test_1, test_2)


def test_apply_features_replays_the_trained_columns_on_test_rows():
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "colour": ["red", "blue"] * 4,
            "flag": [True, False] * 4,
            "when": pd.to_datetime(["2024-01-01"] * 4 + ["2024-06-01"] * 4),
            "y": [0, 1] * 4,
        }
    )
    test = pd.DataFrame(
        {
            "num": [2.5, 7.5],
            "colour": ["green", "red"],        # "green" never appeared in train
            "flag": [False, True],
            "when": pd.to_datetime(["2024-03-03", "2024-09-09"]),
            "y": [1, 0],
        }
    )

    X_train, _y, _report, plan = fit_features(train, _profile(len(train)), _spec())
    X_test, y_test = apply_features(test, plan, _spec())

    assert list(X_test.columns) == list(X_train.columns)
    assert list(y_test) == [1, 0] and len(X_test) == 2
    assert not X_test.isna().any().any()
    colour_cols = [c for c in X_test.columns if c.startswith("colour_")]
    assert colour_cols and (X_test.iloc[0][colour_cols] == 0).all()


def test_feature_report_is_computed_from_train_rows_only():
    report = engineer_features_tool(
        data_source="builtin:breast_cancer", target_column="diagnosis", task_type="classification"
    )
    assert f"{N_TRAIN} labeled rows" in report.target_distribution_notes


def test_feature_tool_caches_train_and_test_separately():
    cache = RunCache()
    engineer_features_tool(
        data_source="builtin:breast_cancer", target_column="diagnosis",
        task_type="classification", cache=cache, run_id="r1",
    )
    assert len(cache.get("r1", "X")) == len(cache.get("r1", "y")) == N_TRAIN
    assert len(cache.get("r1", "X_test")) == len(cache.get("r1", "y_test")) == N_TEST
    assert list(cache.get("r1", "X_test").columns) == list(cache.get("r1", "X").columns)


def test_tuning_tool_sees_train_rows_only(monkeypatch):
    seen = {}
    real = tuning_tools.tune_hyperparameters
    monkeypatch.setattr(
        tuning_tools, "tune_hyperparameters",
        lambda X, y, spec: seen.setdefault("rows", len(X)) and real(X, y, spec),
    )
    tune_model_tool(
        data_source="builtin:breast_cancer", target_column="diagnosis",
        task_type="classification", max_tuning_trials=2,
    )
    assert seen["rows"] == N_TRAIN


def test_training_tool_tunes_on_train_and_evaluates_on_the_held_out_rows(monkeypatch):
    seen = {}
    real = training_tools.tune_hyperparameters

    def spy(X, y, spec):
        seen["rows"] = len(X)
        return real(X, y, spec)

    monkeypatch.setattr(training_tools, "tune_hyperparameters", spy)
    report = train_and_evaluate_tool(
        data_source="builtin:breast_cancer", target_column="diagnosis",
        task_type="classification", success_metric="f1", metric_threshold=0.5,
        max_tuning_trials=2,
    )
    assert seen["rows"] == N_TRAIN
    assert sum(sum(row) for row in report.confusion_matrix) == N_TEST


def test_training_tool_uses_cached_test_rows_without_resplitting():
    cache = RunCache()
    engineer_features_tool(
        data_source="builtin:breast_cancer", target_column="diagnosis",
        task_type="classification", cache=cache, run_id="r2",
    )
    report = train_and_evaluate_tool(
        data_source="builtin:this_source_does_not_exist",   # would raise if reloaded
        target_column="diagnosis", task_type="classification",
        success_metric="f1", metric_threshold=0.5, max_tuning_trials=2,
        cache=cache, run_id="r2",
    )
    assert sum(sum(row) for row in report.confusion_matrix) == N_TEST


# ==========================================================================
# Bug 2 — convergence check was inverted
# ==========================================================================

def _scripted_scores(monkeypatch, scores):
    it = iter(scores)
    monkeypatch.setattr(
        tuning_stage, "cross_val_score",
        lambda model, X, y, cv, scoring: np.array([next(it)] * 3),
    )


def _tune(n_trials):
    X, y = _xy()
    spec = _spec()
    spec.constraints.max_tuning_trials = n_trials
    return tune_hyperparameters(X, y, spec)


def test_search_that_peaked_early_is_converged(monkeypatch):
    _scripted_scores(monkeypatch, [0.90] + [0.80] * 9)      # best is trial 0 of 10
    result = _tune(10)
    assert result.converged is True
    assert "plateaued" in result.convergence_notes.lower()


def test_search_still_improving_at_the_end_is_not_converged(monkeypatch):
    _scripted_scores(monkeypatch, [0.50 + 0.01 * i for i in range(10)])   # best is the last trial
    result = _tune(10)
    assert result.converged is False
    assert "plateaued" not in result.convergence_notes.lower()


def test_improvement_inside_the_final_fifth_is_not_converged(monkeypatch):
    _scripted_scores(monkeypatch, [0.80] * 8 + [0.95, 0.80])   # best lands in the last 20%
    assert _tune(10).converged is False


# ==========================================================================
# Bug 3 — tuning ignored the requested metric
# ==========================================================================

@pytest.mark.parametrize(
    "task, metric, scorer",
    [
        ("classification", "roc_auc", "roc_auc"),
        ("classification", "recall", "recall_macro"),
        ("classification", "precision", "precision_macro"),
        ("classification", "accuracy", "accuracy"),
        ("classification", "f1", "f1_macro"),
        ("regression", "r2", "r2"),
        ("regression", "mae", "neg_mean_absolute_error"),
        ("regression", "rmse", "neg_root_mean_squared_error"),
    ],
)
def test_tuning_optimises_the_requested_metric(monkeypatch, task, metric, scorer):
    seen = []
    monkeypatch.setattr(
        tuning_stage, "cross_val_score",
        lambda model, X, y, cv, scoring: seen.append(scoring) or np.array([0.5] * 3),
    )
    X, y = _xy(task=task)
    spec = _spec(task_type=task, success_metric=metric)
    spec.constraints.max_tuning_trials = 2
    result = tune_hyperparameters(X, y, spec)

    assert set(seen) == {scorer}
    assert result.metric_optimized == scorer


def test_unknown_metric_has_no_scorer():
    with pytest.raises(ValueError):
        cv_scorer("not_a_metric")


def test_tuning_tool_forwards_the_requested_metric(monkeypatch):
    seen = {}
    X, y = _xy()
    cache = RunCache()
    cache.set("r", "X", X)
    cache.set("r", "y", y)
    real = tuning_tools.tune_hyperparameters
    monkeypatch.setattr(
        tuning_tools, "tune_hyperparameters",
        lambda X, y, spec: seen.setdefault("metric", spec.success_metric) and real(X, y, spec),
    )
    tune_model_tool(
        data_source="x.csv", target_column="y", task_type="classification",
        max_tuning_trials=2, cache=cache, run_id="r", success_metric="roc_auc",
    )
    assert seen["metric"] == "roc_auc"


# ==========================================================================
# Bug 4 — Optuna sampler and CV folds unseeded
# ==========================================================================

def test_same_data_and_spec_give_the_same_search():
    X, y = _xy(n=80)
    spec = _spec()
    spec.constraints.max_tuning_trials = 6

    first = tune_hyperparameters(X, y, spec)
    second = tune_hyperparameters(X, y, spec)

    assert first.best_params == second.best_params
    assert [t.params for t in first.search_history] == [t.params for t in second.search_history]
    assert [t.score for t in first.search_history] == [t.score for t in second.search_history]


def test_cv_folds_are_shuffled_and_seeded():
    for task in ("classification", "regression"):
        splitter = cv_splitter(task)
        assert splitter.shuffle is True and splitter.random_state is not None


# ==========================================================================
# Bug 5 — .gitignore / tracked secrets and junk
# ==========================================================================

def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
@pytest.mark.parametrize(
    "path",
    [".env", "uploads/x.csv", "pipeline/__pycache__/a.pyc", "_t.csv", ".venv/bin/python", "mlruns/0/meta.yaml"],
)
def test_gitignore_covers_secrets_and_junk(path):
    assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_nothing_sensitive_or_generated_is_tracked():
    tracked = _git("ls-files").stdout.splitlines()
    bad = [
        p for p in tracked
        if p == ".env" or p == "_t.csv" or p.startswith("uploads/") or "__pycache__" in p
    ]
    assert bad == []


def test_env_example_documents_the_key_without_a_real_value():
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in text
    assert "sk-" not in text
