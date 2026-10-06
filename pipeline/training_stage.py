"""
pipeline/training_stage.py

Phase 1 — deterministic Model Training & Testing stage. No LLM here.

`failure_analysis` is rule-based in Phase 1 — a deliberate stand-in for
what becomes an LLM judgment call in Phase 3, kept rule-based for now so
the whole pipeline stays deterministic and testable without any API calls.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from schemas.evaluation_report import EvaluationReport, FailureAnalysis
from schemas.problem_spec import ProblemSpec
from schemas.tuning_result import TuningResult

HIGHER_IS_BETTER = {"accuracy", "f1", "precision", "recall", "roc_auc", "r2"}

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ProblemSpec,
    tuning: TuningResult,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> tuple[object, EvaluationReport]:
    """Fit on (X, y) and evaluate on the held-out (X_test, y_test).

    The pipeline makes its one train/test split up front (data_stage.
    split_and_clean) and hands the test rows in here, so tuning and feature
    selection never saw them. If no test rows are supplied this falls back to
    splitting X/y itself — only for direct callers that did no earlier work
    on those rows; the pipeline never relies on it.
    """
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if spec.task_type == "classification" else None,
        )
    else:
        X_train, y_train = X, y

    # tuning.best_params came from Optuna as plain floats/ints for each
    # hyperparameter — we copy them into a fresh dict rather than mutating
    # tuning.best_params directly, then add random_state for reproducibility,
    # same discipline as _build_model() in tuning_stage.py.
    params = {k: v for k, v in tuning.best_params.items()}
    params["random_state"] = 42
    model_cls = RandomForestClassifier if spec.task_type == "classification" else RandomForestRegressor
    model = model_cls(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    confusion = None
    residual_summary = None

    if spec.task_type == "classification":
        # Every metric ProblemSpec's validator accepts for classification is
        # computed here, keyed by the EXACT name the validator allows. The
        # dict used to hold a single "f1_macro" entry, which no valid
        # success_metric could ever match — see the explicit lookup below.
        f1 = float(f1_score(y_test, preds, average="macro"))
        test_metrics = {
            "f1": f1,
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(
                precision_score(y_test, preds, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, preds, average="macro", zero_division=0)
            ),
        }
        confusion = confusion_matrix(y_test, preds).tolist()

        # roc_auc is the one metric that cannot always be produced: it needs
        # predicted probabilities for a binary target. A model without
        # predict_proba raises AttributeError; a non-binary target makes
        # roc_auc_score raise ValueError. In either case the metric is simply
        # OMITTED rather than filled with a guess — a missing key turns into
        # a loud error at the lookup below if someone actually asked for it,
        # which is the honest outcome. Inventing a value would not be.
        try:
            proba = model.predict_proba(X_test)[:, 1]
            test_metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except (AttributeError, ValueError):
            pass
    else:
        # root mean squared error. Uses root_mean_squared_error() rather
        # than mean_squared_error(squared=False): the `squared` parameter
        # was deprecated in scikit-learn 1.4 and REMOVED in 1.6, so the old
        # call raises TypeError on this project's pinned sklearn (1.9).
        # The regression branch had no test coverage until now, which is
        # why a hard crash on every regression run went unnoticed.
        rmse = float(root_mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        test_metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        residuals = (y_test.values - preds)
        residual_summary = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
        }

    # Explicit lookup, no fallback, for BOTH branches. The previous
    # `.get(spec.success_metric, <default>)` silently substituted a
    # different metric whenever the requested key was absent, while
    # metric_optimized below still reported the name that was asked for —
    # so a run could report "precision = 0.94, PASS" having actually
    # measured f1. A wrong pass/fail verdict with no warning is far worse
    # than a crash.
    #
    # This should be unreachable: ProblemSpec's _metric_matches_task
    # validator already restricts success_metric to exactly these keys per
    # task_type. It is kept as a real raise rather than an assert because
    # the cost of being wrong here is a silently invalid result, and the
    # regression branch is now held to the same rule as classification even
    # though all three of its keys happen to match today.
    if spec.success_metric not in test_metrics:
        raise ValueError(
            f"success_metric '{spec.success_metric}' was not computed for "
            f"task_type '{spec.task_type}'. Available metrics: "
            f"{sorted(test_metrics)}."
        )
    metric_value = test_metrics[spec.success_metric]


    # Same HIGHER_IS_BETTER lookup pattern as tuning_stage.py's task_type
    # branching — one boolean decides which comparison direction to use.

    higher_is_better = spec.success_metric in HIGHER_IS_BETTER
    if higher_is_better:
        passed = metric_value >= spec.metric_threshold
    else:
        passed = metric_value <= spec.metric_threshold

    failure_analysis = None
    if not passed:
        # gap: how far off the threshold we are, expressed as a positive
        # number regardless of direction — lets one formula work whether
        # the metric is "higher is better" or "lower is better".
        gap = (
            spec.metric_threshold - metric_value
            if higher_is_better
            else metric_value - spec.metric_threshold
        )

        # this is the loop-back decision itself - three rules, in order:
        if tuning.converged and gap > 0.1 * abs(spec.metric_threshold or 1):
            # Tuning found its best answer already (converged) but we're
            # STILL far from the threshold — more tuning won't help, the
            # features probably lack the signal needed. Route to features.
            step = "revisit_features"
            reason = (
                f"Tuning converged (best_cv_score={tuning.best_cv_score:.4f}) but test "
                f"performance still misses the threshold by a wide margin — the feature "
                f"set likely lacks sufficient signal, not the hyperparameters."
            )

        elif not tuning.converged:
            # Tuning ran out of budget while still improving — worth
            # giving it more trials before concluding the features are insufficient.
            step = "expand_hyperparam_search"
            reason = (
                "Tuning did not converge within its trial budget — worth expanding the "
                "search before concluding the features are insufficient."
            )

        else:
            step = "insufficient_data"
            reason = (
                f"Close to threshold (gap={gap:.4f}) with converged tuning and reasonable "
                "features — likely needs more training data rather than pipeline changes."
            )
        failure_analysis = FailureAnalysis(summary=reason, recommended_next_step=step)

    report = EvaluationReport(
        test_metrics=test_metrics,
        confusion_matrix=confusion,
        residual_summary=residual_summary,
        metric_optimized=spec.success_metric,
        metric_value=round(float(metric_value), 4),
        metric_threshold=float(spec.metric_threshold),
        pass_fail="pass" if passed else "fail",
        failure_analysis=failure_analysis,
    )
    return model, report