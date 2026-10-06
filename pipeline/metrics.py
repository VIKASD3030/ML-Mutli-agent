"""
pipeline/metrics.py

One place that maps the metric a user asked for (ProblemSpec.success_metric)
to the scikit-learn CV scorer used while tuning, and that fixes the seed and
fold layout every CV run uses.

Before this existed, tuning_stage.py always optimised f1_macro (or RMSE)
whatever the spec said, so a run asking for roc_auc was tuned on one metric
and judged on another; and the CV folds / Optuna sampler were unseeded, so
two runs on the same data gave different results.
"""

from __future__ import annotations

from sklearn.model_selection import KFold, StratifiedKFold

SEED = 42
CV_FOLDS = 3

# spec metric -> sklearn scorer name. Higher is always better for a scorer,
# which is why error metrics use sklearn's negated variants.
CV_SCORERS: dict[str, str] = {
    "accuracy": "accuracy",
    "f1": "f1_macro",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "roc_auc": "roc_auc",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def cv_scorer(success_metric: str) -> str:
    """The CV scorer that optimises exactly `success_metric`. Raises for an
    unknown metric rather than quietly substituting a different one."""
    try:
        return CV_SCORERS[success_metric]
    except KeyError:
        raise ValueError(
            f"No CV scorer for success_metric '{success_metric}'. "
            f"Known metrics: {sorted(CV_SCORERS)}."
        ) from None


def cv_splitter(task_type: str, seed: int = SEED):
    """Shuffled, seeded folds: stratified for classification."""
    if task_type == "classification":
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
