"""
Phase 1 — deterministic Hyperparameter Tuning stage. No LLM here — the
search SPACE and BUDGET are fixed constants for now. Phase 3's Tuning
Agent will eventually choose those intelligently; today they're hardcoded,
same principle as every other Phase 1 stage.
"""

from __future__ import annotations

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from pipeline.metrics import SEED, cv_scorer, cv_splitter

from schemas.problem_spec import ProblemSpec
from schemas.tuning_result import TrialRecord, TuningResult

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _build_model(task_type: str, trial: optuna.Trial):
    """Given one Optuna trial, propose a set of hyperparameters and build
    the corresponding model. Returns (model, params) so the caller can log
    exactly which params produced which score."""
    params = {
        # trial.suggest_int(name, low, high) tells Optuna "propose an
        # integer in this range for this trial" — Optuna decides the
        # actual value based on what it's learned from prior trials.
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": 42,  # fixed seed — every trial's randomness is
        # reproducible; without this, re-running the same trial's params
        # could give a slightly different score purely from randomness,
        # muddying the comparison between trials.
    }

    model_cls = (
        RandomForestClassifier if task_type == "classification" else RandomForestRegressor
    )

    return model_cls(**params), params

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, spec: ProblemSpec) -> TuningResult:
    """Search for good RandomForest hyperparameters via cross-validated
    Optuna trials. Never touches a held-out test set — that's reserved
    for training_stage.py."""

    n_trials = spec.constraints.max_tuning_trials or 25
    
    # Classification and regression need DIFFERENT scoring functions —
    # "f1_macro" is meaningless for continuous targets, and RMSE is
    # meaningless for class labels. This is the same task_type branching
    # pattern as class_balance in data_stage.py.

    # The scorer comes from the metric the user asked for (pipeline/metrics.py),
    # so the search optimises the same thing the Training stage later judges.
    scoring = cv_scorer(spec.success_metric)

     # Why "neg_" RMSE: scikit-learn's cross_val_score always maximizes the
    # scoring function internally. RMSE is an error metric — lower is
    # better — so sklearn provides a negated version specifically so that
    # "maximize the score" and "minimize the error" become the same
    # instruction. Without the negation, cross_val_score would think a
    # WORSE (higher) RMSE was a better result.

    history: list[TrialRecord] =[]

    def objective(trial: optuna.Trial) -> float:
        """Optuna calls this once per trial. Whatever this function
        returns is the number Optuna is trying to maximize."""
        model, params = _build_model(spec.task_type, trial)
        scores = cross_val_score(
            model, X, y, cv=cv_splitter(spec.task_type), scoring=scoring
        )
        mean_score = float(scores.mean())
        # Log this trial immediately, not after the whole search finishes —
        # if the search crashes partway through, you still have a record
        # of every trial that completed.
        history.append(
            TrialRecord(trial_number=trial.number, params=params, score=mean_score)
        )
        return mean_score

    # Seeded sampler: same data + same spec => same search, run after run.
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trials_used = len(study.trials)

    # Convergence heuristic: look at the last 20% of trials — if the best
    # score among them matches the overall best score (within floating-
    # point tolerance), the search likely plateaued rather than running
    # out of budget mid-improvement. This is intentionally simple —
    # rule-based, deterministic, no LLM — same spirit as
    # training_stage.py's rule-based failure_analysis in Phase 1.

    # Converged = the best score was already reached BEFORE the final 20% of
    # the trials, i.e. the last stretch of the search found nothing better.
    # The earlier check was inverted: it called the search converged when the
    # best score sat INSIDE the last 20% — exactly when it was still improving.
    tail = max(1, best_trials_used // 5)
    earlier = [t.score for t in history[:-tail]]
    overall_best = study.best_value
    plateaued = bool(earlier) and max(earlier) >= overall_best - 1e-9

    convergence_notes = (
        "score plateaued: the best result came before the final 20% of "
        "trials - likely converged."
        if plateaued
        else "Score was still improving near the trial budget limit - "
        "consider a larger search_budget_total next run."
    )

    return TuningResult(
        best_params = study.best_params,
        best_cv_score = float(study.best_value),
        metric_optimized = scoring,
        # Every completed trial, not just the winner. `history` was being
        # built correctly and then dropped on the floor here, leaving
        # TuningResult.search_history permanently empty — so the per-trial
        # record the objective() function goes out of its way to append
        # immediately (specifically so a mid-search crash still leaves
        # evidence) never actually reached the caller.
        search_history = history,
        converged = plateaued,
        search_budget_used = best_trials_used,
        search_budget_total = n_trials,
        convergence_notes = convergence_notes,
    )