"""
TuningResult — produced by the Hyperparameter Tuning Agent (Agent 4).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrialRecord(BaseModel):
    trial_number: int
    params: dict[str, Any]
    score: float


class TuningResult(BaseModel):
    best_params: dict[str, Any]
    best_cv_score: float
    metric_optimized: str
    search_history: list[TrialRecord] = Field(default_factory=list)
    search_budget_used: int = Field(
        ..., description="Number of trials actually run."
    )
    search_budget_total: int = Field(
        ..., description="Number of trials allowed by ProblemSpec.constraints."
    )
    convergence_notes: str = ""
    converged: bool = Field(
        ...,
        description="Did the hyperparameter search plateau, or was it still "
        "improving when it ran out of budget? Set by tune_hyperparameters() "
        "from the same last-20%-of-trials plateau comparison that produces "
        "convergence_notes.",
    )
    # WHY THIS IS A REQUIRED FIELD AND NOT A DERIVED PROPERTY
    #
    # This used to be `@property converged -> search_budget_used <
    # search_budget_total`. That expression can never be True in production:
    # tuning_stage.py calls study.optimize(objective, n_trials=n_trials) with
    # no early-stopping callback, so Optuna always runs exactly n_trials and
    # search_budget_used == search_budget_total on every real run.
    #
    # training_stage.py's three-way failure_analysis ladder branches on this
    # value, so the two branches requiring converged=True — revisit_features
    # and insufficient_data — were unreachable dead code for the entire
    # project, and every failed run routed to expand_hyperparam_search
    # regardless of what had actually gone wrong.
    #
    # Making it a REQUIRED field (no default) is deliberate: convergence is a
    # fact about a specific search that only the code running that search can
    # know. A default would let a caller silently construct a TuningResult
    # that claims something about a search it never observed — which is the
    # same class of quiet-wrong-answer bug this replaced.
