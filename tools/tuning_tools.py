"""
tools/tuning_tools.py

LLM-callable tool wrapper around pipeline/tuning_stage.py.

KNOWN LIMITATION (deliberate, not accidental): like feature_tools.py, this tool
recomputes load_data(), clean_and_profile(), and engineer_features() internally
rather than reusing previous stage outputs because there is no shared state store yet.
"""

from __future__ import annotations

from typing import Literal, Optional

from pipeline.data_stage import clean_and_profile, load_data
from pipeline.feature_stage import engineer_features
from pipeline.run_cache import RunCache
from pipeline.tuning_stage import tune_hyperparameters
from schemas.problem_spec import PipelineConstraints, ProblemSpec
from schemas.tuning_result import TuningResult
from tools.data_tools import PipelineBlockedError


def tune_model_tool(
    data_source: str,
    target_column: str,
    task_type: Literal["classification", "regression"],
    max_tuning_trials: Optional[int] = 25,
    cache: RunCache | None = None,
    run_id: str | None = None,
) -> TuningResult:
    """LLM-callable tool: load & clean data, engineer features, then search for optimal
    hyperparameters via cross-validated Optuna trials.

    Parameters
    ----------
    data_source, target_column, task_type
        Identical meaning to other pipeline tools — simple flat arguments.
    max_tuning_trials : int, optional
        Maximum trial budget allowed for hyperparameter search (default: 25).

    Returns
    -------
    TuningResult
        Best hyperparameter parameters, best CV score, metric optimized, and convergence notes.

    Raises
    ------
    PipelineBlockedError
        If the underlying data itself is unusable before hyperparameter search begins.
    """
    """
    cache, run_id : optional
        If both are provided AND the cache already has "X"/"y" cached
        under run_id (from a prior Feature Agent call in this run), those
        are reused directly — skipping load_data(), clean_and_profile(),
        AND engineer_features() entirely. If the cache is empty, missing,
        or either param is None, this falls back to the full recompute
        exactly as before. This fallback matters: a standalone test call
        (no cache) must behave identically to today, and even inside a
        real pipeline run, a cache MISS should degrade gracefully to a
        correct (if wasteful) result — never raise, never silently use
        stale/wrong data.
    """
    spec = ProblemSpec(
        task_type=task_type,
        target_column=target_column,
        success_metric="accuracy" if task_type == "classification" else "rmse",
        metric_threshold=0.0,
        data_source=data_source,
        constraints=PipelineConstraints(max_tuning_trials=max_tuning_trials),
    )

    X = y = None
    if cache is not None and run_id is not None:
        X = cache.get(run_id, "X")
        y = cache.get(run_id, "y")

    if X is None or y is None:
        raw_df = load_data(spec)
        cleaned_df, profile = clean_and_profile(raw_df, spec)
        if profile.blocking_issue is not None:
            raise PipelineBlockedError(stage="tuning_agent", reason=profile.blocking_issue)

        X, y, _report = engineer_features(cleaned_df, profile, spec)

    return tune_hyperparameters(X, y, spec)
