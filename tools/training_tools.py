"""
tools/training_tools.py

LLM-callable tool wrapper around pipeline/training_stage.py.

COST WARNING (bigger than feature_tools.py's or tuning_tools.py's limitation):
this tool runs the ENTIRE Phase-1 chain internally — load_data(),
clean_and_profile(), engineer_features(), tune_hyperparameters(), AND
train_and_evaluate() — because train_and_evaluate() needs a TuningResult,
and there's no shared state store yet to reuse a prior stage's output.
This is the most expensive single tool call in the pipeline. Worth fixing
once PipelineRun-backed caching exists (orchestrator phase); until then,
calling this tool means re-running the full hyperparameter search every
single time, not just re-cleaning data.
"""

from __future__ import annotations

from typing import Literal, Optional

from pipeline.data_stage import clean_and_profile, load_data, split_and_clean
from pipeline.feature_stage import apply_features, fit_features
from pipeline.run_cache import RunCache
from pipeline.training_stage import train_and_evaluate
from pipeline.tuning_stage import tune_hyperparameters
from schemas.evaluation_report import EvaluationReport
from schemas.problem_spec import PipelineConstraints, ProblemSpec
from schemas.tuning_result import TuningResult
from tools.data_tools import PipelineBlockedError

def train_and_evaluate_tool(
    data_source: str,
    target_column: str,
    task_type: Literal["classification", "regression"],
    success_metric: str,
    metric_threshold: float,
    max_tuning_trials: Optional[int] = 25,
    cache: RunCache | None = None,
    run_id: str | None = None,
    tuning_result: TuningResult | None = None,
) -> EvaluationReport:
    """LLM-callable tool: run the full pipeline through final model
    training and testing.

    Parameters
    ----------
    data_source, target_column, task_type
        Same meaning as every other tool in this project.
    success_metric : str
        Must be valid for task_type (enforced by ProblemSpec's own
        validator — same _metric_matches_task check every other tool
        has relied on implicitly via its dummy values, except here it's
        NOT a dummy: this is the real metric the Training stage judges
        pass/fail against, so it must be a real, deliberate argument,
        not a placeholder like the other tools used.
    metric_threshold : float
        Same reasoning — this tool is the first one where the threshold
        actually matters to the computation, not just a dummy 0.0.
    max_tuning_trials : int, optional
        Passed straight through to the internal tuning step.

    Returns
    -------
    EvaluationReport
        Test metrics, pass/fail, and (if failed) failure_analysis —
        the exact object run_pipeline.py's loop-back logic reads.

    Raises
    ------
    PipelineBlockedError
        If the underlying data is unusable before any of the downstream
        stages get a chance to run.
    """
    """
    cache, run_id : optional
        Same X/y reuse pattern as tune_model_tool() — skips data cleaning
        and feature engineering if a prior stage already cached them.
    tuning_result : optional
        If provided (typically pipeline_run.tuning_result from a prior
        TuningAgent call in this run), the internal tune_hyperparameters()
        call is SKIPPED ENTIRELY — this is the expensive part this tool
        was flagged for recomputing. If omitted, falls back to running
        tuning internally, exactly as before.
    """
    spec = ProblemSpec(
        task_type=task_type,
        target_column=target_column,
        success_metric=success_metric,
        metric_threshold=metric_threshold,
        data_source=data_source,
        constraints=PipelineConstraints(max_tuning_trials=max_tuning_trials),
    )

    X = y = X_test = y_test = None
    if cache is not None and run_id is not None:
        X = cache.get(run_id, "X")
        y = cache.get(run_id, "y")
        X_test = cache.get(run_id, "X_test")
        y_test = cache.get(run_id, "y_test")

    # A hit needs the train AND the test rows the feature stage produced;
    # anything less recomputes, so train and test always come from one split.
    if X is None or y is None or X_test is None or y_test is None:
        raw_df = load_data(spec)
        cleaned_df, profile = clean_and_profile(raw_df, spec, impute=False)
        if profile.blocking_issue is not None:
            raise PipelineBlockedError(stage="training_agent", reason=profile.blocking_issue)
        train_df, test_df = split_and_clean(cleaned_df, spec)
        X, y, _report, plan = fit_features(train_df, profile, spec)
        X_test, y_test = apply_features(test_df, plan, spec)

    if tuning_result is None:
        tuning_result = tune_hyperparameters(X, y, spec)
    _model, eval_report = train_and_evaluate(
        X, y, spec, tuning_result, X_test=X_test, y_test=y_test
    )

    return eval_report