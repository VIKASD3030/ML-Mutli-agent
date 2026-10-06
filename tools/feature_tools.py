"""
tools/feature_tools.py

LLM-callable tool wrapper around pipeline/feature_stage.py.

KNOWN LIMITATION (deliberate, not accidental): this tool recomputes
load_data() + clean_and_profile() internally rather than reusing a
previous Data Agent call's result, because there's no shared state store
yet — that's the orchestrator/state-machine phase's job, further down
the roadmap. For a single agent call this is fine; if a real pipeline run
calls the Data Agent AND the Feature Agent back to back, cleaning
currently happens twice. Worth fixing once PipelineRun-backed caching
exists, not before.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from pipeline.data_stage import clean_and_profile, load_data, split_and_clean
from pipeline.feature_stage import apply_features, fit_features
from pipeline.run_cache import RunCache
from schemas.data_profile import DataProfile
from schemas.eda_report import EDAReport
from schemas.problem_spec import ProblemSpec
from tools.data_tools import PipelineBlockedError

# Reusing PipelineBlockedError from data_tools.py rather than defining a
# second copy — one exception type for "the pipeline can't proceed",
# regardless of which stage raised it. The Orchestrator (later) only
# needs to catch one type, not one per stage.

def engineer_features_tool(
    data_source: str,
    target_column: str,
    task_type: Literal["classification", "regression"],
    cache: RunCache | None = None,
    run_id: str | None = None,
    profile : DataProfile | None = None,
) -> EDAReport:
    """LLM-callable tool: clean the data, then engineer/filter features.

    Parameters
    ----------
    data_source, target_column, task_type
        Identical meaning to profile_dataset_tool()'s parameters — same
        flat-args philosophy, same reason (LLM function-calling needs
        simple types, not a DataFrame or a nested object).

    Returns
    -------
    EDAReport
        Correlation summary, engineered/dropped feature decisions, and
        crucially leakage_warnings — the field a Feature Agent's whole
        judgment is built around. Note: the actual feature matrix (X) is
        NOT returned here. An agent reasoning about feature quality only
        needs the REPORT describing what happened, never the raw
        numbers — same "LLM never touches raw data" boundary as the
        Data tool, just applied one stage later.

    Raises
    ------
    PipelineBlockedError
        If the underlying data itself is unusable (blocking_issue set)
        before feature engineering even gets a chance to run.
    """
    # Rebuild a minimal ProblemSpec — same plumbing pattern as
    # profile_dataset_tool(). Dummies for success_metric/metric_threshold
    # are safe here for the same reason as before: neither
    # clean_and_profile() nor engineer_features() reads them.

    spec = ProblemSpec(
        task_type=task_type,
        target_column=target_column,
        success_metric="accuracy" if task_type == "classification" else "rmse",
        metric_threshold=0.0,
        data_source=data_source,
    )

    cleaned_df = None
    if cache is not None and run_id is not None:
        cleaned_df = cache.get(run_id, "cleaned_df")

    # Only a genuine cache hit if we have BOTH pieces we need — cleaned_df
    # from the cache AND profile passed in. Missing either means we can't
    # skip recompute, so treat it as a miss rather than half-using stale data.
    if cleaned_df is None or profile is None:
        raw_df = load_data(spec)
        # impute=False: the frame kept for splitting must not carry fill
        # values computed from test rows (the profile still describes them).
        cleaned_df, profile = clean_and_profile(raw_df, spec, impute=False)
        if profile.blocking_issue is not None:
            raise PipelineBlockedError(stage="feature_agent", reason=profile.blocking_issue)
        # Populate the cache even on a miss, so a LATER call in the same
        # run (or a retry) can still benefit — mirrors what Data Agent's
        # tool already does for "cleaned_df".
        if cache is not None and run_id is not None:
            cache.set(run_id, "cleaned_df", cleaned_df)

    # One split, then everything is decided on train rows only; the test
    # rows are only transformed with what train taught.
    train_df, test_df = split_and_clean(cleaned_df, spec)
    X, y, report, plan = fit_features(train_df, profile, spec)
    X_test, y_test = apply_features(test_df, plan, spec)

    # THE FIX for bug #3 — this is what makes Tuning/Training's cache
    # reads actually hit, ever. Without this, every downstream "if cache
    # is not None..." check I gave you would silently always miss.
    if cache is not None and run_id is not None:
        cache.set(run_id, "X", X)
        cache.set(run_id, "y", y)
        cache.set(run_id, "X_test", X_test)
        cache.set(run_id, "y_test", y_test)

    return report