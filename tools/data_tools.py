"""
tools/data_tools.py

LLM-callable tool wrappers around pipeline/data_stage.py.

These functions are the ONLY interface an agent is allowed to touch. They
take flat, primitive arguments (what an LLM's function-calling API can
actually populate), validate everything with Pydantic, and delegate all
real computation to the existing Phase-1 functions. No pandas logic is
duplicated here — this file is a validated boundary, not an implementation.
"""
from __future__ import annotations

from typing import Literal

from pipeline.data_stage import clean_and_profile, load_data
from pipeline.run_cache import RunCache
from schemas.data_profile import DataProfile
from schemas.problem_spec import ProblemSpec

class PipelineBlockedError(Exception):
    """Raised when a stage's output signals the pipeline cannot proceed
    (e.g. DataProfile.blocking_issue is set).

    A generic ValueError would get silently caught by broad exception
    handlers elsewhere in an agent loop; a distinct type lets calling
    code catch this SPECIFIC case and route to escalation deliberately,
    rather than accidentally swallowing it alongside unrelated bugs.
    """
    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage}] pipeline blocked: {reason}")

def profile_dataset_tool(
    data_source: str,
    target_column: str,
    task_type: Literal["classification", "regression"],
    cache: RunCache | None = None,
    run_id: str | None = None,

) -> DataProfile:
    """LLM-callable tool: load and profile a dataset.

    Parameters
    ----------
    data_source: str
        Where the raw data lives (e.g. "builtin:breast_cancer" or a
        CSV/parquet path). Passed straight through to load_data().
    target_column: str
        Name of the label column. clean_and_profile() branches on this
        for missingness checks — must be the real value, never a dummy.
    task_type: "classification" | "regression"
        Now a REAL parameter, not a hardcoded dummy — clean_and_profile()
        only computes class_balance when task_type == "classification",
        so a wrong value here would silently produce a meaningless
        class_balance block for regression data. Making the caller state
        this explicitly (rather than guessing inside the tool) means the
        tool's behavior is fully determined by its actual inputs, not by
        an assumption buried in the implementation.

    Returns
    -------
    DataProfile
        The exact same schema clean_and_profile() already produces —
        no new shape invented here.

    Raises
    ------
    PipelineBlockedError
        If the resulting DataProfile has blocking_issue set. Converts a
        condition that would otherwise just be "a field on a returned
        object the caller might forget to check" into something the
        calling agent physically cannot ignore without an explicit
        try/except.
    """
    # Build a minimal ProblemSpec — clean_and_profile() requires the full
    # object, but this tool only exposes 3 flat args to the LLM. task_type
    # now comes from the REAL caller-supplied value (Option B), not a
    # hardcoded guess — this is the fix for the open question from before.
    #
    # success_metric/metric_threshold are still dummies, but that's fine:
    # unlike task_type, clean_and_profile() never reads either of them —
    # only train_and_evaluate() does, several stages downstream of this
    # tool. A dummy here has zero effect on what this specific tool does.

    """

    cache, run_id : optional
        If BOTH are provided, the cleaned DataFrame this tool computes
        internally is stashed in the cache under run_id, so a later
        stage's tool (e.g. engineer_features_tool) can reuse it instead
        of recomputing load_data()+clean_and_profile() from scratch.
        Neither parameter changes this tool's return value or its
        behavior when omitted — an LLM or a standalone test calling this
        with just the original 3 arguments sees IDENTICAL behavior to
        before this change.
    """
    spec = ProblemSpec(
        task_type = task_type,
        target_column = target_column,
        success_metric="accuracy" if task_type == "classification" else "rmse",
        metric_threshold =0.0,
        data_source= data_source,
    )

    raw_df= load_data(spec)
    cleaned_df, profile = clean_and_profile(raw_df, spec)

    if profile.blocking_issue is not None:
        raise PipelineBlockedError(stage="data_agent", reason=profile.blocking_issue)
    
    if cache is not None and run_id is not None:
        cache.set(run_id, "cleaned_df", cleaned_df)

    return profile