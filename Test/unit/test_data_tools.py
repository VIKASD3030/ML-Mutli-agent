"""
Test/unit/test_data_tools.py

UNIT tests — tools/data_tools.py contains no LLM call at all, so nothing
here is faked and nothing here costs money. profile_dataset_tool() is a
validated boundary, not an implementation: it builds a ProblemSpec from
flat args, delegates to load_data()/clean_and_profile(), and converts one
specific field (blocking_issue) into a raised exception. Those four
responsibilities — validate, delegate, cache, escalate — are exactly what
these tests pin, because each one is a place where the tool can silently
stop doing its job without any downstream test noticing.

ONE DELIBERATE SEAM: the blocking-issue tests monkeypatch
tools.data_tools.load_data. That is NOT mocking the behaviour under test —
the behaviour under test is "profile.blocking_issue set => raise
PipelineBlockedError", and load_data() only supports the single hardcoded
source "builtin:breast_cancer", whose target column is renamed into
existence on every load. There is therefore no real data_source that can
produce unusable-but-loadable data. The patch substitutes the INPUT
FIXTURE (a DataFrame), then lets the real clean_and_profile() and the real
tool logic run untouched.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from pipeline.run_cache import RunCache
from schemas.data_profile import DataProfile
from tools.data_tools import PipelineBlockedError, profile_dataset_tool
import tools.data_tools as data_tools


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_profile_dataset_tool_returns_valid_dataprofile():
    """A valid classification call must return a real, fully-populated
    DataProfile — not a partially-filled one. row_count/column_count come
    from the cleaned frame, and class_balance must be populated because
    task_type is 'classification' (it stays None for regression)."""
    profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
    )

    assert isinstance(profile, DataProfile)
    assert profile.blocking_issue is None
    assert profile.is_usable is True
    assert profile.row_count > 0
    assert profile.column_count > 0
    # Classification => class_balance populated, and it must be a real
    # distribution (proportions summing to ~1.0), not an empty dict.
    assert profile.class_balance is not None
    assert pytest.approx(sum(profile.class_balance.values()), abs=1e-3) == 1.0


def test_profile_dataset_tool_populates_class_balance_only_for_classification():
    """task_type is a REAL parameter here, not a dummy — the tool's own
    docstring calls this out. clean_and_profile() only computes
    class_balance when task_type == 'classification', so this is the
    assertion that proves the caller-supplied value is actually reaching
    the stage function rather than being hardcoded inside the tool."""
    profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="regression",
    )

    assert profile.class_balance is None


# --------------------------------------------------------------------------
# Validation happens BEFORE any data loading
# --------------------------------------------------------------------------

def test_invalid_task_type_raises_validation_error_before_loading_data():
    """The Literal type hint is documentation, not enforcement — nothing
    checks it at runtime. What actually rejects a typo is ProblemSpec's
    own validation, and it does so while BUILDING the spec, i.e. before
    load_data() is ever reached.

    The proof that ordering is correct: data_source is deliberately
    nonsense here. If the tool loaded data first, we would get ValueError
    from load_data(). Getting ValidationError instead is what pins the
    'validate before you do expensive work' ordering."""
    with pytest.raises(ValidationError):
        profile_dataset_tool(
            data_source="builtin:definitely_not_a_real_source",
            target_column="diagnosis",
            task_type="clasification",  # deliberate typo
        )


def test_unrecognized_data_source_raises_value_error():
    """A well-formed spec pointing at an unsupported source must fail
    loudly from load_data(), with the offending source named — NOT be
    converted into a PipelineBlockedError. This distinction matters: a
    PipelineBlockedError means 'the data is unusable, escalate to a
    human', while this is 'the caller asked for something that does not
    exist', which is a bug, not a data-quality finding. The agents catch
    only the former, so conflating them would turn a programming error
    into a silent escalation."""
    with pytest.raises(ValueError) as exc_info:
        profile_dataset_tool(
            data_source="builtin:some_regression_set",
            target_column="price",
            task_type="regression",
        )

    assert "some_regression_set" in str(exc_info.value)
    # Explicitly NOT the escalation type.
    assert not isinstance(exc_info.value, PipelineBlockedError)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

def test_cleaned_df_is_cached_when_cache_and_run_id_provided():
    """The whole point of the optional cache/run_id params: the cleaned
    DataFrame this tool computes internally must be left behind for
    engineer_features_tool() to pick up, under the exact key 'cleaned_df'
    and the exact run_id given. A wrong key here fails silently — the
    downstream tool just recomputes and nobody notices except the clock."""
    cache = RunCache()
    run_id = "run-abc-123"

    profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )

    cached = cache.get(run_id, "cleaned_df")
    assert isinstance(cached, pd.DataFrame)
    # The cached frame must be the CLEANED one the profile describes,
    # not the raw load — row_count is the cheapest way to prove that.
    assert len(cached) == profile.row_count
    assert "diagnosis" in cached.columns
    # Scoped to this run_id only — a different run must not see it.
    assert cache.get("some-other-run", "cleaned_df") is None


@pytest.mark.parametrize(
    "cache, run_id",
    [
        (None, "run-abc-123"),  # run_id without a cache
        (RunCache(), None),     # cache without a run_id
        (None, None),           # neither
    ],
    ids=["run_id_only", "cache_only", "neither"],
)
def test_partial_cache_arguments_do_not_break_the_tool(cache, run_id):
    """Both params are required for caching to happen; supplying only one
    must degrade to plain uncached behaviour rather than raising. This is
    the 'a standalone call sees IDENTICAL behaviour' promise the tool's
    docstring makes, and it is what lets every other test in this suite
    call the tool with three arguments."""
    profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )

    assert profile.blocking_issue is None
    if cache is not None and run_id is not None:
        pass  # not reachable with these params; guard kept for clarity
    elif cache is not None:
        assert cache.get("anything", "cleaned_df") is None


# --------------------------------------------------------------------------
# Escalation: blocking_issue -> PipelineBlockedError
# --------------------------------------------------------------------------

def _patch_load_data_to_return(monkeypatch, df: pd.DataFrame) -> None:
    """Substitute the INPUT to the tool, not its logic. See the module
    docstring for why this seam is unavoidable."""
    monkeypatch.setattr(data_tools, "load_data", lambda spec: df)


def test_blocking_issue_raises_pipeline_blocked_error_with_stage_and_reason(monkeypatch):
    """The single most important line in this tool: a blocking_issue must
    become a RAISED exception, not a field on a returned object the caller
    might forget to check. The stage name is asserted too, because that is
    what tells a human reading an escalation WHERE the run died — and it
    is a hand-typed string with no compiler checking it."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    _patch_load_data_to_return(monkeypatch, df_without_target)

    with pytest.raises(PipelineBlockedError) as exc_info:
        profile_dataset_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
        )

    err = exc_info.value
    assert err.stage == "data_agent"
    # The reason must carry clean_and_profile()'s explanation through
    # verbatim — the agents surface this string straight to the user.
    assert "diagnosis" in err.reason
    assert str(err).startswith("[data_agent] pipeline blocked:")


def test_blocking_issue_short_circuits_before_caching_anything(monkeypatch):
    """A blocked run must not leave a half-populated cache behind. The
    raise happens before cache.set(), so a later stage in the same run
    can never pick up a cleaned_df that was derived from data we already
    declared unusable."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    _patch_load_data_to_return(monkeypatch, df_without_target)

    cache = RunCache()
    run_id = "run-blocked"

    with pytest.raises(PipelineBlockedError):
        profile_dataset_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
            cache=cache,
            run_id=run_id,
        )

    assert cache.get(run_id, "cleaned_df") is None


def test_pipeline_blocked_error_is_not_a_value_error():
    """Regression test for the design decision written into the exception's
    own docstring: it exists as a DISTINCT type precisely so a broad
    `except ValueError` somewhere in an agent loop cannot swallow it. If
    someone ever 'simplifies' this to inherit from ValueError, the
    unsupported-data_source test above would start catching escalations
    too, and the two failure modes would become indistinguishable."""
    err = PipelineBlockedError(stage="data_agent", reason="target missing")

    assert isinstance(err, Exception)
    assert not isinstance(err, ValueError)
    assert err.stage == "data_agent"
    assert err.reason == "target missing"


# --------------------------------------------------------------------------
# Regression test for a fixed bug
# --------------------------------------------------------------------------

def test_column_schema_describes_every_column():
    """Regression test for a real indentation bug in data_stage.py: the
    column_schema.append(...) block sat one level too far OUT, outside the
    `for col in df.columns:` loop, so column_schema only ever recorded the
    LAST column no matter how wide the frame was.

    Nothing failed when this was broken — DataProfile stayed schema-valid,
    the pipeline ran, and every downstream stage carried on — the profile
    just quietly described 1 of 31 columns. That is why it needs an
    explicit count assertion rather than a smoke test."""
    profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
    )

    assert len(profile.column_schema) == profile.column_count
    assert profile.column_count > 1  # guards against a vacuous pass

    # Exactly one column is flagged as the target, and it is the right one.
    targets = [c for c in profile.column_schema if c.is_target]
    assert len(targets) == 1
    assert targets[0].name == "diagnosis"

    # Names are unique — a loop bug that appended the same column
    # repeatedly would satisfy the count check but not this one.
    names = [c.name for c in profile.column_schema]
    assert len(set(names)) == len(names)
