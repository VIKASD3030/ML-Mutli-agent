"""
Test/unit/test_feature_tools.py

UNIT tests — engineer_features_tool() wraps pure pandas/numpy work with no
LLM anywhere in the call chain, so everything here runs for real.

The interesting logic in this tool is NOT the feature engineering (that
belongs to test_feature_stage.py); it is the cache protocol around it,
which has two halves that fail in opposite directions:

  READ  — a cache hit is only honoured when BOTH cleaned_df is cached AND
          a profile was passed in. Half a hit must be treated as a miss,
          because using a cached frame with a freshly-derived profile
          would silently pair data with a description of different data.

  WRITE — this tool is the ONLY writer of the "X"/"y" cache keys, which
          tune_model_tool() and train_and_evaluate_tool() both read. The
          tool's own comment calls this "THE FIX for bug #3": without the
          write, every downstream cache check silently always misses and
          the pipeline quietly re-does the most expensive work it has.
          A missing write breaks nothing visibly — it just makes the
          pipeline slow — which is precisely why it needs an assertion
          rather than a code review.

The cache-hit tests prove the skip NEGATIVELY, by pointing data_source at
a source load_data() does not support. If the recompute branch were taken,
load_data() would raise ValueError; completing successfully is therefore
proof that the branch was skipped, with no mocking involved.

ONE DELIBERATE SEAM: the blocking-issue test monkeypatches load_data for
the same reason documented at length in test_data_tools.py — the only
supported source cannot produce unusable-but-loadable data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.run_cache import RunCache
from schemas.data_profile import DataProfile
from schemas.eda_report import EDAReport
from tools.data_tools import PipelineBlockedError, profile_dataset_tool
from tools.feature_tools import engineer_features_tool
import tools.feature_tools as feature_tools

UNLOADABLE_SOURCE = "builtin:this_source_does_not_exist"


@pytest.fixture
def primed_cache() -> tuple[RunCache, str, DataProfile]:
    """A cache in exactly the state the Data Agent leaves it in: a real
    cleaned_df under a real run_id, plus the real DataProfile that
    describes it. Built by calling the actual upstream tool rather than
    hand-assembling one, so this fixture cannot drift away from what the
    pipeline really produces."""
    cache = RunCache()
    run_id = "run-feature-tools"
    profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )
    return cache, run_id, profile


# --------------------------------------------------------------------------
# Happy path / cache-miss
# --------------------------------------------------------------------------

def test_cache_miss_recomputes_and_returns_valid_eda_report():
    """No cache at all — the tool must do the full load/clean/engineer
    chain itself and still produce a correct EDAReport. This is the
    standalone-call path every early manual test in this project used,
    and it must keep working unchanged."""
    report = engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
    )

    assert isinstance(report, EDAReport)
    assert len(report.final_feature_names) > 0
    assert len(report.correlation_summary) > 0
    # The target must never survive into the feature set.
    assert "diagnosis" not in report.final_feature_names
    # Correlations are stored as absolute values, per EDAReport's docstring.
    assert all(0.0 <= v <= 1.0 for v in report.correlation_summary.values())


def test_report_does_not_leak_raw_data():
    """The tool's docstring draws a hard boundary: the feature matrix X is
    deliberately NOT returned, only the report describing what happened,
    because the LLM downstream must never touch raw numbers. Pinning the
    return TYPE is what keeps someone from 'helpfully' widening this to
    return (X, y, report) later and quietly pushing 569 rows of floats
    into a prompt."""
    result = engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
    )

    assert isinstance(result, EDAReport)
    assert not isinstance(result, tuple)


# --------------------------------------------------------------------------
# Cache READ — the hit path
# --------------------------------------------------------------------------

def test_cache_hit_skips_load_data_entirely(primed_cache):
    """The negative proof described in the module docstring: data_source
    points at a source load_data() would reject. Completing without an
    exception is only possible if the cached cleaned_df + passed-in
    profile were used and load_data() was never called."""
    cache, run_id, profile = primed_cache

    report = engineer_features_tool(
        data_source=UNLOADABLE_SOURCE,  # would raise ValueError if loaded
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
        profile=profile,
    )

    assert isinstance(report, EDAReport)
    assert len(report.final_feature_names) > 0


def test_cache_hit_produces_same_report_as_full_recompute(primed_cache):
    """Taking the shortcut must not change the answer. If the cached path
    and the recompute path ever diverge, the pipeline's behaviour would
    depend on whether an earlier agent happened to run — a genuinely
    nasty class of bug, since it only shows up in full runs and never in
    a standalone call."""
    cache, run_id, profile = primed_cache

    from_cache = engineer_features_tool(
        data_source=UNLOADABLE_SOURCE,
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
        profile=profile,
    )
    from_scratch = engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
    )

    assert from_cache.model_dump_json() == from_scratch.model_dump_json()


@pytest.mark.parametrize(
    "pass_profile, use_run_id",
    [
        (False, True),   # cleaned_df cached, but no profile => must be a MISS
        (True, False),   # profile given, but no run_id to look under => MISS
    ],
    ids=["profile_missing", "run_id_missing"],
)
def test_half_a_cache_hit_is_treated_as_a_miss(primed_cache, pass_profile, use_run_id):
    """'Only a genuine cache hit if we have BOTH pieces' — the tool's own
    comment. Here data_source is VALID, so a miss is survivable; the
    assertion is simply that the tool still succeeds rather than pairing
    a cached frame with a mismatched or absent profile."""
    cache, run_id, profile = primed_cache

    report = engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id if use_run_id else None,
        profile=profile if pass_profile else None,
    )

    assert isinstance(report, EDAReport)
    assert len(report.final_feature_names) > 0


def test_half_a_cache_hit_actually_reloads_the_data(primed_cache):
    """The sharper version of the test above: with no profile passed in,
    the tool MUST fall through to load_data(). Pointing at an unloadable
    source turns that fall-through into an observable ValueError — which
    is the positive proof that the miss branch really was taken, rather
    than the tool quietly half-using the cached frame."""
    cache, run_id, _profile = primed_cache

    with pytest.raises(ValueError):
        engineer_features_tool(
            data_source=UNLOADABLE_SOURCE,
            target_column="diagnosis",
            task_type="classification",
            cache=cache,
            run_id=run_id,
            profile=None,  # the missing half
        )


# --------------------------------------------------------------------------
# Cache WRITE — "THE FIX for bug #3"
# --------------------------------------------------------------------------

def test_x_and_y_are_cached_after_a_successful_call():
    """The regression test for the real bug: without these two writes,
    tune_model_tool() and train_and_evaluate_tool() re-run the entire
    load/clean/engineer chain on every call and nothing anywhere fails —
    the pipeline just silently costs several times more. X must be the
    engineered matrix (matching final_feature_names, which INCLUDES the
    interaction column), and y the target series."""
    cache = RunCache()
    run_id = "run-writes-xy"

    report = engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )

    X = cache.get(run_id, "X")
    y = cache.get(run_id, "y")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    # X must be the FINAL matrix the report describes — same columns, in
    # the same order — not the pre-drop intermediate.
    assert list(X.columns) == report.final_feature_names
    assert len(X) == len(y)
    assert y.name == "diagnosis"


def test_cleaned_df_is_repopulated_on_a_miss():
    """On a miss the tool re-derives cleaned_df, and must leave it in the
    cache for any later call in the same run — mirroring what the Data
    Agent's tool already does. Without this, a run that starts at the
    Feature stage would never populate the cache at all."""
    cache = RunCache()
    run_id = "run-miss-repopulates"

    engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )

    assert isinstance(cache.get(run_id, "cleaned_df"), pd.DataFrame)


def test_no_cache_writes_when_cache_or_run_id_is_omitted():
    """Symmetry with the read path: caching is all-or-nothing on both
    params. A run_id with no cache must not raise, and a cache with no
    run_id must stay completely empty rather than writing under some
    invented default key."""
    cache = RunCache()

    engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=None,
    )

    assert cache.get("run-anything", "X") is None
    assert cache.get("run-anything", "y") is None

    # And the mirror case: run_id but no cache must simply not raise.
    report = engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=None,
        run_id="run-orphaned",
    )
    assert isinstance(report, EDAReport)


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------

def test_blocking_issue_raises_with_feature_agent_stage(monkeypatch):
    """Same escalation contract as data_tools, but the stage name must be
    'feature_agent' — the two tools share one exception TYPE (deliberately,
    so the orchestrator catches one thing) and are distinguished only by
    this hand-typed string. A copy-paste of 'data_agent' into this file
    would misreport where a run died, and nothing but this test would
    catch it."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(feature_tools, "load_data", lambda spec: df_without_target)

    with pytest.raises(PipelineBlockedError) as exc_info:
        engineer_features_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
        )

    assert exc_info.value.stage == "feature_agent"
    assert "diagnosis" in exc_info.value.reason


def test_blocked_run_leaves_no_x_or_y_in_the_cache(monkeypatch):
    """A blocked run must not poison the cache for whatever runs next.
    The raise happens before engineer_features(), so X/y can never be
    written from data already declared unusable."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(feature_tools, "load_data", lambda spec: df_without_target)

    cache = RunCache()
    run_id = "run-blocked-feature"

    with pytest.raises(PipelineBlockedError):
        engineer_features_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
            cache=cache,
            run_id=run_id,
        )

    assert cache.get(run_id, "X") is None
    assert cache.get(run_id, "y") is None
