"""
Test/unit/test_tuning_tools.py

UNIT tests — no LLM in this call chain, so Optuna and RandomForest run for
real. Every test here uses a deliberately tiny trial budget (and, where
possible, a tiny synthetic dataset injected through the cache) because the
thing under test is the tool's PLUMBING, not the quality of the search.
Spending thirty seconds finding good hyperparameters would tell us nothing
that three trials does not.

What is NOT asserted here, on purpose: best_params and best_cv_score.
tuning_stage.py calls optuna.create_study() with no seeded sampler, so the
TPE sampler draws from process entropy and those values legitimately
differ run to run. Asserting on them would produce a test that fails at
random, which is worse than no test. Everything asserted below is either a
structural contract (schema validity, budget accounting) or a control-flow
fact (which branch ran), both of which are fully deterministic.

The cache-hit test proves the skip NEGATIVELY, the same technique used in
test_feature_tools.py: point data_source at something load_data() cannot
handle, and let a successful return prove the recompute branch never ran.
This matters more here than anywhere else in the pipeline, because the
branch being skipped is the single most expensive one in the project —
load, clean, AND engineer, every call.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.run_cache import RunCache
from schemas.tuning_result import TuningResult
from tools.data_tools import PipelineBlockedError
from tools.tuning_tools import tune_model_tool
import tools.tuning_tools as tuning_tools

UNLOADABLE_SOURCE = "builtin:this_source_does_not_exist"

# Three-fold CV is hardcoded in tuning_stage.py, so any synthetic target
# needs at least 3 examples of each class. 30 rows is comfortably above
# that and still near-instant to fit.
SYNTHETIC_ROWS = 30


@pytest.fixture
def synthetic_xy() -> tuple[pd.DataFrame, pd.Series]:
    """A trivially separable two-feature classification set. Deliberately
    NOT the breast-cancer data: the point of these tests is that the tool
    uses whatever X/y it was handed via the cache, and using an obviously
    different dataset makes it impossible for a passing test to be
    quietly running on the real data instead."""
    labels = [0, 1] * (SYNTHETIC_ROWS // 2)
    X = pd.DataFrame({
        "feat_a": [float(v) for v in labels],
        "feat_b": [float(v) * 2.0 + 0.5 for v in labels],
    })
    y = pd.Series(labels, name="diagnosis")
    return X, y


@pytest.fixture
def cache_with_xy(synthetic_xy) -> tuple[RunCache, str]:
    """The cache in exactly the state engineer_features_tool() leaves it —
    "X" and "y" populated under a run_id. Written with cache.set() rather
    than by running the real feature tool so the data stays small and the
    test stays fast; the KEY NAMES are the contract, and they are asserted
    against the real writer over in test_feature_tools.py."""
    X, y = synthetic_xy
    cache = RunCache()
    run_id = "run-tuning-tools"
    cache.set(run_id, "X", X)
    cache.set(run_id, "y", y)
    return cache, run_id


# --------------------------------------------------------------------------
# Happy path / cache-miss fallback
# --------------------------------------------------------------------------

def test_valid_call_returns_tuning_result_with_correct_budget_accounting():
    """The budget fields are the tool's only hard numeric contract: the
    caller asks for N trials, and both search_budget_used and
    search_budget_total must report N.

    Note these two being EQUAL is the normal, expected outcome —
    study.optimize() has no early stopping, so it always spends the whole
    budget. `converged` used to be derived from used < total and was
    therefore always False; it is now an independent field set from the
    plateau heuristic, which is why it is asserted separately below rather
    than inferred from these numbers."""
    result = tune_model_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        max_tuning_trials=3,
    )

    assert isinstance(result, TuningResult)
    assert result.search_budget_total == 3
    assert result.search_budget_used == 3
    # No success_metric passed, so the tool falls back to accuracy — and the
    # search now really optimises it (it used to be tuned on f1_macro regardless).
    assert result.metric_optimized == "accuracy"
    assert result.best_params  # non-empty
    assert result.convergence_notes  # non-empty
    # converged is a real stored field now, not a budget comparison, so it
    # must survive the trip through the tool. Its VALUE depends on where
    # the unseeded sampler happened to find its best trial, so only the
    # type is asserted here — the value's meaning is pinned in
    # test_tuning_stage.py against convergence_notes.
    assert isinstance(result.converged, bool)
    # Structural check on best_params: the keys must be the three the
    # search space actually defines, and each value must sit inside the
    # declared range. This pins the search SPACE without pinning the
    # (non-deterministic) result.
    assert set(result.best_params) == {"n_estimators", "max_depth", "min_samples_split"}
    assert 50 <= result.best_params["n_estimators"] <= 300
    assert 2 <= result.best_params["max_depth"] <= 20
    assert 2 <= result.best_params["min_samples_split"] <= 10


def test_cache_miss_with_no_cache_at_all_still_works():
    """The standalone path: no cache, no run_id. Must fall back to the
    full recompute and return a valid result, because this is how every
    manual test and any direct LLM tool call invokes it."""
    result = tune_model_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        max_tuning_trials=2,
    )

    assert isinstance(result, TuningResult)
    assert result.search_budget_used == 2


@pytest.mark.parametrize(
    "use_cache, use_run_id",
    [(True, False), (False, True)],
    ids=["cache_without_run_id", "run_id_without_cache"],
)
def test_partial_cache_arguments_fall_back_to_recompute(use_cache, use_run_id):
    """Caching needs BOTH params. Supplying one must degrade gracefully to
    a correct (if wasteful) recompute — never raise, and never silently
    read from a run_id-less bucket."""
    result = tune_model_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        max_tuning_trials=2,
        cache=RunCache() if use_cache else None,
        run_id="run-partial" if use_run_id else None,
    )

    assert isinstance(result, TuningResult)


def test_empty_cache_degrades_gracefully_to_recompute():
    """A cache MISS must degrade to a correct result, not raise. The
    tool's docstring is explicit that this matters even inside a real run:
    if the Feature Agent somehow did not populate X/y, tuning should still
    produce an answer rather than taking the whole pipeline down."""
    result = tune_model_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        max_tuning_trials=2,
        cache=RunCache(),          # present but empty
        run_id="run-nothing-cached",
    )

    assert isinstance(result, TuningResult)
    assert result.search_budget_used == 2


# --------------------------------------------------------------------------
# Cache-hit: the expensive branch must be skipped entirely
# --------------------------------------------------------------------------

def test_cache_hit_skips_load_clean_and_engineer(cache_with_xy):
    """The core regression test. data_source is a source load_data() would
    reject outright, so reaching a TuningResult at all is proof that
    load_data(), clean_and_profile() AND engineer_features() were all
    bypassed in favour of the cached X/y."""
    cache, run_id = cache_with_xy

    result = tune_model_tool(
        data_source=UNLOADABLE_SOURCE,  # would raise ValueError if loaded
        target_column="diagnosis",
        task_type="classification",
        max_tuning_trials=2,
        cache=cache,
        run_id=run_id,
    )

    assert isinstance(result, TuningResult)
    assert result.search_budget_used == 2
    # Tuned on the trivially separable synthetic set, so the score should
    # be near-perfect. Asserted as a loose floor, not an exact value —
    # enough to prove it really trained on the injected data, without
    # depending on which hyperparameters the unseeded sampler happened
    # to pick.
    assert result.best_cv_score > 0.9


def test_cache_hit_requires_both_x_and_y(synthetic_xy):
    """Half a hit is a miss here too. With only "X" cached, the tool must
    fall through to the recompute branch — which, against an unloadable
    source, surfaces as ValueError. Without this check the tool could
    reach tune_hyperparameters() with y=None and fail deep inside sklearn
    with a far less obvious error."""
    X, _y = synthetic_xy
    cache = RunCache()
    run_id = "run-half-cached"
    cache.set(run_id, "X", X)  # deliberately no "y"

    with pytest.raises(ValueError):
        tune_model_tool(
            data_source=UNLOADABLE_SOURCE,
            target_column="diagnosis",
            task_type="classification",
            max_tuning_trials=2,
            cache=cache,
            run_id=run_id,
        )


def test_cache_is_scoped_to_its_own_run_id(cache_with_xy):
    """Cached features from one run must never bleed into another. Asking
    for a different run_id has to be a miss, which against an unloadable
    source is observable as ValueError rather than a silent wrong answer
    computed on someone else's data."""
    cache, _run_id = cache_with_xy

    with pytest.raises(ValueError):
        tune_model_tool(
            data_source=UNLOADABLE_SOURCE,
            target_column="diagnosis",
            task_type="classification",
            max_tuning_trials=2,
            cache=cache,
            run_id="a-completely-different-run",
        )


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------

def test_blocking_issue_raises_with_tuning_agent_stage(monkeypatch):
    """Same escalation contract as the other tools, with the stage string
    that identifies THIS tool. See test_data_tools.py's module docstring
    for why load_data has to be substituted to reach this branch at all.

    Note the branch is only reachable on a cache MISS — on a hit the tool
    never builds a profile, so there is no blocking_issue to check. That
    is correct: cached X/y can only exist if an upstream stage already
    cleared the data."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(tuning_tools, "load_data", lambda spec: df_without_target)

    with pytest.raises(PipelineBlockedError) as exc_info:
        tune_model_tool(
            data_source="builtin:breast_cancer",
            target_column="diagnosis",
            task_type="classification",
            max_tuning_trials=2,
        )

    assert exc_info.value.stage == "tuning_agent"
    assert "diagnosis" in exc_info.value.reason


def test_unsupported_data_source_raises_value_error_not_blocked_error():
    """As in data_tools: 'this source does not exist' is a caller bug and
    must stay a ValueError, distinct from the escalation type the agents
    catch. Conflating them would turn a typo into a silent escalation."""
    with pytest.raises(ValueError) as exc_info:
        tune_model_tool(
            data_source=UNLOADABLE_SOURCE,
            target_column="diagnosis",
            task_type="classification",
            max_tuning_trials=2,
        )

    assert not isinstance(exc_info.value, PipelineBlockedError)
