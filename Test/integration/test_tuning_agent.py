"""
Test/integration/test_tuning_agent.py

INTEGRATION tests — every LLM call goes through the fake client from
conftest.py. No network, no cost, no variance.

The behaviour this file exists to pin is an ASYMMETRY that is invisible
from inside TuningAgent itself. All three of Data, Feature and Tuning
return an identically-shaped result with a `proceed` flag, but the
orchestrator treats Tuning's flag completely differently: Data's and
Feature's proceed=False halt the run, while Tuning's does not. Per
run_pipeline.py, a Tuning 'no' is logged as a note and the pipeline
carries on to Training deliberately — the reasoning being that a
hyperparameter search that looks unconvincing is a CONCERN, not a
verdict, and the real held-out metric is a better judge of whether the
run is doomed than a review of the search that produced it.

Because that decision lives in the orchestrator, nothing inside
TuningAgent enforces it, and the failure mode if someone 'fixes' the
inconsistency is nasty: the agent would start nulling out its result or
setting status='escalated' on proceed=False, and runs that should have
reached a real evaluation would die one stage early. The asymmetry test
below is what makes that regression loud.

Speed note: these tests pre-seed the cache with a tiny synthetic X/y so
the real Optuna search runs on 30 rows with 2 trials instead of the full
dataset. The search is genuinely executed — only its input is shrunk.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from agents.tuning_agent import TuningAgent, _Judgment
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec
from schemas.tuning_result import TuningResult
import tools.tuning_tools as tuning_tools

# tuning_stage.py hardcodes cv=3, so each class needs at least 3 rows.
SYNTHETIC_ROWS = 30


def _make_spec() -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
    )


def _make_run() -> PipelineRun:
    return PipelineRun(run_id=str(uuid.uuid4()), problem_spec=_make_spec())


@pytest.fixture
def run_and_cache() -> tuple[PipelineRun, RunCache]:
    """A PipelineRun plus a cache already holding X/y, exactly as
    FeatureAgent would have left it earlier in the same run. Keeping the
    injected data trivially separable and tiny is purely a speed choice —
    the agent, the tool and the Optuna search all run for real."""
    run_state = _make_run()
    cache = RunCache()
    labels = [0, 1] * (SYNTHETIC_ROWS // 2)
    X = pd.DataFrame({
        "feat_a": [float(v) for v in labels],
        "feat_b": [float(v) * 2.0 + 0.5 for v in labels],
    })
    y = pd.Series(labels, name="diagnosis")
    cache.set(run_state.run_id, "X", X)
    cache.set(run_state.run_id, "y", y)
    return run_state, cache


def _run_agent(agent: TuningAgent, pipeline_run=None, cache=None, trials=2):
    return agent.run(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        max_tuning_trials=trials,
        pipeline_run=pipeline_run,
        cache=cache,
    )


# --------------------------------------------------------------------------
# proceed=True
# --------------------------------------------------------------------------

def test_proceed_true_writes_full_pipeline_run_state(make_fake_client, run_and_cache):
    """The mutations TuningAgent owns. pipeline_run.tuning_result is the
    important one — the agent's own source comment calls that assignment
    'the actual fix for the double-tuning problem', because TrainingAgent
    reads exactly this field and re-runs the entire search when it finds
    None. Dropping the line costs nothing visible except a doubled bill."""
    canned = _Judgment(proceed=True, summary="Search converged.", concerns=[])
    fake_client = make_fake_client(canned)
    run_state, cache = run_and_cache

    result = _run_agent(TuningAgent(client=fake_client), run_state, cache)

    assert result.proceed is True
    assert isinstance(result.result, TuningResult)

    assert run_state.status == "tuned"
    assert run_state.tuning_result is not None
    assert run_state.tuning_result.search_budget_total == 2
    assert any(e.stage == "tuning_agent" for e in run_state.history)
    assert len(run_state.trace) == 1
    assert run_state.trace[0].agent == "tuning_agent"
    assert run_state.trace[0].total_tokens > 0


def test_llm_receives_the_serialised_tuning_result(make_fake_client, run_and_cache):
    """The agent's prompt asks the model to weigh convergence_notes
    against the budget fields, so those fields must actually be in the
    payload. Asserting on the serialisation (not on wording) keeps this
    test structural."""
    canned = _Judgment(proceed=True, summary="ok", concerns=[])
    fake_client = make_fake_client(canned)
    run_state, cache = run_and_cache

    result = _run_agent(TuningAgent(client=fake_client), run_state, cache)

    user_msg = fake_client.last_messages[-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == result.result.model_dump_json()
    assert "convergence_notes" in user_msg["content"]
    assert "search_budget_used" in user_msg["content"]


# --------------------------------------------------------------------------
# THE ASYMMETRY: proceed=False must not gate anything
# --------------------------------------------------------------------------

def test_proceed_false_still_leaves_a_usable_result_downstream(make_fake_client, run_and_cache):
    """The core test of this file. A Tuning 'no' is advisory: the run
    continues to Training for a real evaluation, so everything Training
    needs must still be in place.

    Concretely, on proceed=False the agent must STILL populate
    pipeline_run.tuning_result (or TrainingAgent silently re-runs the
    search), STILL set status='tuned' (NOT 'escalated'), and STILL return
    the TuningResult on the result object. Only the advisory fields —
    proceed and concerns — should differ from a proceed=True run."""
    canned = _Judgment(
        proceed=False,
        summary="Search had not converged at the budget limit.",
        concerns=["may be under-searched"],
    )
    fake_client = make_fake_client(canned)
    run_state, cache = run_and_cache

    result = _run_agent(TuningAgent(client=fake_client), run_state, cache)

    assert result.proceed is False
    assert result.concerns == ["may be under-searched"]

    # ...and yet every downstream-facing thing is fully populated.
    assert isinstance(result.result, TuningResult)
    assert run_state.tuning_result is not None
    assert run_state.tuning_result.best_params
    assert run_state.status == "tuned"
    assert run_state.status != "escalated"
    assert len(run_state.trace) == 1


def test_proceed_false_and_true_differ_only_in_the_advisory_fields(
    make_fake_client, run_and_cache
):
    """Sharper statement of the same rule, by comparison rather than by
    a list of assertions: run the identical setup twice, differing only
    in the model's verdict, and confirm the pipeline-visible state is
    indistinguishable. If a future change ever makes Tuning's proceed
    gate anything, these two runs stop matching."""
    run_yes, cache_yes = run_and_cache
    agent_yes = TuningAgent(
        client=make_fake_client(_Judgment(proceed=True, summary="ok", concerns=[]))
    )
    _run_agent(agent_yes, run_yes, cache_yes)

    # A second, independent run seeded with the same cached X/y.
    run_no = _make_run()
    cache_no = RunCache()
    cache_no.set(run_no.run_id, "X", cache_yes.get(run_yes.run_id, "X"))
    cache_no.set(run_no.run_id, "y", cache_yes.get(run_yes.run_id, "y"))
    agent_no = TuningAgent(
        client=make_fake_client(_Judgment(proceed=False, summary="no", concerns=["c"]))
    )
    _run_agent(agent_no, run_no, cache_no)

    assert run_yes.status == run_no.status == "tuned"
    assert (run_yes.tuning_result is None) == (run_no.tuning_result is None) is False
    assert len(run_yes.trace) == len(run_no.trace) == 1


# --------------------------------------------------------------------------
# PipelineBlockedError — short-circuit BEFORE any LLM call
# --------------------------------------------------------------------------

@pytest.fixture
def blocked_data(monkeypatch):
    """Force the tool's blocking branch by giving load_data() a frame with
    no target column — substituting INPUT, not logic (see
    Test/unit/test_data_tools.py). Note this only bites on a cache MISS,
    which is why the tests using it pass no cache: cached X/y can only
    exist if an earlier stage already cleared the data."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(tuning_tools, "load_data", lambda spec: df_without_target)


def test_blocked_data_escalates_without_calling_the_llm(make_fake_client, blocked_data):
    """Unlike a proceed=False verdict, blocked data IS a genuine
    escalation — the agent sets status itself here. And it must do so
    without consulting the model: there is no TuningResult to review, so
    a call would be pure waste."""
    canned = _Judgment(proceed=True, summary="should never be reached", concerns=[])
    fake_client = make_fake_client(canned)
    run_state = _make_run()

    result = _run_agent(TuningAgent(client=fake_client), run_state, cache=None)

    assert fake_client.call_count == 0

    assert result.proceed is False
    assert result.result is None
    assert result.summary.startswith("Cannot proceed:")

    assert run_state.status == "escalated"
    assert run_state.tuning_result is None
    assert run_state.trace == []
    assert run_state.total_cost_usd == 0.0


def test_blocked_data_standalone_does_not_raise(make_fake_client, blocked_data):
    """Every state write in the blocked branch is guarded by `if
    pipeline_run is not None`; a missing guard turns a handled escalation
    into an AttributeError."""
    fake_client = make_fake_client(_Judgment(proceed=True, summary="x", concerns=[]))

    result = _run_agent(TuningAgent(client=fake_client))

    assert result.proceed is False
    assert result.result is None
    assert fake_client.call_count == 0


# --------------------------------------------------------------------------
# Standalone use
# --------------------------------------------------------------------------

def test_works_standalone_without_pipeline_run_or_cache(make_fake_client):
    """No pipeline_run, no cache — the full recompute path, as a bare LLM
    tool call would hit it. Two trials keeps it quick; the point is that
    it completes and returns a real result rather than that it searches
    well."""
    canned = _Judgment(proceed=True, summary="Fine.", concerns=[])
    fake_client = make_fake_client(canned)

    result = _run_agent(TuningAgent(client=fake_client), trials=2)

    assert result.proceed is True
    assert isinstance(result.result, TuningResult)
    assert result.result.search_budget_used == 2
    assert fake_client.call_count == 1


def test_max_tuning_trials_reaches_the_search(make_fake_client, run_and_cache):
    """The trial budget is the one number the orchestrator actively
    changes between loop iterations (it multiplies it by 1.5 on an
    expand_hyperparam_search loop-back). If the agent dropped the
    argument on the way to the tool, that whole loop-back strategy would
    become a no-op that still costs a full extra pipeline pass."""
    canned = _Judgment(proceed=True, summary="ok", concerns=[])
    run_state, cache = run_and_cache

    result = _run_agent(TuningAgent(client=make_fake_client(canned)), run_state, cache, trials=3)

    assert result.result.search_budget_total == 3
    assert result.result.search_budget_used == 3
