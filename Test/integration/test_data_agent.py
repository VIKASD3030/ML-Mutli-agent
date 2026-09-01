"""
tests/integration/test_data_agent.py

Uses the fake client fixture — NO real API calls, NO real cost, fully
deterministic regardless of what a real model would actually say.
"""

import uuid

from agents.data_agent import DataAgent, _Judgment
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec


def _make_spec() -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
    )


def test_data_agent_proceed_true_writes_pipeline_run_state(make_fake_client):
    """When the fake LLM says proceed=True, verify DataAgent correctly
    updates pipeline_run — status, data_profile, log entry, AND the
    trace entry (the exact thing we just spent several turns debugging
    a missing record_call() line for)."""
    canned = _Judgment(proceed=True, summary="Looks fine.", concerns=[])
    fake_client = make_fake_client(canned)

    run_state = PipelineRun(run_id=str(uuid.uuid4()), problem_spec=_make_spec())
    cache = RunCache()

    agent = DataAgent(client=fake_client)
    result = agent.run(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        pipeline_run=run_state,
        cache=cache,
    )

    assert result.proceed is True
    assert run_state.status == "data_ready"
    assert run_state.data_profile is not None
    assert any(e.stage == "data_agent" for e in run_state.history)
    # Regression test for the exact bug we found and fixed — if
    # record_call() ever silently disappears again, THIS assertion
    # catches it automatically instead of requiring a manual print
    # and eyeball check like last time.
    assert len(run_state.trace) == 1
    assert run_state.trace[0].agent == "data_agent"


def test_data_agent_proceed_false_still_writes_state_not_just_returns(make_fake_client):
    """proceed=False from the LLM is a different code path than
    PipelineBlockedError — worth testing separately, since it's easy to
    accidentally only handle one of the two 'this run shouldn't
    continue' cases."""
    canned = _Judgment(proceed=False, summary="Too many concerns.", concerns=["bad data"])
    fake_client = make_fake_client(canned)

    run_state = PipelineRun(run_id=str(uuid.uuid4()), problem_spec=_make_spec())
    agent = DataAgent(client=fake_client)

    result = agent.run(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        pipeline_run=run_state,
        cache=RunCache(),
    )

    assert result.proceed is False
    # This still ran the full LLM-judgment path (unlike PipelineBlockedError,
    # which short-circuits BEFORE any LLM call) — so state should still be
    # written, unlike the blocked-data case.
    assert run_state.data_profile is not None
    assert len(run_state.trace) == 1


def test_data_agent_works_standalone_without_pipeline_run(make_fake_client):
    """Regression test for the optional-params design decision itself —
    calling an agent with NO pipeline_run/cache must still work exactly
    as it did before the caching redesign, since every early test in
    this project depends on that."""
    canned = _Judgment(proceed=True, summary="Fine.", concerns=[])
    fake_client = make_fake_client(canned)

    agent = DataAgent(client=fake_client)
    result = agent.run(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        # pipeline_run and cache both omitted — must not raise.
    )

    assert result.proceed is True
    assert fake_client.call_count == 1