"""
Test/integration/test_feature_agent.py

INTEGRATION tests — FeatureAgent is the first class in this file's call
chain that talks to an LLM, so every test here injects the fake client
from conftest.py via make_fake_client. Nothing in this file costs money,
touches the network, or varies between runs.

What "integration" buys us that the unit tests do not: FeatureAgent is the
seam where a deterministic tool result and a non-deterministic model
verdict get stitched into one object AND written into shared run state.
The tool is already covered in test_feature_tools.py and the model's
opinion is not ours to test — what is left, and what is tested here, is
the wiring: does the right verdict reach the right field, does
pipeline_run get exactly the mutations it should, and does the blocked
path bail out BEFORE spending a call.

That last one is why call_count is asserted so often below. A blocked
dataset that still fires an LLM call is not a crash — it is a silent,
recurring waste that only shows up on a bill.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from agents.feature_agent import FeatureAgent, _Judgment
from pipeline.run_cache import RunCache
from schemas.eda_report import EDAReport
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec
import tools.feature_tools as feature_tools


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


def _run_agent(agent: FeatureAgent, pipeline_run=None, cache=None):
    return agent.run(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        pipeline_run=pipeline_run,
        cache=cache,
    )


# --------------------------------------------------------------------------
# proceed=True
# --------------------------------------------------------------------------

def test_proceed_true_writes_full_pipeline_run_state(make_fake_client):
    """The complete set of mutations FeatureAgent owns: the report, the
    status transition, a history line, and exactly ONE trace entry.

    'Exactly one' is the load-bearing part. record_call() is a single
    easily-dropped line, and the cost/latency accounting in
    total_cost_usd is only as trustworthy as every agent remembering to
    call it — a missing entry under-reports the run's cost forever, and
    a duplicated one over-reports it."""
    canned = _Judgment(proceed=True, summary="Features look sound.", concerns=[])
    fake_client = make_fake_client(canned)
    run_state = _make_run()

    result = _run_agent(FeatureAgent(client=fake_client), run_state, RunCache())

    assert result.proceed is True
    assert isinstance(result.report, EDAReport)

    assert run_state.status == "features_ready"
    assert run_state.eda_report is not None
    assert any(e.stage == "feature_agent" for e in run_state.history)
    assert len(run_state.trace) == 1
    assert run_state.trace[0].agent == "feature_agent"
    # The trace entry must carry real usage numbers through from the
    # response, not zeros — timed_llm_call() silently records 0 tokens if
    # `usage` is missing, which would make cost tracking pass vacuously.
    assert run_state.trace[0].total_tokens > 0
    assert run_state.trace[0].estimated_cost_usd > 0


def test_llm_receives_the_report_not_raw_data(make_fake_client):
    """The 'LLM never touches raw data' boundary, asserted at the last
    point where it could be broken. The user message must be the
    serialised EDAReport — so leakage_warnings reaches the model (the
    entire reason this agent's prompt exists), while the 569-row feature
    matrix does not."""
    canned = _Judgment(proceed=True, summary="ok", concerns=[])
    fake_client = make_fake_client(canned)

    result = _run_agent(FeatureAgent(client=fake_client))

    user_msg = fake_client.last_messages[-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == result.report.model_dump_json()
    assert "leakage_warnings" in user_msg["content"]


# --------------------------------------------------------------------------
# proceed=False — a halt, but a fully-recorded one
# --------------------------------------------------------------------------

def test_proceed_false_records_state_and_is_not_an_escalation(make_fake_client):
    """proceed=False is a DIFFERENT path from PipelineBlockedError, and
    conflating the two is easy. Here the tool succeeded and the model was
    genuinely consulted, so the report, the trace entry and the
    'features_ready' status must all be written exactly as on a pass.

    The status assertion is the subtle one: the agent does NOT set
    'escalated' here. run_pipeline.py is what turns this verdict into a
    halt (see test_run_pipeline.py) — the agent's job is to report, the
    orchestrator's job is to route. Keeping that split is what stops
    routing decisions from leaking into five different agent files."""
    canned = _Judgment(
        proceed=False,
        summary="Leakage warnings present.",
        concerns=["possible leakage"],
    )
    fake_client = make_fake_client(canned)
    run_state = _make_run()

    result = _run_agent(FeatureAgent(client=fake_client), run_state, RunCache())

    assert result.proceed is False
    assert result.concerns == ["possible leakage"]
    # Consulted, so all the normal bookkeeping still happened.
    assert result.report is not None
    assert run_state.eda_report is not None
    assert len(run_state.trace) == 1
    assert fake_client.call_count == 1
    # The agent itself does not escalate on a mere 'no'.
    assert run_state.status == "features_ready"


def test_concerns_survive_even_when_proceeding(make_fake_client):
    """'proceed' and 'no concerns' are not the same thing — the agents'
    prompts say so explicitly. A test that only ever pairs proceed=True
    with an empty concerns list would never catch a refactor that dropped
    the concerns field on the success path."""
    canned = _Judgment(
        proceed=True,
        summary="Fine, but watch this.",
        concerns=["dropped several low-signal features"],
    )
    fake_client = make_fake_client(canned)

    result = _run_agent(FeatureAgent(client=fake_client))

    assert result.proceed is True
    assert result.concerns == ["dropped several low-signal features"]


# --------------------------------------------------------------------------
# PipelineBlockedError — short-circuit BEFORE any LLM call
# --------------------------------------------------------------------------

@pytest.fixture
def blocked_data(monkeypatch):
    """Make the underlying tool raise PipelineBlockedError by handing
    load_data() a frame with no target column. Substituting the tool's
    INPUT, not its logic — the same seam documented in
    Test/unit/test_data_tools.py, and unavoidable because the only
    supported data source always has a usable target."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(feature_tools, "load_data", lambda spec: df_without_target)


def test_blocked_data_escalates_without_calling_the_llm(make_fake_client, blocked_data):
    """THE cost-control test. When the tool says the data is unusable,
    there is nothing for a model to judge — asking it anyway would burn a
    call to be told what the pipeline already knows. call_count == 0 is
    the assertion that keeps that true, and it is the one thing no amount
    of reading the agent's return value would reveal."""
    canned = _Judgment(proceed=True, summary="should never be reached", concerns=[])
    fake_client = make_fake_client(canned)
    run_state = _make_run()

    result = _run_agent(FeatureAgent(client=fake_client), run_state, RunCache())

    assert fake_client.call_count == 0

    assert result.proceed is False
    assert result.report is None
    assert result.summary.startswith("Cannot proceed:")
    assert result.concerns and "diagnosis" in result.concerns[0]

    assert run_state.status == "escalated"
    assert any("escalated" in e.event for e in run_state.history)
    # No LLM call means no trace entry — the run genuinely cost nothing.
    assert run_state.trace == []
    assert run_state.total_cost_usd == 0.0
    # And nothing was written that a later stage could mistake for a
    # successful feature run.
    assert run_state.eda_report is None


def test_blocked_data_standalone_does_not_raise(make_fake_client, blocked_data):
    """The blocked path with pipeline_run=None exercises a different
    branch — every state write is guarded by `if pipeline_run is not
    None`. A missing guard would turn an escalation into an
    AttributeError, converting a handled condition into a crash."""
    fake_client = make_fake_client(_Judgment(proceed=True, summary="x", concerns=[]))

    result = _run_agent(FeatureAgent(client=fake_client))

    assert result.proceed is False
    assert result.report is None
    assert fake_client.call_count == 0


# --------------------------------------------------------------------------
# Standalone use
# --------------------------------------------------------------------------

def test_works_standalone_without_pipeline_run_or_cache(make_fake_client):
    """pipeline_run and cache are optional by design, so the agent stays
    callable on its own. This is the contract every early manual script in
    this project relied on, and it must not quietly become mandatory."""
    canned = _Judgment(proceed=True, summary="Fine.", concerns=[])
    fake_client = make_fake_client(canned)

    result = _run_agent(FeatureAgent(client=fake_client))

    assert result.proceed is True
    assert isinstance(result.report, EDAReport)
    assert fake_client.call_count == 1


def test_uses_prior_profile_and_cache_from_pipeline_run(make_fake_client):
    """FeatureAgent reads pipeline_run.data_profile and passes it to the
    tool as the `profile` argument — which is half of what the tool needs
    for a cache hit. This test pins that hand-off: with a cleaned_df in
    the cache and a profile on the run, the agent must produce a correct
    report, and must leave X/y cached for the Tuning and Training stages
    that follow."""
    from tools.data_tools import profile_dataset_tool

    canned = _Judgment(proceed=True, summary="ok", concerns=[])
    fake_client = make_fake_client(canned)
    run_state = _make_run()
    cache = RunCache()

    # Stand in for a prior DataAgent call in the same run.
    run_state.data_profile = profile_dataset_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_state.run_id,
    )

    result = _run_agent(FeatureAgent(client=fake_client), run_state, cache)

    assert result.proceed is True
    assert cache.get(run_state.run_id, "X") is not None
    assert cache.get(run_state.run_id, "y") is not None
