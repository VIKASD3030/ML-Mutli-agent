"""
Test/integration/test_training_agent.py

INTEGRATION tests — all LLM calls go through the fake client from
conftest.py. No network, no cost, no variance.

TrainingAgent is shaped differently from the other three and the
differences are the point of this file:

  * It has NO `proceed` field. The pass/fail decision was already made by
    deterministic code in training_stage.py, so there is nothing for the
    model to vote on. Its result carries
    `agrees_with_rule_based_recommendation` instead — an advisory
    sanity-check on the rule-based routing, explicitly not a competing
    decision. The agent's prompt is emphatic that the model must not
    invent its own recommended_next_step.

  * It is the only agent that FEEDS the orchestrator's control flow.
    run_pipeline.py reads report.passed and
    report.failure_analysis.recommended_next_step to decide whether to
    finish, loop back, or escalate. So the tests here care a great deal
    about the report being carried through intact and unmodified.

  * It consumes pipeline_run.tuning_result to skip re-running the
    hyperparameter search. That hand-off gets its own test, because a
    break in it is invisible in the output and only shows up as cost.

One test below deliberately documents a GAP rather than a guarantee — see
test_agrees_field_is_passed_through_without_python_side_enforcement.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from agents.training_agent import TrainingAgent, _Judgment
from pipeline.run_cache import RunCache
from schemas.evaluation_report import EvaluationReport
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec
from schemas.tuning_result import TuningResult
from tools.feature_tools import engineer_features_tool
import tools.training_tools as training_tools


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


def _cheap_tuning_result() -> TuningResult:
    """A hand-built TuningResult so no Optuna search has to run. The
    params are legal for a RandomForest but deliberately tiny — these
    tests are about the agent's wiring, not about model quality."""
    return TuningResult(
        best_params={"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
        best_cv_score=0.95,
        metric_optimized="f1_macro",
        search_budget_used=10,
        search_budget_total=10,
        convergence_notes="hand-built for testing",
        # converged=False keeps the failure route at
        # expand_hyperparam_search, matching what the old derived property
        # (used < total) always produced. These tests are about the agent's
        # wiring — which fields reach pipeline_run, whether the LLM is
        # called — not about which loop-back verb the rules pick, so the
        # route is held constant deliberately.
        converged=False,
    )


@pytest.fixture(scope="module")
def real_features() -> tuple[pd.DataFrame, pd.Series]:
    """The genuine engineered features, computed once for the module.
    Real data matters here: pass_fail is a judgement about realistic
    performance, and a toy separable set would score 1.0 and make the
    'fail' branch unreachable."""
    cache = RunCache()
    run_id = "module-features"
    engineer_features_tool(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        cache=cache,
        run_id=run_id,
    )
    return cache.get(run_id, "X"), cache.get(run_id, "y")


@pytest.fixture
def primed_run(real_features) -> tuple[PipelineRun, RunCache]:
    """A PipelineRun and cache in the state the first three agents leave
    behind: X/y cached, and tuning_result already on the run. Together
    these let TrainingAgent skip everything except the final fit."""
    X, y = real_features
    run_state = _make_run()
    cache = RunCache()
    cache.set(run_state.run_id, "X", X)
    cache.set(run_state.run_id, "y", y)
    run_state.tuning_result = _cheap_tuning_result()
    return run_state, cache


def _run_agent(agent, pipeline_run=None, cache=None, threshold=0.90, trials=2):
    return agent.run(
        data_source="builtin:breast_cancer",
        target_column="diagnosis",
        task_type="classification",
        success_metric="f1",
        metric_threshold=threshold,
        max_tuning_trials=trials,
        pipeline_run=pipeline_run,
        cache=cache,
    )


# --------------------------------------------------------------------------
# The passing run
# --------------------------------------------------------------------------

def test_pass_writes_full_pipeline_run_state(make_fake_client, primed_run):
    """The mutations TrainingAgent owns on a successful evaluation.

    Note status becomes 'evaluated', NOT 'done' — 'done' is set by the
    orchestrator after it reads report.passed. Same separation as
    everywhere else in this project: the agent reports, the orchestrator
    routes. An agent that set 'done' itself would be making a routing
    decision it cannot see enough of the run to make."""
    canned = _Judgment(
        summary="Model comfortably beat the threshold.",
        agrees_with_rule_based_recommendation=None,
        concerns=[],
    )
    fake_client = make_fake_client(canned)
    run_state, cache = primed_run

    result = _run_agent(TrainingAgent(client=fake_client), run_state, cache, threshold=0.90)

    assert isinstance(result.report, EvaluationReport)
    assert result.report.pass_fail == "pass"

    assert run_state.status == "evaluated"
    assert run_state.evaluation_report is not None
    assert any(e.stage == "training_agent" for e in run_state.history)
    assert len(run_state.trace) == 1
    assert run_state.trace[0].agent == "training_agent"
    assert run_state.trace[0].total_tokens > 0


def test_agrees_is_none_on_a_passing_run(make_fake_client, primed_run):
    """The specific design rule: when pass_fail is 'pass' there is no
    rule-based recommendation in existence (failure_analysis is None), so
    there is nothing to agree or disagree with and the field must be None.

    What this test actually guarantees is the PLUMBING — that a None from
    the model survives the trip into TrainingAgentResult without being
    coerced to False. That is a real risk: the field is Optional[bool],
    and None and False mean very different things here (None = 'not
    applicable', False = 'the model actively disputes the routing')."""
    canned = _Judgment(
        summary="Passed.",
        agrees_with_rule_based_recommendation=None,
        concerns=[],
    )
    run_state, cache = primed_run

    result = _run_agent(
        TrainingAgent(client=make_fake_client(canned)), run_state, cache, threshold=0.90
    )

    assert result.report.pass_fail == "pass"
    assert result.report.failure_analysis is None
    assert result.agrees_with_rule_based_recommendation is None
    assert result.agrees_with_rule_based_recommendation is not False


def test_agrees_field_is_passed_through_without_python_side_enforcement(
    make_fake_client, primed_run
):
    """DOCUMENTS A GAP, deliberately — this test asserts what the code
    does, not what the design says it should do.

    The 'None when passing' rule is stated only in the system prompt.
    Nothing in training_agent.py checks it: the judgment's field is
    copied straight onto the result. So a model that returns True on a
    passing run produces a nonsensical 'agrees with the recommendation'
    on a run that has no recommendation, and no code anywhere objects.

    Written as a passing test rather than an xfail because this IS the
    current behaviour and pretending otherwise would be misleading. If
    the rule is ever enforced in Python — which is the safer design,
    since prompts are not contracts — this test will fail and should be
    rewritten to assert the coercion instead."""
    canned = _Judgment(
        summary="Passed.",
        agrees_with_rule_based_recommendation=True,  # nonsensical on a pass
        concerns=[],
    )
    run_state, cache = primed_run

    result = _run_agent(
        TrainingAgent(client=make_fake_client(canned)), run_state, cache, threshold=0.90
    )

    assert result.report.pass_fail == "pass"
    assert result.report.failure_analysis is None
    # No enforcement: the nonsensical value passes straight through.
    assert result.agrees_with_rule_based_recommendation is True


# --------------------------------------------------------------------------
# The failing run
# --------------------------------------------------------------------------

def test_fail_carries_failure_analysis_through_untouched(make_fake_client, primed_run):
    """On failure the report must arrive at the orchestrator with its
    rule-based failure_analysis intact — that field IS the loop-back
    routing input. The agent is advisory here and must not edit it."""
    canned = _Judgment(
        summary="Missed the threshold.",
        agrees_with_rule_based_recommendation=True,
        concerns=["threshold may be unrealistic"],
    )
    run_state, cache = primed_run

    result = _run_agent(
        TrainingAgent(client=make_fake_client(canned)), run_state, cache, threshold=0.999
    )

    assert result.report.pass_fail == "fail"
    assert result.report.passed is False
    assert result.report.failure_analysis is not None
    assert result.report.failure_analysis.recommended_next_step in {
        "revisit_features",
        "expand_hyperparam_search",
        "insufficient_data",
        "bad_spec",
    }
    # The report on the run is the same object the orchestrator will read.
    assert run_state.evaluation_report.failure_analysis == result.report.failure_analysis
    assert result.agrees_with_rule_based_recommendation is True


def test_model_can_disagree_without_changing_the_routing(make_fake_client, primed_run):
    """agrees=False is advisory only. The recommended_next_step must be
    exactly what the deterministic rules produced — the model's dissent
    is recorded for a human, never acted on. This is the guardrail that
    keeps routing deterministic while still surfacing model doubt."""
    canned = _Judgment(
        summary="I am not convinced by the suggested next step.",
        agrees_with_rule_based_recommendation=False,
        concerns=["disagrees with routing"],
    )
    run_state, cache = primed_run

    result = _run_agent(
        TrainingAgent(client=make_fake_client(canned)), run_state, cache, threshold=0.999
    )

    assert result.agrees_with_rule_based_recommendation is False
    # Dissent recorded, routing untouched.
    assert result.report.failure_analysis is not None
    assert run_state.status == "evaluated"


# --------------------------------------------------------------------------
# The tuning hand-off
# --------------------------------------------------------------------------

def test_prior_tuning_result_is_used_instead_of_a_fresh_search(make_fake_client, primed_run):
    """TrainingAgent reads pipeline_run.tuning_result and passes it to the
    tool, which then skips tune_hyperparameters() entirely. This is the
    agent-level half of the doubled-Optuna fix that
    Test/unit/test_training_tools.py pins at the tool level.

    Verified with a call counter on the collaborator, because 'was this
    function called' is not observable from the returned report — which
    is exactly why the bug survived as long as it did."""
    calls: list[int] = []
    real = training_tools.tune_hyperparameters

    def counting(X, y, spec):
        calls.append(1)
        return real(X, y, spec)

    run_state, cache = primed_run
    canned = _Judgment(summary="ok", agrees_with_rule_based_recommendation=None, concerns=[])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(training_tools, "tune_hyperparameters", counting)
        result = _run_agent(
            TrainingAgent(client=make_fake_client(canned)), run_state, cache, threshold=0.90
        )

    assert calls == []
    assert result.report is not None


def test_missing_tuning_result_falls_back_to_searching(make_fake_client, real_features):
    """The mirror case: no prior TuningResult on the run means the tool
    must search once. The fallback has to keep working, since a
    standalone TrainingAgent call has no upstream stage to inherit from."""
    X, y = real_features
    run_state = _make_run()
    cache = RunCache()
    cache.set(run_state.run_id, "X", X)
    cache.set(run_state.run_id, "y", y)
    assert run_state.tuning_result is None

    canned = _Judgment(summary="ok", agrees_with_rule_based_recommendation=None, concerns=[])
    result = _run_agent(
        TrainingAgent(client=make_fake_client(canned)),
        run_state,
        cache,
        threshold=0.90,
        trials=2,
    )

    assert result.report is not None
    assert run_state.evaluation_report is not None


# --------------------------------------------------------------------------
# PipelineBlockedError — short-circuit BEFORE any LLM call
# --------------------------------------------------------------------------

@pytest.fixture
def blocked_data(monkeypatch):
    """Force the tool's blocking branch via unusable INPUT — the seam
    documented in Test/unit/test_data_tools.py. Only reachable on a cache
    miss, so tests using this pass no cache."""
    df_without_target = pd.DataFrame({"not_the_target": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(training_tools, "load_data", lambda spec: df_without_target)


def test_blocked_data_escalates_without_calling_the_llm(make_fake_client, blocked_data):
    """No EvaluationReport means nothing for the model to explain, so the
    call must not happen. Note agrees_with_rule_based_recommendation is
    set to None on this path too — correctly, since no rule-based
    recommendation was ever produced."""
    canned = _Judgment(
        summary="should never be reached",
        agrees_with_rule_based_recommendation=True,
        concerns=[],
    )
    fake_client = make_fake_client(canned)
    run_state = _make_run()

    result = _run_agent(TrainingAgent(client=fake_client), run_state, cache=None)

    assert fake_client.call_count == 0

    assert result.report is None
    assert result.summary.startswith("Cannot proceed:")
    assert result.agrees_with_rule_based_recommendation is None

    assert run_state.status == "escalated"
    assert run_state.evaluation_report is None
    assert run_state.trace == []
    assert run_state.total_cost_usd == 0.0


def test_blocked_data_standalone_does_not_raise(make_fake_client, blocked_data):
    """Every state write on the blocked path is guarded by `if
    pipeline_run is not None`; a missing guard would turn a handled
    escalation into an AttributeError."""
    canned = _Judgment(
        summary="x", agrees_with_rule_based_recommendation=None, concerns=[]
    )
    fake_client = make_fake_client(canned)

    result = _run_agent(TrainingAgent(client=fake_client))

    assert result.report is None
    assert fake_client.call_count == 0


# --------------------------------------------------------------------------
# Standalone use
# --------------------------------------------------------------------------

def test_works_standalone_without_pipeline_run_or_cache(make_fake_client):
    """The bare call: no run, no cache, no prior tuning. Runs the whole
    chain including a (2-trial) search, and must return a complete
    result without raising."""
    canned = _Judgment(
        summary="Fine.", agrees_with_rule_based_recommendation=None, concerns=[]
    )
    fake_client = make_fake_client(canned)

    result = _run_agent(TrainingAgent(client=fake_client), trials=2)

    assert isinstance(result.report, EvaluationReport)
    assert result.report.pass_fail in {"pass", "fail"}
    assert fake_client.call_count == 1


def test_llm_receives_the_serialised_evaluation_report(make_fake_client, primed_run):
    """The prompt asks the model to sanity-check against the metrics it is
    shown, naming the confusion_matrix specifically — so the payload must
    be the serialised report, confusion matrix included."""
    canned = _Judgment(
        summary="ok", agrees_with_rule_based_recommendation=None, concerns=[]
    )
    fake_client = make_fake_client(canned)
    run_state, cache = primed_run

    result = _run_agent(TrainingAgent(client=fake_client), run_state, cache, threshold=0.90)

    user_msg = fake_client.last_messages[-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == result.report.model_dump_json()
    assert "confusion_matrix" in user_msg["content"]
