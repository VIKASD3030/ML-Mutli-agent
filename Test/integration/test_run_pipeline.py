"""
Test/integration/test_run_pipeline.py

INTEGRATION test for the orchestrator — all four agents are driven by fake
clients from conftest.py. No network, no cost, no variance.

A TESTABILITY GAP, AND WHAT THIS FILE DOES ABOUT IT
---------------------------------------------------
Every agent in this project takes `client: OpenAI | None = None` precisely
so tests can inject a fake. run() does not extend that seam: it takes only
a ProblemSpec and constructs DataAgent(), FeatureAgent(), TuningAgent()
and TrainingAgent() itself, each of which then falls back to a real
OpenAI() client. There is no supported way to hand it fakes.

So this file monkeypatches the four class names in the run_pipeline module
namespace, replacing each with a zero-argument factory that returns the
REAL agent class wired to a fake client. The agents, the tools and the
whole sklearn/Optuna chain all run for real; only the client is swapped.

That is a workaround, not a design. It is coupled to the fact that run()
happens to construct its agents by those exact module-level names, so a
harmless-looking refactor (importing the module instead of the classes,
say) would break these tests without breaking the code. The clean fix is
one optional parameter — `run(spec, agents=None)`, or `run(spec,
client=None)` threaded into the four constructors — which would let this
file delete the patching entirely. Worth doing; until then, the patching
is what makes any orchestrator coverage possible at all, and orchestrator
coverage is where the routing bugs actually live.

WHY THE ROUTING TESTS MATTER MORE THAN THE HAPPY PATH
-----------------------------------------------------
The resume_from mechanism is the whole reason run_pipeline.py was
rewritten: an expand_hyperparam_search loop-back used to re-invoke
FeatureAgent on unchanged data, burning an LLM call and re-exposing a
settled stage to fresh judgment variance. Nothing about that bug is
visible in the final status — a run with the bug and a run without it both
end 'escalated'. It is only detectable by counting who got called, which
is exactly what these tests do (via the trace, which records one entry per
agent per LLM call — the run's own bookkeeping, not a test-only spy).

SPEED
-----
max_tuning_trials starts at 2, and the run's RunCache means DataAgent's
cleaned frame flows into FeatureAgent, whose X/y flow into Tuning, whose
TuningResult flows into Training. The full four-agent run costs a couple
of seconds.
"""

from __future__ import annotations

import pytest

from agents.data_agent import DataAgent, _Judgment as DataJudgment
from agents.feature_agent import FeatureAgent, _Judgment as FeatureJudgment
from agents.training_agent import TrainingAgent, _Judgment as TrainingJudgment
from agents.tuning_agent import TuningAgent, _Judgment as TuningJudgment
from schemas.problem_spec import PipelineConstraints, ProblemSpec
import pipeline.run_pipeline as run_pipeline
import tools.tuning_tools as tuning_tools


def _spec(metric_threshold: float, max_loop_backs: int = 3) -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=metric_threshold,
        data_source="builtin:breast_cancer",
        constraints=PipelineConstraints(
            max_tuning_trials=2,
            max_loop_backs=max_loop_backs,
        ),
    )


def _agent_call_count(run_state, agent_name: str) -> int:
    """How many times an agent actually reached its LLM call, read off the
    run's own trace. Using the trace rather than a bespoke spy means these
    counts come from the same bookkeeping the cost report uses — if the
    trace is wrong, that is a bug worth failing on regardless."""
    return sum(1 for entry in run_state.trace if entry.agent == agent_name)


def _history_events(run_state, stage: str) -> list[str]:
    return [e.event for e in run_state.history if e.stage == stage]


@pytest.fixture
def fake_pipeline(monkeypatch, make_fake_client):
    """Installs fake-client-backed versions of all four agents into the
    run_pipeline namespace and hands back the four fake clients so a test
    can assert on call counts directly.

    Defaults are all-systems-go: every agent proceeds. A test that wants a
    different verdict passes its own judgment object."""

    def install(
        data=None,
        feature=None,
        tuning=None,
        training=None,
    ):
        data = data or DataJudgment(proceed=True, summary="data ok", concerns=[])
        feature = feature or FeatureJudgment(proceed=True, summary="features ok", concerns=[])
        tuning = tuning or TuningJudgment(proceed=True, summary="tuning ok", concerns=[])
        training = training or TrainingJudgment(
            summary="evaluated",
            agrees_with_rule_based_recommendation=None,
            concerns=[],
        )

        clients = {}
        for name, cls, canned in (
            ("DataAgent", DataAgent, data),
            ("FeatureAgent", FeatureAgent, feature),
            ("TuningAgent", TuningAgent, tuning),
            ("TrainingAgent", TrainingAgent, training),
        ):
            client = make_fake_client(canned)
            clients[name] = client

            def factory(_cls=cls, _client=client):
                return _cls(client=_client)

            monkeypatch.setattr(run_pipeline, name, factory)

        return clients

    return install


@pytest.fixture
def force_convergence(monkeypatch):
    """Pin TuningResult.converged for every search this run performs.

    WHY THIS IS NEEDED. `converged` is no longer derived from budget usage
    (which was always False); it now comes from the plateau heuristic —
    whether the best trial landed in the final 20% of trials. Optuna's
    sampler is unseeded, so on a 2-3 trial budget that is genuinely a coin
    flip, and it selects between three DIFFERENT failure routes:

        converged=False              -> expand_hyperparam_search (loops back)
        converged=True, wide gap     -> revisit_features         (loops back)
        converged=True, narrow gap   -> insufficient_data        (escalates)

    A loop-back test cannot assert anything about loop counts while the
    route is being chosen at random — the run either loops to the cap or
    terminates after one iteration. So each test below states which route
    it is exercising and pins it.

    This overrides ONE field on a genuinely-computed TuningResult; the real
    search, the real rules in training_stage.py and the real orchestrator
    routing all still run. It selects a branch, it does not simulate one."""

    def _force(converged: bool):
        real = tuning_tools.tune_hyperparameters

        def wrapper(X, y, spec):
            return real(X, y, spec).model_copy(update={"converged": converged})

        monkeypatch.setattr(tuning_tools, "tune_hyperparameters", wrapper)

    return _force


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_full_run_with_reachable_threshold_completes(fake_pipeline):
    """A run that passes on the first attempt: status 'done', zero
    loop-backs, and every stage's artefact present on the run state.

    The four artefact assertions are what prove all four agents genuinely
    ran and wrote their output — a status of 'done' alone would also be
    consistent with a stage being silently skipped."""
    clients = fake_pipeline()

    final = run_pipeline.run(_spec(metric_threshold=0.50))

    assert final.status == "done"
    assert final.loop_count == 0

    assert final.data_profile is not None
    assert final.eda_report is not None
    assert final.tuning_result is not None
    assert final.evaluation_report is not None
    assert final.evaluation_report.passed is True

    # Exactly one LLM call per agent on a single clean pass.
    for name in ("DataAgent", "FeatureAgent", "TuningAgent", "TrainingAgent"):
        assert clients[name].call_count == 1
    assert len(final.trace) == 4
    assert final.total_cost_usd > 0


def test_data_agent_refusal_halts_before_any_other_stage(fake_pipeline):
    """DataAgent's proceed=False is one of only two verdicts that gate the
    pipeline. The run must escalate immediately, and — the real assertion
    — no downstream agent may be consulted at all. A halt that still pays
    for three more LLM calls is not a halt."""
    clients = fake_pipeline(
        data=DataJudgment(proceed=False, summary="unusable", concerns=["bad data"])
    )

    final = run_pipeline.run(_spec(metric_threshold=0.50))

    assert final.status == "escalated"
    assert clients["DataAgent"].call_count == 1
    assert clients["FeatureAgent"].call_count == 0
    assert clients["TuningAgent"].call_count == 0
    assert clients["TrainingAgent"].call_count == 0
    assert final.eda_report is None
    assert final.evaluation_report is None


def test_feature_agent_refusal_halts_before_tuning(fake_pipeline):
    """The other gating verdict. FeatureAgent's own tests confirm it does
    not set 'escalated' itself — this is where that status actually gets
    applied, by the orchestrator, which is the layer that owns routing."""
    clients = fake_pipeline(
        feature=FeatureJudgment(proceed=False, summary="leakage", concerns=["leak"])
    )

    final = run_pipeline.run(_spec(metric_threshold=0.50))

    assert final.status == "escalated"
    assert clients["FeatureAgent"].call_count == 1
    assert clients["TuningAgent"].call_count == 0
    assert clients["TrainingAgent"].call_count == 0
    assert final.tuning_result is None


def test_tuning_agent_refusal_does_not_halt_the_run(fake_pipeline):
    """The asymmetry, asserted at the level where it actually lives.
    TuningAgent returning proceed=False must NOT stop the pipeline — the
    orchestrator logs a note and continues to Training so a real held-out
    metric gets the final word. Compare with the two tests above: same
    flag, same shape of result, opposite consequence."""
    clients = fake_pipeline(
        tuning=TuningJudgment(
            proceed=False, summary="under-searched", concerns=["not converged"]
        )
    )

    final = run_pipeline.run(_spec(metric_threshold=0.50))

    # Continued despite the 'no'.
    assert clients["TrainingAgent"].call_count == 1
    assert final.status == "done"
    assert final.evaluation_report is not None
    # And the concern was surfaced rather than swallowed.
    notes = _history_events(final, "orchestrator")
    assert any("TuningAgent flagged concerns" in n for n in notes)


# --------------------------------------------------------------------------
# resume_from — the regression this orchestrator rewrite was for
# --------------------------------------------------------------------------

def test_expand_loopback_does_not_reinvoke_feature_agent(fake_pipeline, force_convergence):
    """THE regression test. An unreachable threshold forces repeated
    failures; with the search reported as NOT converged, the rules route
    every one of them to expand_hyperparam_search, which means the
    features are not the suspect and must not be recomputed or re-judged.

    FeatureAgent must therefore be called exactly ONCE across the whole
    run, no matter how many loop-backs happen, while TuningAgent and
    TrainingAgent are called once per iteration. Before the resume_from
    fix, FeatureAgent's count tracked the loop count instead — the run
    still ended in the same state, which is why only a call count
    catches it."""
    clients = fake_pipeline()
    force_convergence(False)

    final = run_pipeline.run(_spec(metric_threshold=0.999, max_loop_backs=3))

    assert final.status == "escalated"
    assert final.loop_count >= 2
    # Confirm the route being tested is the one that actually fired.
    assert all(
        "expand_hyperparam_search" in n
        for n in _history_events(final, "orchestrator")
        if "recommended_next_step=" in n
    )

    # The heart of it: one feature pass, several tuning/training passes.
    assert clients["FeatureAgent"].call_count == 1
    assert _agent_call_count(final, "feature_agent") == 1
    assert clients["TuningAgent"].call_count == final.loop_count
    assert clients["TrainingAgent"].call_count == final.loop_count

    # And the orchestrator said out loud that it was skipping the stage.
    # One skip per iteration AFTER the first: the opening pass legitimately
    # runs FeatureAgent, and the run ends the moment loop_count hits the
    # limit, so there is no skip logged for the final increment.
    skips = [
        n for n in _history_events(final, "orchestrator")
        if "skipping feature_agent" in n
    ]
    assert len(skips) == final.loop_count - 1
    assert len(skips) >= 1


def test_expand_loopback_grows_the_tuning_budget(fake_pipeline, force_convergence):
    """The other half of the expand_hyperparam_search branch: each
    loop-back multiplies max_tuning_trials by 1.5. If the budget never
    grew, the loop would re-run an identical search and fail identically
    — burning the entire loop-back allowance to learn nothing."""
    clients = fake_pipeline()
    force_convergence(False)
    spec = _spec(metric_threshold=0.999, max_loop_backs=2)

    final = run_pipeline.run(spec)

    expansions = [
        n for n in _history_events(final, "orchestrator")
        if "expanded max_tuning_trials" in n
    ]
    # One expansion per loop-back that was actually followed by another
    # iteration — the last failure escalates instead of expanding.
    assert len(expansions) == final.loop_count - 1
    # Started at 2; each loop-back scales by 1.5 (int-truncated). The growth
    # is visible on the run's OWN spec copy and in the search that actually
    # ran, not on the caller's object — see the isolation test below.
    assert final.problem_spec.constraints.max_tuning_trials > 2
    assert final.tuning_result.search_budget_total > 2


def test_run_does_not_mutate_the_callers_spec(fake_pipeline, force_convergence):
    """Regression test for a real isolation bug: run() used to write the
    expanded trial budget straight back onto the ProblemSpec it was handed,
    so the caller's object came back modified and two runs from one spec
    were not independent — the second silently inherited the first's
    inflated budget and cost more for no stated reason.

    ProblemSpec's own docstring says no other agent may modify it; the
    orchestrator now holds itself to that too by deep-copying on entry.
    The budget still grows — just on the run's private copy."""
    fake_pipeline()
    force_convergence(False)  # expand route, so the budget actually grows
    spec = _spec(metric_threshold=0.999, max_loop_backs=2)
    original_budget = spec.constraints.max_tuning_trials

    final = run_pipeline.run(spec)

    # The caller's object is untouched...
    assert spec.constraints.max_tuning_trials == original_budget
    # ...while the run's own copy carries the expansion.
    assert final.problem_spec.constraints.max_tuning_trials > original_budget
    assert final.problem_spec is not spec


def test_two_runs_from_one_spec_are_independent(fake_pipeline, force_convergence):
    """The consequence that makes the isolation worth having. Running the
    same spec twice must produce two identical runs; before the fix the
    second inherited a larger trial budget and did strictly more work."""
    fake_pipeline()
    force_convergence(False)
    spec = _spec(metric_threshold=0.999, max_loop_backs=2)

    first = run_pipeline.run(spec)
    second = run_pipeline.run(spec)

    assert first.loop_count == second.loop_count
    assert first.status == second.status
    assert (
        first.problem_spec.constraints.max_tuning_trials
        == second.problem_spec.constraints.max_tuning_trials
    )
    assert len(first.trace) == len(second.trace)


# --------------------------------------------------------------------------
# Routes that only became reachable once `converged` was fixed
# --------------------------------------------------------------------------

def test_converged_search_escalates_via_insufficient_data(fake_pipeline, force_convergence):
    """NEW COVERAGE for a branch that was dead code until `converged`
    stopped being derived from budget usage.

    With the search reported as converged and the model landing close to
    (but under) the threshold, the rules conclude more DATA is needed —
    not more tuning, not different features. insufficient_data is a
    terminal verdict, not a loop-back: run_pipeline.py's routing has no
    branch for it, so it falls through to escalation on the FIRST failure.

    Before the fix this could never happen. Every failed run, whatever the
    actual cause, was labelled expand_hyperparam_search and burned the
    whole loop-back allowance re-searching hyperparameters."""
    clients = fake_pipeline()
    force_convergence(True)

    final = run_pipeline.run(_spec(metric_threshold=0.999, max_loop_backs=3))

    assert final.status == "escalated"
    # Terminal on the first failure — no loop-back was attempted, even
    # though the budget allowed three.
    assert final.loop_count == 1
    assert final.evaluation_report.failure_analysis.recommended_next_step == (
        "insufficient_data"
    )

    # One pass of every agent; nothing was re-run.
    for name in ("DataAgent", "FeatureAgent", "TuningAgent", "TrainingAgent"):
        assert clients[name].call_count == 1
    assert any(
        "requires human input" in n for n in _history_events(final, "orchestrator")
    )


def test_revisit_features_is_unreachable_on_this_dataset(fake_pipeline, force_convergence):
    """DOCUMENTS AN ARITHMETIC LIMIT, not a bug.

    revisit_features needs `converged and gap > 0.1 * threshold`. On
    breast_cancer a tuned forest scores f1 ~= 0.95, so reaching that gap
    would need threshold > 0.95 / 0.9 ~= 1.06 — impossible, since f1 is
    capped at 1.0. The branch is therefore unreachable through the
    orchestrator on this data no matter what `converged` says, and a
    converged failure always lands on insufficient_data instead.

    This is NOT the same as the bug that was just fixed. That bug made the
    branch unreachable for EVERY dataset because the flag was always
    False. This is a property of a dataset the model happens to fit well;
    the branch fires correctly when a model genuinely underperforms, which
    is pinned in Test/unit/test_training_stage.py against a deliberately
    signal-free dataset.

    Kept as an explicit test so that if the pipeline's default dataset or
    default model ever changes, this assumption gets re-examined rather
    than silently carried forward."""
    fake_pipeline()
    force_convergence(True)

    final = run_pipeline.run(_spec(metric_threshold=0.999, max_loop_backs=3))

    report = final.evaluation_report
    gap = report.metric_threshold - report.metric_value
    assert gap <= 0.1 * report.metric_threshold
    assert report.failure_analysis.recommended_next_step != "revisit_features"


# --------------------------------------------------------------------------
# Loop-back budget
# --------------------------------------------------------------------------

def test_max_loop_backs_is_respected_and_terminates(fake_pipeline, force_convergence):
    """A run that can never pass must escalate rather than loop forever,
    and loop_count must stop exactly at the limit — never overshoot it.
    This is the only thing standing between an unreachable threshold and
    an unbounded spend, so it is asserted as a hard equality, not a
    'roughly'."""
    fake_pipeline()
    force_convergence(False)  # every failure loops back rather than escalating

    final = run_pipeline.run(_spec(metric_threshold=0.999, max_loop_backs=2))

    assert final.status == "escalated"
    assert final.loop_count == 2
    assert any("max_loop_backs" in n for n in _history_events(final, "orchestrator"))


@pytest.mark.parametrize("max_loop_backs", [1, 2, 3])
def test_loop_count_never_exceeds_the_limit(fake_pipeline, force_convergence, max_loop_backs):
    """Same guarantee across several budgets, to catch an off-by-one that
    a single value would hide. The trace also has to stay consistent with
    the loop count — one Tuning and one Training call per iteration, plus
    the initial pass."""
    clients = fake_pipeline()
    force_convergence(False)

    final = run_pipeline.run(
        _spec(metric_threshold=0.999, max_loop_backs=max_loop_backs)
    )

    assert final.status == "escalated"
    assert final.loop_count == max_loop_backs
    assert clients["FeatureAgent"].call_count == 1
    # One Tuning+Training pass per iteration, and the run performs exactly
    # max_loop_backs iterations: the opening pass plus one per loop-back,
    # with the final failure escalating instead of looping again.
    assert clients["TuningAgent"].call_count == max_loop_backs
    assert clients["TrainingAgent"].call_count == max_loop_backs


def test_every_llm_call_is_recorded_exactly_once_in_the_trace(fake_pipeline, force_convergence):
    """Cross-check between the two independent records of what happened:
    the fake clients' own call counts, and the run's trace. They must
    agree exactly. A mismatch means some agent forgot record_call(), and
    total_cost_usd is under-reporting the run — a failure mode that
    otherwise shows up only on an invoice."""
    clients = fake_pipeline()
    force_convergence(False)

    final = run_pipeline.run(_spec(metric_threshold=0.999, max_loop_backs=2))

    expected_total = sum(c.call_count for c in clients.values())
    assert len(final.trace) == expected_total
    assert _agent_call_count(final, "data_agent") == clients["DataAgent"].call_count
    assert _agent_call_count(final, "feature_agent") == clients["FeatureAgent"].call_count
    assert _agent_call_count(final, "tuning_agent") == clients["TuningAgent"].call_count
    assert _agent_call_count(final, "training_agent") == clients["TrainingAgent"].call_count
    assert final.total_tokens == sum(e.total_tokens for e in final.trace)
