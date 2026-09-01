"""
pipeline/run_pipeline.py

Orchestrator, further refined: the while loop now tracks WHICH stage each
iteration should resume from (`resume_from`), instead of unconditionally
restarting from Feature Engineering every time. This directly fixes a
real observed bug: an `expand_hyperparam_search` loop-back was
accidentally re-invoking FeatureAgent on data that hadn't changed and
wasn't the reason for the failure — burning an extra LLM call and
exposing a stage to judgment variance it had no business being re-judged
on. Now, a loop-back only re-runs the stage(s) actually implicated by
the failure and everything downstream of it — never anything upstream.

DECISION (settled): Data/Feature's proceed=False still halts immediately,
uncapped by loop_count. With temperature=0 now in place across all
agents, a retry on identical cached input would almost certainly repeat
the same verdict — a retry cushion here would waste attempts confirming
the same "no," not rescue anything. The real fix for the observed
problem was resume-point precision, not a broader retry allowance.
"""

from __future__ import annotations

import uuid

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.tuning_agent import TuningAgent
from agents.training_agent import TrainingAgent
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec


def run(spec: ProblemSpec) -> PipelineRun:
    # Work on a COPY. The expand_hyperparam_search branch below rewrites
    # spec.constraints.max_tuning_trials, and mutating the caller's object
    # meant two runs from one spec were not independent — the second
    # silently inherited the first's inflated trial budget, which is the
    # kind of coupling that only shows up as an unexplained cost jump.
    # ProblemSpec is documented as "no other agent is allowed to modify
    # it"; the orchestrator should hold itself to the same rule.
    spec = spec.model_copy(deep=True)
    run_state = PipelineRun(run_id=str(uuid.uuid4()), problem_spec=spec)
    cache = RunCache()
    run_state.log("orchestrator", "run started")

    data_agent = DataAgent()
    feature_agent = FeatureAgent()
    tuning_agent = TuningAgent()
    training_agent = TrainingAgent()

    # --- Stage: Data (outside the loop — nothing ever routes back here) ---
    data_result = data_agent.run(
        data_source=spec.data_source,
        target_column=spec.target_column,
        task_type=spec.task_type,
        pipeline_run=run_state,
        cache=cache,
    )

    if not data_result.proceed:
        if run_state.status != "escalated":
            run_state.status = "escalated"
            run_state.log("orchestrator", f"escalated: DataAgent did not recommend proceeding — {data_result.summary}")
        return run_state

    # NEW: tracks which stage this loop iteration should start from.
    # "feature" on the first pass and after a revisit_features loop-back;
    # "tuning" after an expand_hyperparam_search loop-back, since nothing
    # about the features changed in that case — re-judging them would be
    # pointless at best (temperature=0 means it'd just repeat the same
    # verdict) and wasteful at worst (an extra LLM call for no reason).
    resume_from = "feature"

    while True:
        # --- Stage: Feature Engineering — only runs if this iteration
        # is actually supposed to start here. ---
        if resume_from == "feature":
            feature_result = feature_agent.run(
                data_source=spec.data_source,
                target_column=spec.target_column,
                task_type=spec.task_type,
                pipeline_run=run_state,
                cache=cache,
            )

            if not feature_result.proceed:
                if run_state.status != "escalated":
                    run_state.status = "escalated"
                    run_state.log("orchestrator", f"escalated: FeatureAgent did not recommend proceeding — {feature_result.summary}")
                return run_state
        else:
            run_state.log("orchestrator", "skipping feature_agent — resuming from tuning, features unchanged")

        # --- Stage: Hyperparameter Tuning — always runs when we reach
        # this point, whether we just computed fresh features or are
        # resuming with the same ones from before. ---
        tuning_result = tuning_agent.run(
            data_source=spec.data_source,
            target_column=spec.target_column,
            task_type=spec.task_type,
            max_tuning_trials=spec.constraints.max_tuning_trials,
            pipeline_run=run_state,
            cache=cache,
        )

        # UNCHANGED: TuningAgent's proceed=False still doesn't halt —
        # concern surfaced, Training still gets to check the real metric.
        if not tuning_result.proceed:
            run_state.log(
                "orchestrator",
                f"note: TuningAgent flagged concerns but pipeline continues to "
                f"Training for a real evaluation — {tuning_result.summary}",
            )

        # --- Stage: Train & Test ---
        training_result = training_agent.run(
            data_source=spec.data_source,
            target_column=spec.target_column,
            task_type=spec.task_type,
            success_metric=spec.success_metric,
            metric_threshold=spec.metric_threshold,
            max_tuning_trials=spec.constraints.max_tuning_trials,
            pipeline_run=run_state,
            cache=cache,
        )
        eval_report = training_result.report

        if eval_report.passed:
            run_state.status = "done"
            run_state.log("orchestrator", "run passed — pipeline complete")
            return run_state

        # --- Loop-back routing ---
        next_step = eval_report.failure_analysis.recommended_next_step
        run_state.loop_count += 1
        run_state.log(
            "orchestrator",
            f"fail -> recommended_next_step={next_step} "
            f"(TrainingAgent agrees: {training_result.agrees_with_rule_based_recommendation}), "
            f"loop_count={run_state.loop_count}",
        )

        if run_state.loop_count >= spec.constraints.max_loop_backs:
            run_state.status = "escalated"
            run_state.log(
                "orchestrator",
                f"max_loop_backs ({spec.constraints.max_loop_backs}) reached — escalating to user",
            )
            return run_state

        if next_step == "revisit_features":
            resume_from = "feature"   # NEW: explicit — re-run Feature next iteration
            continue
        elif next_step == "expand_hyperparam_search":
            spec.constraints.max_tuning_trials = int(
                (spec.constraints.max_tuning_trials or 25) * 1.5
            )
            run_state.log("orchestrator", f"expanded max_tuning_trials to {spec.constraints.max_tuning_trials}")
            resume_from = "tuning"    # NEW: skip Feature next iteration
            continue
        else:
            run_state.status = "escalated"
            run_state.log("orchestrator", f"escalating: {next_step} requires human input")
            return run_state


if __name__ == "__main__":
    demo_spec = ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
    )
    demo_spec.constraints.max_tuning_trials = 15

    final_state = run(demo_spec)

    print("\n=== PIPELINE RUN SUMMARY ===")
    print(f"run_id:  {final_state.run_id}")
    print(f"status:  {final_state.status}")
    print(f"loops:   {final_state.loop_count}")
    if final_state.evaluation_report:
        print(
            f"metric:  {final_state.evaluation_report.metric_optimized} = "
            f"{final_state.evaluation_report.metric_value} "
            f"(threshold {final_state.evaluation_report.metric_threshold})"
        )
        print(f"result:  {final_state.evaluation_report.pass_fail}")

    print(f"\ntotal_tokens:    {final_state.total_tokens}")
    print(f"total_cost_usd:  ${final_state.total_cost_usd}")
    print("\n--- per-call trace ---")
    for entry in final_state.trace:
        print(
            f"[{entry.agent}] {entry.model} | "
            f"{entry.total_tokens} tokens | "
            f"{entry.latency_ms:.0f}ms | "
            f"${entry.estimated_cost_usd}"
        )

    print("\n--- history ---")
    for entry in final_state.history:
        print(f"[{entry.stage}] {entry.event}")