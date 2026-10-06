"""
pipeline/run_pipeline.py

Two new parameters, both optional and additive:

- run_id: lets the CALLER choose the run's identity up front. Without
  this, run() always generated its own uuid internally — which would
  silently produce two different ids (one returned to the API client,
  one actually persisted) the instant this got wired into an async API.
- on_update: an optional callback invoked after every point run_state
  materially changes. This is what makes live progress polling
  possible — this file has NO knowledge of the database; the API's
  background task passes a callback that calls persist_run_state(),
  keeping the orchestrator decoupled from persistence, same separation
  agents already keep from knowing whether they're standalone or wired
  into a real run.
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.tuning_agent import TuningAgent
from agents.training_agent import TrainingAgent
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec


def run(
    spec: ProblemSpec,
    run_id: Optional[str] = None,
    on_update: Optional[Callable[[PipelineRun], None]] = None,
) -> PipelineRun:
    spec = spec.model_copy(deep=True)

    resolved_run_id = run_id or str(uuid.uuid4())
    run_state = PipelineRun(run_id=resolved_run_id, problem_spec=spec)
    cache = RunCache()
    run_state.log("orchestrator", "run started")

    def _sync() -> None:
        # on_update=None is checked in exactly ONE place, not at every
        # call site — every existing standalone/test call omits it and
        # behaves exactly as before.
        if on_update is not None:
            on_update(run_state)

    _sync()

    data_agent = DataAgent()
    feature_agent = FeatureAgent()
    tuning_agent = TuningAgent()
    training_agent = TrainingAgent()

    data_result = data_agent.run(
        data_source=spec.data_source, target_column=spec.target_column,
        task_type=spec.task_type, pipeline_run=run_state, cache=cache,
    )
    _sync()

    if not data_result.proceed:
        if run_state.status != "escalated":
            run_state.status = "escalated"
            run_state.log("orchestrator", f"escalated: DataAgent did not recommend proceeding — {data_result.summary}")
            _sync()
        return run_state

    resume_from = "feature"

    while True:
        if resume_from == "feature":
            feature_result = feature_agent.run(
                data_source=spec.data_source, target_column=spec.target_column,
                task_type=spec.task_type, pipeline_run=run_state, cache=cache,
            )
            _sync()

            if not feature_result.proceed:
                if run_state.status != "escalated":
                    run_state.status = "escalated"
                    run_state.log("orchestrator", f"escalated: FeatureAgent did not recommend proceeding — {feature_result.summary}")
                    _sync()
                return run_state
        else:
            run_state.log("orchestrator", "skipping feature_agent — resuming from tuning, features unchanged")
            _sync()

        tuning_result = tuning_agent.run(
            data_source=spec.data_source, target_column=spec.target_column,
            task_type=spec.task_type, max_tuning_trials=spec.constraints.max_tuning_trials,
            pipeline_run=run_state, cache=cache,
            success_metric=spec.success_metric,
        )
        _sync()

        if not tuning_result.proceed:
            run_state.log(
                "orchestrator",
                f"note: TuningAgent flagged concerns but pipeline continues to "
                f"Training for a real evaluation — {tuning_result.summary}",
            )
            _sync()

        training_result = training_agent.run(
            data_source=spec.data_source, target_column=spec.target_column,
            task_type=spec.task_type, success_metric=spec.success_metric,
            metric_threshold=spec.metric_threshold,
            max_tuning_trials=spec.constraints.max_tuning_trials,
            pipeline_run=run_state, cache=cache,
        )
        eval_report = training_result.report
        _sync()

        if eval_report.passed:
            run_state.status = "done"
            run_state.log("orchestrator", "run passed — pipeline complete")
            _sync()
            return run_state

        next_step = eval_report.failure_analysis.recommended_next_step
        run_state.loop_count += 1
        run_state.log(
            "orchestrator",
            f"fail -> recommended_next_step={next_step} "
            f"(TrainingAgent agrees: {training_result.agrees_with_rule_based_recommendation}), "
            f"loop_count={run_state.loop_count}",
        )
        _sync()

        if run_state.loop_count >= spec.constraints.max_loop_backs:
            run_state.status = "escalated"
            run_state.log("orchestrator", f"max_loop_backs ({spec.constraints.max_loop_backs}) reached — escalating to user")
            _sync()
            return run_state

        if next_step == "revisit_features":
            resume_from = "feature"
            continue
        elif next_step == "expand_hyperparam_search":
            spec.constraints.max_tuning_trials = int((spec.constraints.max_tuning_trials or 25) * 1.5)
            run_state.log("orchestrator", f"expanded max_tuning_trials to {spec.constraints.max_tuning_trials}")
            _sync()
            resume_from = "tuning"
            continue
        else:
            run_state.status = "escalated"
            run_state.log("orchestrator", f"escalating: {next_step} requires human input")
            _sync()
            return run_state


if __name__ == "__main__":
    demo_spec = ProblemSpec(
        task_type="classification", target_column="diagnosis", success_metric="f1",
        metric_threshold=0.90, data_source="builtin:breast_cancer",
    )
    demo_spec.constraints.max_tuning_trials = 15
    final_state = run(demo_spec)  # run_id/on_update omitted — unchanged behavior
    print(f"status: {final_state.status}, loops: {final_state.loop_count}")