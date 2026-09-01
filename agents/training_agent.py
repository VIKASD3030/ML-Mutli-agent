"""
agents/training_agent.py
"""

from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

from pipeline.observability import timed_llm_call
from pipeline.run_cache import RunCache
from schemas.evaluation_report import EvaluationReport
from schemas.pipeline_run import PipelineRun
from tools.data_tools import PipelineBlockedError
from tools.training_tools import train_and_evaluate_tool


class TrainingAgentResult(BaseModel):
    report: EvaluationReport | None
    summary: str
    agrees_with_rule_based_recommendation: Optional[bool]
    concerns: list[str]


class _Judgment(BaseModel):
    summary: str
    agrees_with_rule_based_recommendation: Optional[bool]
    concerns: list[str]


class TrainingAgent:
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self.client = client or OpenAI()
        self.model = model

    def run(
        self,
        data_source: str,
        target_column: str,
        task_type: Literal["classification", "regression"],
        success_metric: str,
        metric_threshold: float,
        max_tuning_trials: Optional[int] = 25,
        pipeline_run: PipelineRun | None = None,
        cache: RunCache | None = None,
    ) -> TrainingAgentResult:
        run_id = pipeline_run.run_id if pipeline_run is not None else None
        prior_tuning_result = pipeline_run.tuning_result if pipeline_run is not None else None

        try:
            report = train_and_evaluate_tool(
                data_source=data_source,
                target_column=target_column,
                task_type=task_type,
                success_metric=success_metric,
                metric_threshold=metric_threshold,
                max_tuning_trials=max_tuning_trials,
                cache=cache,
                run_id=run_id,
                tuning_result=prior_tuning_result,
            )
        except PipelineBlockedError as e:
            result = TrainingAgentResult(
                report=None,
                summary=f"Cannot proceed: {e.reason}",
                agrees_with_rule_based_recommendation=None,
                concerns=[e.reason],
            )
            if pipeline_run is not None:
                pipeline_run.status = "escalated"
                pipeline_run.log("training_agent", f"escalated: {e.reason}")
            return result

        judgment, trace_entry = timed_llm_call(
            client=self.client,
            agent_name= "training_agent",
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a model-evaluation reviewer for an ML "
                        "pipeline. You will be given a structured "
                        "EvaluationReport describing a trained model's test "
                        "performance. Explain the result in plain language a "
                        "non-technical user could follow. "
                        "\n\n"
                        "If pass_fail is 'fail', the report already contains "
                        "a rule-based failure_analysis.recommended_next_step "
                        "(one of: revisit_features, expand_hyperparam_search, "
                        "insufficient_data). Your job is NOT to produce a "
                        "competing recommendation — it is to sanity-check the "
                        "existing one against the actual metrics you're shown "
                        "(e.g. the confusion_matrix, if present) and report "
                        "whether you agree with it via "
                        "agrees_with_rule_based_recommendation. Set this field "
                        "to null/None if pass_fail is 'pass' — there is "
                        "nothing to agree or disagree with on a successful run. "
                        "\n\n"
                        "Do not invent a different recommended_next_step value "
                        "yourself — your role is advisory agreement/disagreement "
                        "on the existing one, not a replacement decision-maker."
                    ),
                },
                {"role": "user", "content": report.model_dump_json()},
            ],
            response_format=_Judgment,
            temperature=0,
        )

        result = TrainingAgentResult(
            report=report,
            summary=judgment.summary,
            agrees_with_rule_based_recommendation=judgment.agrees_with_rule_based_recommendation,
            concerns=judgment.concerns,
        )

        if pipeline_run is not None:
            pipeline_run.evaluation_report = report
            pipeline_run.status = "evaluated"
            pipeline_run.log("training_agent", f"pass_fail={report.pass_fail}")
            pipeline_run.record_call(trace_entry)

        return result