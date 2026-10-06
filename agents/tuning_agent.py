"""
agents/tuning_agent.py
"""

from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

from pipeline.observability import timed_llm_call
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.tuning_result import TuningResult
from tools.data_tools import PipelineBlockedError
from tools.tuning_tools import tune_model_tool


class TuningAgentResult(BaseModel):
    result: TuningResult | None
    proceed: bool
    summary: str
    concerns: list[str]


class _Judgment(BaseModel):
    proceed: bool
    summary: str
    concerns: list[str]


class TuningAgent:
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self.client = client or OpenAI()
        self.model = model

    def run(
        self,
        data_source: str,
        target_column: str,
        task_type: Literal["classification", "regression"],
        max_tuning_trials: Optional[int] = 25,
        pipeline_run: PipelineRun | None = None,
        cache: RunCache | None = None,
        success_metric: Optional[str] = None,
    ) -> TuningAgentResult:
        run_id = pipeline_run.run_id if pipeline_run is not None else None

        try:
            tuning_result = tune_model_tool(
                data_source=data_source,
                target_column=target_column,
                task_type=task_type,
                max_tuning_trials=max_tuning_trials,
                cache=cache,
                run_id=run_id,
                success_metric=success_metric,
            )
        except PipelineBlockedError as e:
            result = TuningAgentResult(
                result=None,
                proceed=False,
                summary=f"Cannot proceed: {e.reason}",
                concerns=[e.reason],
            )
            if pipeline_run is not None:
                pipeline_run.status = "escalated"
                pipeline_run.log("tuning_agent", f"escalated: {e.reason}")
            return result

        judgment, trace_entry = timed_llm_call(
            client= self.client,
            agent_name = "tuning_agent",
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hyperparameter-search reviewer for an ML "
                        "pipeline. You will be given a structured TuningResult. "
                        "Your primary job is weighing convergence_notes together "
                        "with search_budget_used vs search_budget_total: a high "
                        "best_cv_score from a search that had NOT converged "
                        "(convergence_notes indicates it was still improving "
                        "near the budget limit) is less trustworthy than the "
                        "same score from a search that plateaued, because the "
                        "search may simply have been cut off before finding a "
                        "better result. If the search used its full budget "
                        "without converging, recommend expanding "
                        "search_budget_total rather than treating best_cv_score "
                        "as final. Note: you cannot verify whether best_params "
                        "landed on the edge of the originally searched range, "
                        "since that range is not included in what you're shown "
                        "— do not speculate about specific parameter bounds you "
                        "cannot see."
                    ),
                },
                {"role": "user", "content": tuning_result.model_dump_json()},
            ],
            response_format=_Judgment,
            temperature=0,
        )

        result = TuningAgentResult(
            result=tuning_result,
            proceed=judgment.proceed,
            summary=judgment.summary,
            concerns=judgment.concerns,
        )

        if pipeline_run is not None:
            # This line is the actual fix for the double-tuning problem —
            # without it, TrainingAgent's fallback re-runs the whole search.
            pipeline_run.tuning_result = tuning_result
            pipeline_run.status = "tuned"
            pipeline_run.log("tuning_agent", f"proceed={judgment.proceed}")
            pipeline_run.record_call(trace_entry)

        return result