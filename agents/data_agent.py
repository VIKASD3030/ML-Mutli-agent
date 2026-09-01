"""
agents/data_agent.py
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

from pipeline.run_cache import RunCache
from pipeline.observability import timed_llm_call
from schemas.data_profile import DataProfile
from schemas.pipeline_run import PipelineRun
from tools.data_tools import PipelineBlockedError, profile_dataset_tool


class DataAgentResult(BaseModel):
    profile: DataProfile | None
    proceed: bool
    summary: str
    concerns: list[str]


class _Judgment(BaseModel):
    proceed: bool
    summary: str
    concerns: list[str]


class DataAgent:
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self.client = client or OpenAI()
        self.model = model

    def run(
        self,
        data_source: str,
        target_column: str,
        task_type: Literal["classification", "regression"],
        pipeline_run: PipelineRun | None = None,
        cache: RunCache | None = None,
    ) -> DataAgentResult:
        run_id = pipeline_run.run_id if pipeline_run is not None else None

        try:
            profile = profile_dataset_tool(
                data_source=data_source,
                target_column=target_column,
                task_type=task_type,
                cache=cache,
                run_id=run_id,
            )
        except PipelineBlockedError as e:
            result = DataAgentResult(
                profile=None,
                proceed=False,
                summary=f"Cannot proceed: {e.reason}",
                concerns=[e.reason],
            )
            if pipeline_run is not None:
                pipeline_run.status = "escalated"  # FIX: typo
                pipeline_run.log("data_agent", f"escalated: {e.reason}")
            return result

        judgment, trace_entry = timed_llm_call(
            client = self.client,
            agent_name = "data_agent",
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data-quality reviewer for an ML pipeline. "
                        "You will be given a structured DataProfile. Decide "
                        "whether the data is good enough to proceed to feature "
                        "engineering, and explain your reasoning in plain "
                        "language a non-technical user could follow. Flag any "
                        "concerns (e.g. class imbalance, outliers, many "
                        "cleaning actions taken) even if you still recommend "
                        "proceeding — 'proceed' and 'no concerns' are not the "
                        "same thing."
                    ),
                },
                {"role": "user", "content": profile.model_dump_json()},
            ],
            response_format=_Judgment,
            temperature=0,
        )

        # FIX: name the result BEFORE returning, so the state update below
        # can actually run.
        result = DataAgentResult(
            profile=profile,
            proceed=judgment.proceed,
            summary=judgment.summary,
            concerns=judgment.concerns,
        )

        if pipeline_run is not None:
            pipeline_run.data_profile = profile
            pipeline_run.status = "data_ready"
            pipeline_run.log("data_agent", f"proceed={judgment.proceed}")
            pipeline_run.record_call(trace_entry)

        return result