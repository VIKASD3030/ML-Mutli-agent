"""
agents/feature_agent.py
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

from pipeline.observability import timed_llm_call
from pipeline.run_cache import RunCache
from schemas.eda_report import EDAReport
from schemas.pipeline_run import PipelineRun
from tools.data_tools import PipelineBlockedError
from tools.feature_tools import engineer_features_tool


class FeatureAgentResult(BaseModel):
    report: EDAReport | None
    proceed: bool
    summary: str
    concerns: list[str]


class _Judgment(BaseModel):
    proceed: bool
    summary: str
    concerns: list[str]


class FeatureAgent:
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
    ) -> FeatureAgentResult:
        run_id = pipeline_run.run_id if pipeline_run is not None else None
        prior_profile = pipeline_run.data_profile if pipeline_run is not None else None

        try:
            report = engineer_features_tool(
                data_source=data_source,
                target_column=target_column,
                task_type=task_type,
                cache=cache,
                run_id=run_id,
                profile=prior_profile,
            )
        except PipelineBlockedError as e:
            result = FeatureAgentResult(
                report=None,
                proceed=False,
                summary=f"Cannot proceed: {e.reason}",
                concerns=[e.reason],
            )
            if pipeline_run is not None:
                pipeline_run.status = "escalated"
                pipeline_run.log("feature_agent", f"escalated: {e.reason}")
            return result

        judgment, trace_entry = timed_llm_call(
            client= self.client,
            agent_name= "feature_agent",
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a feature-engineering reviewer for an ML "
                        "pipeline. You will be given a structured EDAReport. "
                        "Your MOST IMPORTANT job is examining leakage_warnings: "
                        "if it is non-empty, treat this as a serious concern "
                        "even if every other signal looks fine — a feature "
                        "that correlates near-perfectly with the target is "
                        "more likely a data leak than a genuinely great "
                        "predictor, and a leaked feature makes the eventual "
                        "model's test performance look good while actually "
                        "being useless in production. Do not recommend "
                        "proceeding confidently if leakage_warnings is "
                        "non-empty — recommend proceeding only with an "
                        "explicit warning to investigate before trusting "
                        "results. Secondary concerns (dropped low-signal "
                        "features, count of engineered features) matter less "
                        "and should not be given equal weight to a leakage "
                        "warning."
                        "\n\n"
                        # NEW — the actual fix. Makes leakage_warnings
                        # explicitly the ONLY authority on leakage risk, and
                        # directly forbids re-deriving a competing opinion
                        # from the raw correlation_summary values.
                        "IMPORTANT: leakage_warnings is the ONLY authoritative "
                        "signal for leakage risk — it was already computed "
                        "using a fixed, deliberate correlation threshold "
                        "specifically calibrated to separate 'strong, useful "
                        "signal' from 'suspiciously perfect.' If "
                        "leakage_warnings is empty, do NOT treat individual "
                        "high values you observe in correlation_summary as "
                        "independent evidence of risk — a feature correlating "
                        "at 0.7-0.9 with the target is normal, expected, and "
                        "desirable in a well-engineered feature set, not a "
                        "red flag. Do not lower your confidence or recommend "
                        "caution based on correlation_summary values alone "
                        "when leakage_warnings is empty."
                    ),
                },
                {"role": "user", "content": report.model_dump_json()},
            ],
            response_format=_Judgment,
            temperature=0,
        )

        result = FeatureAgentResult(
            report=report,
            proceed=judgment.proceed,
            summary=judgment.summary,
            concerns=judgment.concerns,
        )

        if pipeline_run is not None:
            pipeline_run.eda_report = report
            pipeline_run.status = "features_ready"
            pipeline_run.log("feature_agent", f"proceed={judgment.proceed}")
            pipeline_run.record_call(trace_entry)

        return result