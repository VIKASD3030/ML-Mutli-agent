"""
agents/requirement_agent.py

Fifth agent, now upgraded: when a file_path is provided, this agent
follows the SAME "deterministic tool call first, LLM judges after"
pattern as every other agent — calling peek_dataset_tool() before the
LLM reasons about anything. When no file is provided, it falls back to
the original text-only behavior, unchanged.

KEY CHANGE: data_source is no longer something the LLM produces when a
file is attached — Python already knows the real path, so it's set
directly. This removes the exact failure class that caused the
'builtin breast cancer dataset' bug (a paraphrased data_source that
didn't match load_data()'s literal requirement) for the upload case
entirely, rather than just defending against it after the fact.
"""

from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

from pipeline.observability import timed_llm_call
from schemas.dataset_peek import DatasetPeek
from schemas.problem_spec import ClarificationNeeded, ProblemSpec
from tools.data_peek_tools import peek_dataset_tool

VALID_METRICS = {
    "classification": {"accuracy", "f1", "precision", "recall", "roc_auc"},
    "regression": {"rmse", "mae", "r2"},
}
KNOWN_BUILTIN_SOURCES = {"builtin:breast_cancer"}
FILE_EXTENSIONS = (".csv", ".parquet")


def is_valid_data_source(value: str) -> bool:
    return value in KNOWN_BUILTIN_SOURCES or value.endswith(FILE_EXTENSIONS)


class _Extraction(BaseModel):
    task_type: Optional[Literal["classification", "regression"]] = None
    target_column: Optional[str] = None
    success_metric: Optional[str] = None
    metric_threshold: Optional[float] = None
    data_source: Optional[str] = None  # only used/required on the no-file path
    unclear_or_missing: list[str] = []
    clarifying_question: Optional[str] = None


def _build_system_prompt(peek: DatasetPeek | None) -> str:
    """Two genuinely different prompts, not one deeply conditional
    string — easier to read and verify correctness of each path
    independently, matching this project's preference for explicit
    branches over clever conditionals."""

    base = (
        "You extract structured ML problem specifications from free-text "
        "requests. Determine task_type (classification or regression), "
        "target_column (the label/thing being predicted), success_metric, "
        "and metric_threshold. "
        f"Valid metrics per task_type: {VALID_METRICS}. "
        "metric_threshold may reasonably default to a sensible value "
        "(e.g. 0.8) without being flagged as unclear. If anything is in "
        "unclear_or_missing, also propose a single, specific "
        "clarifying_question."
    )

    if peek is not None:
        return (
            base
            + "\n\nThe user has attached a real dataset. Its ACTUAL schema "
            "(never invent or paraphrase a column name — use these exactly "
            f"as given):\n{peek.model_dump_json()}\n\n"
            "target_column MUST exactly match one of the column names "
            "above. If you cannot confidently identify which real column "
            "the user means, add target_column to unclear_or_missing "
            "rather than guessing. Use the identified target column's "
            "dtype and unique_count relative to row_count to inform "
            "task_type: low unique_count with a non-numeric or few-valued "
            "dtype suggests classification; a continuous numeric column "
            "with many unique values suggests regression. "
            "Do NOT determine data_source — it is already known and will "
            "be supplied separately; do not ask about or guess it."
        )
    else:
        return (
            base
            + "\n\nAlso determine data_source. If it isn't clearly stated, "
            "leave it null and add 'data_source' to unclear_or_missing — "
            "do NOT guess a plausible-sounding value, since a wrong guess "
            "would silently point the pipeline at the wrong data. Also do "
            "NOT guess target_column if it isn't clearly stated, for the "
            "same reason."
        )


class RequirementAgent:
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self.client = client or OpenAI()
        self.model = model

    def run(self, user_request: str, file_path: str | None = None) -> ProblemSpec | ClarificationNeeded:
        peek: DatasetPeek | None = None
        if file_path is not None:
            # Deterministic tool call FIRST — no LLM involved in this step.
            peek = peek_dataset_tool(file_path)

        extracted, _trace_entry = timed_llm_call(
            client=self.client,
            agent_name="requirement_agent",
            model=self.model,
            messages=[
                {"role": "system", "content": _build_system_prompt(peek)},
                {"role": "user", "content": user_request},
            ],
            response_format=_Extraction,
            temperature=0,
        )
        # NOTE: trace_entry isn't recorded to a PipelineRun here, since
        # this agent doesn't take one — it runs BEFORE a run exists. If
        # you want this call's cost tracked too, that requires deciding
        # where a pre-run trace should live; left as an open item.

        # Python-side backstop validating target_column against the REAL
        # peeked columns — same defence-in-depth principle as
        # is_valid_data_source(). The prompt already tells the model not to
        # invent a column name; this catches it when that is not followed.
        #
        # ORDERING IS THE POINT, and it is why this sits ABOVE the
        # unclear_or_missing branch rather than further down. Both branches
        # return a ClarificationNeeded, so whichever runs first WINS and the
        # other never executes. A model that invents a column name will
        # often also volunteer its own vaguer clarifying_question; if the
        # unclear_or_missing branch went first, that self-generated wording
        # would be what the user sees, and this precise, deterministic
        # message — which names the bad column AND lists the real ones —
        # would be unreachable. The deterministic backstop must outrank the
        # model's own commentary about the same mistake.
        #
        # Guarded on target_column being non-null: a null target is not a
        # WRONG column, it is a missing one, and the `missing` check below
        # already reports that case correctly.
        if peek is not None and extracted.target_column is not None:
            real_columns = {c.name for c in peek.columns}
            if extracted.target_column not in real_columns:
                return ClarificationNeeded(
                    missing_fields=["target_column"],
                    question_for_user=(
                        f"'{extracted.target_column}' isn't a column in this "
                        f"dataset. Real columns: {sorted(real_columns)}."
                    ),
                )

        if extracted.unclear_or_missing:
            return ClarificationNeeded(
                missing_fields=extracted.unclear_or_missing,
                question_for_user=extracted.clarifying_question
                or "Could you clarify the missing details?",
            )

        required = {
            "task_type": extracted.task_type,
            "target_column": extracted.target_column,
            "success_metric": extracted.success_metric,
            "metric_threshold": extracted.metric_threshold,
        }
        if peek is None:
            required["data_source"] = extracted.data_source
        missing = [name for name, value in required.items() if value is None]
        if missing:
            return ClarificationNeeded(
                missing_fields=missing,
                question_for_user=f"I need more detail on: {', '.join(missing)}.",
            )

        # data_source resolution: KNOWN deterministically when a file was
        # given; falls back to the LLM-extracted, validated value otherwise.
        resolved_data_source = file_path if peek is not None else extracted.data_source

        if peek is None and not is_valid_data_source(resolved_data_source):
            return ClarificationNeeded(
                missing_fields=["data_source"],
                question_for_user=(
                    f"'{resolved_data_source}' isn't a recognized data source. "
                    f"Please specify one of {sorted(KNOWN_BUILTIN_SOURCES)} or "
                    "a file path ending in .csv/.parquet."
                ),
            )

        # (The target_column-vs-real-columns backstop that used to live
        # here has moved ABOVE the unclear_or_missing branch — see the
        # comment there. Nothing is lost by removing it: by this point
        # target_column is guaranteed non-null, because the `missing` check
        # above returns for a null one, so the earlier check has already
        # validated exactly the same condition.)

        try:
            return ProblemSpec(
                task_type=extracted.task_type,
                target_column=extracted.target_column,
                success_metric=extracted.success_metric,
                metric_threshold=extracted.metric_threshold,
                data_source=resolved_data_source,
            )
        except ValidationError as e:
            return ClarificationNeeded(
                missing_fields=["success_metric"],
                question_for_user=(
                    f"The metric '{extracted.success_metric}' doesn't fit a "
                    f"{extracted.task_type} task — could you specify a valid "
                    f"metric instead? ({e})"
                ),
            )


if __name__ == "__main__":
    import pandas as pd
    import tempfile

    agent = RequirementAgent()

    print("=== Text-only request (no file, existing behavior) ===")
    result = agent.run(
        "Predict whether a breast tumor is malignant or benign using the "
        "builtin breast cancer dataset, target column diagnosis, aim for "
        "at least 0.9 f1 score."
    )
    print(type(result).__name__, "->", result)

    print("\n=== Real uploaded file (new behavior) ===")
    # A tiny synthetic dataset, so this demo runs with no real upload needed.
    df = pd.DataFrame({
        "customer_id": range(50),
        "monthly_spend": [i * 3.7 for i in range(50)],
        "churned": [i % 3 == 0 for i in range(50)],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    result = agent.run(
        "Predict whether a customer will churn, using this uploaded data.",
        file_path=temp_path,
    )
    print(type(result).__name__, "->", result)

    print("\n=== Real file, but user names a column that doesn't exist ===")
    result = agent.run(
        "Predict the customer_status column.",  # doesn't exist in the file
        file_path=temp_path,
    )
    print(type(result).__name__, "->", result)