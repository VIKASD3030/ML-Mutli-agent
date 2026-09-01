"""
agents/requirement_agent.py

Fifth and final agent — the one that doesn't wrap an existing Phase-1
stage. Its job is turning free-text user requests into a ProblemSpec,
the object every other agent in this project depends on.

KEY DESIGN POINT: the LLM never constructs ProblemSpec directly. It fills
out a looser, all-optional _Extraction schema; Python decides whether
those proposed values are complete and pass ProblemSpec's own validators
(including the metric/task_type match check) before ever producing a
real ProblemSpec. This matters because OpenAI's structured outputs only
enforce JSON SHAPE (types, enum membership) — not custom cross-field
validators like ProblemSpec's _metric_matches_task, so the model could
legally propose an invalid combination that only Pydantic would catch.
"""
from __future__ import annotations

from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

from schemas.problem_spec import ClarificationNeeded, ProblemSpec

VALID_METRICS = {
    "classification": {"accuracy", "f1", "precision", "recall", "roc_auc"},
    "regression": {"rmse", "mae", "r2"},
}

# Exact data_source tokens that load_data() actually supports.
# Keeping this as a module-level constant so (a) the system prompt can
# reference it and (b) the post-extraction validation can check against
# it — single source of truth, same principle as VALID_METRICS above.
SUPPORTED_DATA_SOURCES = [
    "builtin:breast_cancer",
]

class _Extraction(BaseModel):
    """What the LLM actually produces — deliberately loose (everything
    Optional) because the input is genuinely unstructured text. This is
    NOT ProblemSpec; it's raw material Python will validate afterward."""

    task_type: Optional[Literal["classification", "regression"]] = None
    target_column: Optional[str] = None
    success_metric: Optional[str] = None
    metric_threshold: Optional[str] = None
    data_source: Optional[str] = None
    unclear_or_missing: list[str] = []
    clarifying_question: Optional[str] = None

class RequirementAgent:
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self.client = client or OpenAI()
        self.model = model

    def run(self, user_request: str) -> ProblemSpec | ClarificationNeeded:
        completion = self.client.beta.chat.completions.parse(
            model = self.model,
            messages= [
                {
                    "role":"system",
                    "content": (
                        "You extract structured ML problem specifications from "
                        "free-text requests. Determine task_type "
                        "(classification or regression), target_column (the "
                        "label/thing being predicted), success_metric, "
                        "metric_threshold, and data_source. "
                        f"Valid metrics per task_type: {VALID_METRICS}. "
                        "\n\n"
                        "CRITICAL — data_source must be an EXACT token from "
                        "this supported list, or a real file path ending in "
                        ".csv or .parquet: "
                        f"{SUPPORTED_DATA_SOURCES}. "
                        "Do NOT paraphrase or describe the data source in "
                        "natural language — use the exact string from the list "
                        "above. If the user's description clearly matches one "
                        "of these supported sources, use that exact token. If "
                        "it doesn't match any supported source, leave "
                        "data_source null and add it to unclear_or_missing. "
                        "\n\n"
                        "If the user's request doesn't clearly state a field, "
                        "leave it null and add its name to unclear_or_missing "
                        "— do NOT guess a plausible-sounding value for "
                        "target_column or data_source specifically, since a "
                        "wrong guess there would silently point the pipeline "
                        "at the wrong data. metric_threshold may reasonably "
                        "default to a sensible value (e.g. 0.8) without "
                        "flagging it as unclear, since it's a preference, "
                        "not a fact about the user's data. "
                        "If anything is in unclear_or_missing, also propose a "
                        "single, specific clarifying_question."
                    ),
                },
                {"role":"user", "content": user_request},
            ],
            response_format=_Extraction,
        )
        extracted = completion.choices[0].message.parsed

        if extracted.unclear_or_missing:
            return ClarificationNeeded(
                missing_fields= extracted.unclear_or_missing,
                question_for_user=extracted.clarifying_question or "Could you Clarify the missing details?",
            )

        required = {
            "task_type": extracted.task_type,
            "target_column": extracted.target_column,
            "success_metric": extracted.success_metric,
            "metric_threshold": extracted.metric_threshold,
            "data_source": extracted.data_source,
        }

        missing = [name for name, value in required.items() if value is None]
        if missing:
            return ClarificationNeeded(
                missing_fields=missing,
                question_for_user=f"I need more detail on: {', '.join(missing)}.",
            )

        # --- Python-side data_source validation ---
        # The LLM might still produce a natural-language description
        # instead of the exact token load_data() requires. Catch it
        # HERE — before ProblemSpec is ever constructed — so the
        # failure surfaces as a clear ClarificationNeeded, not a
        # silent ValueError three stages later inside load_data().
        ds = extracted.data_source
        is_file_path = ds and (ds.endswith(".csv") or ds.endswith(".parquet"))
        if ds and ds not in SUPPORTED_DATA_SOURCES and not is_file_path:
            return ClarificationNeeded(
                missing_fields=["data_source"],
                question_for_user=(
                    f"I understood the data source as '{ds}', but that "
                    f"doesn't match any supported source. Supported "
                    f"built-in sources are: {SUPPORTED_DATA_SOURCES}. "
                    f"Did you mean one of these, or is it a file path "
                    f"ending in .csv/.parquet?"
                ),
            )

        try:
            return ProblemSpec(
                task_type=extracted.task_type,
                target_column=extracted.target_column,
                success_metric=extracted.success_metric,
                metric_threshold=extracted.metric_threshold,
                data_source=extracted.data_source,
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
    agent = RequirementAgent()

    print("=== Clear request ===")
    result = agent.run(
        "Predict whether a breast tumor is malignant or benign using the "
        "builtin breast cancer dataset, target column diagnosis, aim for "
        "at least 0.9 f1 score."
    )

    print(type(result).__name__, "->", result)

    print("\n=== Vague request ===")
    result = agent.run("Help me build somrthing to predict stuff.")
    print(type(result).__name__, "->", result)