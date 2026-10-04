"""
tools/data_peek_tool.py

Deterministic tool: reads an uploaded/local file's SCHEMA only — no
cleaning, no full pipeline processing. This is what lets RequirementAgent
ground its extraction in real column names/types instead of guessing
from free text alone. Deliberately independent of load_data() — peeking
happens BEFORE a ProblemSpec exists, so there's no data_source/spec to
route through yet.
"""

from __future__ import annotations

import pandas as pd

from schemas.dataset_peek import ColumnPeek, DatasetPeek

SUPPORTED_EXTENSIONS = (".csv", ".parquet")
MAX_EXAMPLE_VALUES =5
BUILTIN_BREAST_CANCER = "builtin:breast_cancer"
BUILTIN_TARGET_NAME = "diagnosis"

def peek_dataset_tool(file_path: str) -> DatasetPeek:
    """Read a file's schema without cleaning or full processing.

    Parameters
    ----------
    file_path: str
        Path to a local CSV or parquet file (e.g. an uploaded file
        already saved to disk server-side).

    Returns
    -------
    DatasetPeek
        Per-column summary — enough for an LLM to identify a real target
        column and infer task_type, without ever seeing a full row.

    Raises
    ------
    ValueError
        If the file extension isn't supported — same explicit-failure
        discipline as load_data()'s own unsupported-source guard, rather
        than a silent best-effort attempt.
    """
    if file_path == BUILTIN_BREAST_CANCER:
        return peek_dataframe(load_builtin_breast_cancer_frame())
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {file_path!r}."
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )
    return peek_dataframe(df)


def load_builtin_breast_cancer_frame() -> pd.DataFrame:
    """sklearn's breast-cancer frame with its generic `target` column
    renamed to BUILTIN_TARGET_NAME. load_data() renames `target` to
    whatever spec.target_column is, so any name works downstream — the
    peek just shows a human-meaningful one so a user can say "predict
    diagnosis" and have it validate against the peeked columns."""
    from sklearn.datasets import load_breast_cancer

    frame = load_breast_cancer(as_frame=True).frame.copy()
    return frame.rename(columns={"target": BUILTIN_TARGET_NAME})


def peek_dataframe(df: pd.DataFrame) -> DatasetPeek:
    columns: list[ColumnPeek] = []
    for col in df.columns:
        series = df[col]
        missing_pct = round(series.isna().mean() * 100, 2)
        unique_vals = series.dropna().unique()

        columns.append(
            ColumnPeek(
                name=str(col),
                dtype=str(series.dtype),
                missing_pct=missing_pct,
                unique_count=len(unique_vals),
                example_value= [str(v) for v in unique_vals[:MAX_EXAMPLE_VALUES]],
            )
        )

    return DatasetPeek(row_count=len(df), column_count=len(df.columns), columns = columns)
