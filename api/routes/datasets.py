"""
api/routes/datasets.py

Read-only dataset discovery for the frontend:

- GET /datasets       the benchmarks a run can use without uploading anything
- GET /datasets/peek  the schema-only DatasetPeek of one of them (or of an
                      uploaded file) — the same object RequirementAgent
                      already grounds itself in, now visible to the user
                      before they start a run.

`source` is caller-supplied, so peek is restricted to the builtin token and
to files that resolve inside uploads/ or sample_data/. Without that, this
endpoint would be an arbitrary-file-read oracle for any .csv/.parquet the
server process can open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from schemas.dataset_peek import DatasetPeek
from tools.data_peek_tools import BUILTIN_BREAST_CANCER, peek_dataset_tool

UPLOAD_DIR = Path("uploads")
SAMPLE_DIR = Path("sample_data")
ALLOWED_ROOTS = (UPLOAD_DIR, SAMPLE_DIR)
SUPPORTED_SUFFIXES = {".csv", ".parquet"}

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetInfo(BaseModel):
    source: str
    label: str
    description: str
    task_type: Optional[Literal["classification", "regression"]] = None


def _catalog() -> list[DatasetInfo]:
    items = [
        DatasetInfo(
            source=BUILTIN_BREAST_CANCER,
            label="Breast Cancer Diagnostic (Wisconsin)",
            description=(
                "Real-world clinical benchmark with 30 continuous numeric "
                "features computed from digitized FNA images of breast masses."
            ),
            task_type="classification",
        )
    ]
    if SAMPLE_DIR.is_dir():
        # .csv and .parquet twins of the same table (customers.*) would show
        # up as duplicates — prefer the csv when both exist.
        seen_stems: set[str] = set()
        for path in sorted(SAMPLE_DIR.iterdir(), key=lambda p: (p.suffix != ".csv", p.name)):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.stem in seen_stems:
                continue
            seen_stems.add(path.stem)
            items.append(
                DatasetInfo(
                    source=path.as_posix(),
                    label=path.stem.replace("_", " ").title(),
                    description=f"Sample file {path.name} shipped in {SAMPLE_DIR.name}/.",
                )
            )
    return items


def _resolve_allowed(source: str) -> str:
    if source == BUILTIN_BREAST_CANCER:
        return source
    path = Path(source)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Unsupported source {source!r}: expected a .csv/.parquet file or a builtin token.")
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in ALLOWED_ROOTS):
        raise HTTPException(400, f"Source {source!r} is outside the uploads/ and sample_data/ folders.")
    if not resolved.is_file():
        raise HTTPException(404, f"Dataset not found: {source!r}")
    return str(resolved)


@router.get("", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    return _catalog()


@router.get("/peek", response_model=DatasetPeek)
def peek_dataset(source: str = Query(..., min_length=1)) -> DatasetPeek:
    safe_source = _resolve_allowed(source)
    try:
        return peek_dataset_tool(safe_source)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:
        raise HTTPException(422, f"Could not read dataset: {e}") from None
