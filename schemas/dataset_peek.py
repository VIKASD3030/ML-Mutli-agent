"""
schemas/dataset_peek.py

DatasetPeek — produced by tools/data_peek_tool.py. A lightweight,
schema-only summary of an uploaded dataset, used to GROUND
RequirementAgent's extraction in real column names/types instead of
letting the LLM guess them from free text alone.

Per-column SUMMARY only — name, dtype, missingness, cardinality, a
handful of example values — never full rows. Same "LLM never touches
raw data directly" boundary as DataProfile/EDAReport elsewhere in this
project, just applied one step earlier, before a ProblemSpec even exists.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

class ColumnPeek(BaseModel):
    name:str
    dtype:str
    missing_pct: float = Field(..., ge=0.0, le=100.0)
    unique_count: int
    example_value: list[str] = Field(default_factory=list)

class DatasetPeek(BaseModel):
    row_count:int
    column_count: int
    columns: list[ColumnPeek]