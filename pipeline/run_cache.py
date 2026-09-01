"""
pipeline/run_cache.py

A minimal in-memory store for the intermediate DataFrames agents need to
share within ONE pipeline run — the counterpart to PipelineRun, which
deliberately holds only reports/decisions, never raw data (same
"schemas describe data, they don't contain it" principle as DataProfile
never holding the DataFrame it describes).

Scope, deliberately narrow, same spirit as load_data() only supporting
one builtin source at first: single-process, in-memory, no persistence,
no thread-safety, no eviction. This is the smallest thing that lets
agents within one run share data — not production infrastructure.
"""

from __future__ import annotations

class RunCache:
    def __init__(self) -> None:
        # Nested dict: run_id -> {key -> value}. Generic key/value rather
        # than dedicated methods per artifact type (set_cleaned_df,
        # set_features, ...) — one flexible store instead of growing a
        # new method every time a future agent needs to cache something
        # new. Documented keys currently in use: "cleaned_df", "X", "y".
        self._store: dict[str, dict[str, object]] = {}

    def set(self, run_id: str, key: str, value: object) -> None:
        self._store.setdefault(run_id, {})[key] = value

    def get(self, run_id: str, key: str) -> object | None:
        return self._store.get(run_id, {}).get(key)

    def clear(self, run_id: str) -> None:
        self._store.pop(run_id, None)