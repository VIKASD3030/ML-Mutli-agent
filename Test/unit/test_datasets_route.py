"""
Test/unit/test_datasets_route.py

No LLM, no DB: a throwaway FastAPI app mounts only the datasets router, so
these run free and fast. The security-relevant cases (path restriction) are
the point — /datasets/peek takes a caller-supplied path.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import datasets
from api.schemas_api import RunCreateRequest
from tools.data_peek_tools import BUILTIN_BREAST_CANCER, BUILTIN_TARGET_NAME, peek_dataset_tool


@pytest.fixture()
def client(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    samples = tmp_path / "sample_data"
    uploads.mkdir()
    samples.mkdir()
    pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "x"]}).to_csv(samples / "tiny.csv", index=False)
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(samples / "tiny.parquet")
    pd.DataFrame({"z": [1]}).to_csv(tmp_path / "outside.csv", index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(datasets, "UPLOAD_DIR", uploads.relative_to(tmp_path))
    monkeypatch.setattr(datasets, "SAMPLE_DIR", samples.relative_to(tmp_path))
    monkeypatch.setattr(datasets, "ALLOWED_ROOTS", (datasets.UPLOAD_DIR, datasets.SAMPLE_DIR))

    app = FastAPI()
    app.include_router(datasets.router)
    return TestClient(app)


def test_builtin_peek_matches_sklearn_shape():
    peek = peek_dataset_tool(BUILTIN_BREAST_CANCER)
    assert (peek.row_count, peek.column_count) == (569, 31)
    assert BUILTIN_TARGET_NAME in {c.name for c in peek.columns}


def test_catalog_lists_builtin_first_and_dedupes_csv_parquet_twins(client):
    items = client.get("/datasets").json()
    assert items[0]["source"] == BUILTIN_BREAST_CANCER
    sources = [i["source"] for i in items]
    assert sources.count("sample_data/tiny.csv") == 1
    assert "sample_data/tiny.parquet" not in sources


def test_peek_builtin(client):
    r = client.get("/datasets/peek", params={"source": BUILTIN_BREAST_CANCER})
    assert r.status_code == 200
    assert r.json()["column_count"] == 31


def test_peek_sample_file_reports_missing_pct(client):
    r = client.get("/datasets/peek", params={"source": "sample_data/tiny.csv"})
    assert r.status_code == 200
    cols = {c["name"]: c for c in r.json()["columns"]}
    assert cols["a"]["missing_pct"] == pytest.approx(33.33, abs=0.01)


@pytest.mark.parametrize(
    "source",
    ["outside.csv", "../outside.csv", "sample_data/../outside.csv"],
)
def test_peek_rejects_paths_outside_allowed_folders(client, source):
    assert client.get("/datasets/peek", params={"source": source}).status_code == 400


def test_peek_rejects_unsupported_extension_and_missing_file(client):
    assert client.get("/datasets/peek", params={"source": "sample_data/x.txt"}).status_code == 400
    assert client.get("/datasets/peek", params={"source": "sample_data/nope.csv"}).status_code == 404


def test_constraints_override_only_applies_fields_the_client_sent():
    req = RunCreateRequest.model_validate({"context": "predict churn", "constraints": {"max_tuning_trials": 15}})
    assert req.constraints.model_dump(exclude_unset=True) == {"max_tuning_trials": 15}
