"""
api/main.py — run locally with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import runs, uploads

app = FastAPI(title="ML Pipeline Agent API")
app.include_router(runs.router)
app.include_router(uploads.router)


@app.get("/")
def root() -> dict:
    """Service descriptor at the root.

    Without this, hitting http://127.0.0.1:8000 in a browser returns a bare
    404 — correct, since no route was registered there, but it reads like a
    failure. Returns JSON rather than redirecting to /docs so that curl and
    other programmatic clients get something meaningful too.
    """
    return {
        "service": "ML Pipeline Agent API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "upload_dataset": "POST /uploads",
            "create_run": "POST /runs",
            "list_runs": "GET /runs",
            "get_run": "GET /runs/{run_id}",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}