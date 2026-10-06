"""
api/main.py — run locally with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import datasets, runs, uploads

app = FastAPI(title="ML Pipeline Agent API")

# Required for the React frontend (Vite dev server on a different origin,
# e.g. http://localhost:5173) to call this API from a browser at all — with
# no CORS policy, every request from the frontend is blocked by the browser
# before it even reaches a route, regardless of any application logic.
# This is transport-layer plumbing, not a change to any request/response
# shape or business logic. Origins are intentionally limited to local dev
# hosts; widen this (or make it env-driven) before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(uploads.router)
app.include_router(datasets.router)


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
            "list_datasets": "GET /datasets",
            "peek_dataset": "GET /datasets/peek?source=...",
            "create_run": "POST /runs",
            "list_runs": "GET /runs",
            "get_run": "GET /runs/{run_id}",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}