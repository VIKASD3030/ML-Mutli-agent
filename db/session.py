"""
db/session.py

Engine + session factory + the get_db() dependency FastAPI routes will
use to get a database session per-request. Also includes init_db() — a
dev-only convenience for creating all 26 tables directly from the
SQLAlchemy models, WITHOUT Alembic. This exists purely to let you test
db/sync.py's round-trip immediately, before setting up real migrations.
Once Alembic is in place, init_db() should stop being used for anything
beyond a fresh local dev database — Alembic becomes the actual source
of truth for schema changes from that point on, since create_all() has
no concept of altering an existing table, only creating missing ones.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import settings
from db.base import Base

# echo=False keeps SQLAlchemy quiet by default — flip to True temporarily
# if you ever need to see the exact SQL being generated, e.g. while
# debugging whether db/sync.py's cascade deletes are actually firing.
engine = create_engine(settings.database_url, echo=False)

# expire_on_commit=False matters specifically for load_run_state():
# without it, every attribute on an object would need re-fetching from
# the DB the instant session.commit() runs, which would silently break
# returning a freshly-loaded PipelineRun's data right after a commit.
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — `db: Session = Depends(get_db)` in a route.
    The yield/finally pattern guarantees the session is closed even if
    the route raises an exception partway through — same "always clean
    up, even on the failure path" discipline as PipelineBlockedError's
    role elsewhere in this project, just applied to a DB connection
    instead of pipeline control flow."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """DEV ONLY. Creates every table defined in db/models.py directly
    from the SQLAlchemy metadata — no migration history, no ability to
    alter an existing table's columns later, just "make these 26 tables
    exist if they don't already." Fine for a fresh local Postgres;
    actively wrong to rely on once real data exists, since re-running it
    after a model change won't pick up column changes on tables that
    already exist — Alembic is the actual fix for that, coming next."""
    Base.metadata.create_all(bind=engine)