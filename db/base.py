"""
db/base.py

SQLAlchemy 2.0-style declarative base — every table class inherits from
this. One shared Base is what lets Alembic (migrations) and the engine
discover every table defined anywhere in db/models.py.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass