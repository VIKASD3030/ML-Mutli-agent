"""
core/config.py

Centralized settings via pydantic-settings — same validation discipline
as every schema in this project, applied to environment configuration
instead of pipeline data. Reads from a .env file if present, environment
variables otherwise.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:root%40123@localhost:5432/ml_pipeline"
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
    