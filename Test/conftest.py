"""
tests/conftest.py

Shared pytest fixtures. The most important one is `fake_openai_client`,
which every agent test uses instead of a real OpenAI() client — this is
what makes agent tests fast, free, and deterministic, exercising the
testability every agent's `client: OpenAI | None = None` parameter was
built for from the very first agent onward.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

class FakeOpenAIClient:
    """A minimal stand-in for openai.OpenAI(). Only implements the one
    method every agent actually calls: client.beta.chat.completions.parse().

    Usage: construct with the Pydantic object you want returned as the
    "parsed" LLM output — the fake ignores the actual prompt/messages
    entirely and just hands back whatever you told it to.
    """
    def __init__(self, canned_response: BaseModel) -> None:
        self._canned_response = canned_response
        self.call_count = 0
        self.last_messages: list[dict] | None = None # lets a test assert
        # on WHAT was sent, not just that a call happened.

    @property
    def beta(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=self))

    def parse(self, model: str, messages: list[dict], response_format: Any, **kwargs) -> SimpleNamespace:
        self.call_count += 1
        self.last_messages = messages

        # Mimic the real OpenAI response shape closely enough for
        # timed_llm_call() to work unmodified — it reads .usage and
        # .choices[0].message.parsed, so both need to exist here.
        fake_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
        )
        fake_message = SimpleNamespace(parsed=self._canned_response)
        fake_choice = SimpleNamespace(message=fake_message)
        return SimpleNamespace(choices=[fake_choice], usage=fake_usage)


@pytest.fixture
def make_fake_client():
    """Factory fixture — tests call make_fake_client(some_judgment_object)
    to get a FakeOpenAIClient pre-loaded with whatever response they want
    this specific test to receive."""
    def _make(canned_response: BaseModel) -> FakeOpenAIClient:
        return FakeOpenAIClient(canned_response)
    return _make