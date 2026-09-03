"""
Test/integration/test_requirement_agent.py

INTEGRATION tests — RequirementAgent makes an LLM call, so every test here
injects the fake client from Test/conftest.py via make_fake_client. No
network, no cost, no variance.

RequirementAgent is the only agent whose LLM output is not a judgement but
a STRUCTURED EXTRACTION, and the only one that runs before a PipelineRun
exists. Its whole design rests on one idea: the model fills in a loose,
all-optional _Extraction, and Python then decides whether those proposed
values are usable. OpenAI's structured outputs enforce JSON shape only —
never that a proposed target_column is a column that actually exists — so
the Python-side checks are the real contract, and their ORDER is part of
that contract.

That ordering is what the first test below exists for, and it is the kind
of bug that is invisible from the outside: both code paths return a
ClarificationNeeded, so the agent "works" either way. Only the wording the
user actually receives differs — a precise, deterministic message naming
the bad column and listing the real ones, versus whatever vaguer sentence
the model happened to generate about the same mistake. A test that only
asserted `isinstance(result, ClarificationNeeded)` would pass against the
bug, which is why these assert on the message text.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agents.requirement_agent import RequirementAgent, _Extraction
from schemas.dataset_peek import ColumnPeek, DatasetPeek
from schemas.problem_spec import ClarificationNeeded, ProblemSpec
import agents.requirement_agent as requirement_agent


REAL_COLUMNS = ["customer_id", "monthly_spend", "churned"]


@pytest.fixture
def peeked_file(monkeypatch, tmp_path):
    """Give the agent a real file to peek at, and return its path.

    A genuine CSV written to tmp_path, peeked by the REAL
    peek_dataset_tool — no mocking of the tool. The peek is the
    deterministic half of this agent and there is no reason to fake it;
    only the LLM call is faked."""
    df = pd.DataFrame({
        "customer_id": range(30),
        "monthly_spend": [i * 3.7 for i in range(30)],
        "churned": [i % 3 == 0 for i in range(30)],
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return str(path)


def _extraction(**overrides) -> _Extraction:
    """A complete, otherwise-valid extraction. Tests override exactly the
    one field they are exercising, so nothing else can be the reason a
    given assertion fires."""
    base = dict(
        task_type="classification",
        target_column="churned",
        success_metric="f1",
        metric_threshold=0.9,
        data_source=None,  # ignored on the file path; Python supplies it
        unclear_or_missing=[],
        clarifying_question=None,
    )
    base.update(overrides)
    return _Extraction(**base)


# --------------------------------------------------------------------------
# THE ORDERING REGRESSION TEST
# --------------------------------------------------------------------------

def test_invented_target_column_beats_the_llms_own_clarifying_question(
    make_fake_client, peeked_file
):
    """THE regression test for the ordering bug.

    Scenario that exposed it: the model proposes a target_column that does
    not exist in the peeked file, AND leaves unclear_or_missing empty. With
    the checks in the wrong order the unclear_or_missing branch is reached
    first and returns early, so the precise real-columns message can never
    be produced.

    The assertion is deliberately on the MESSAGE, not on the type. Both
    orderings return a ClarificationNeeded, so type alone cannot tell the
    fixed code from the broken code. What distinguishes them is that the
    deterministic backstop names the invented column and enumerates the
    real ones — information the model does not reliably provide about its
    own mistake."""
    canned = _extraction(target_column="customer_status")  # not a real column
    fake_client = make_fake_client(canned)

    result = RequirementAgent(client=fake_client).run(
        "Predict the customer_status column.", file_path=peeked_file
    )

    assert isinstance(result, ClarificationNeeded)
    assert result.missing_fields == ["target_column"]

    # The precise, deterministic message — this is the whole point.
    assert "isn't a column in this dataset" in result.question_for_user
    assert "customer_status" in result.question_for_user
    # ...and it lists the real columns, so the user can actually correct it.
    for real_column in REAL_COLUMNS:
        assert real_column in result.question_for_user


def test_invented_target_column_wins_even_when_the_llm_also_flags_it(
    make_fake_client, peeked_file
):
    """The sharper version, and the one that genuinely pins the ORDER.

    Here the model both invents a column AND populates unclear_or_missing
    with its own clarifying_question. Two branches now compete for the same
    return, so this test can only pass if the real-columns check runs
    first. In the test above, an empty unclear_or_missing means the buggy
    ordering would fall through and still reach the right message —
    here it cannot.

    The negative assertion matters as much as the positive one: the
    model's wording must NOT be what comes back."""
    canned = _extraction(
        target_column="customer_status",
        unclear_or_missing=["target_column"],
        clarifying_question="Which column should I predict?",
    )
    fake_client = make_fake_client(canned)

    result = RequirementAgent(client=fake_client).run(
        "Predict the customer_status column.", file_path=peeked_file
    )

    assert isinstance(result, ClarificationNeeded)
    assert "isn't a column in this dataset" in result.question_for_user
    # The vaguer, model-generated question lost, as it must.
    assert result.question_for_user != "Which column should I predict?"
    assert "Which column should I predict?" not in result.question_for_user


def test_the_backstop_does_not_fire_on_a_valid_column(make_fake_client, peeked_file):
    """The guard has to be precise, not merely loud. A target_column that
    IS real must sail through to a ProblemSpec, with data_source set by
    Python to the actual file path rather than anything the model said."""
    fake_client = make_fake_client(_extraction(target_column="churned"))

    result = RequirementAgent(client=fake_client).run(
        "Predict whether a customer will churn.", file_path=peeked_file
    )

    assert isinstance(result, ProblemSpec)
    assert result.target_column == "churned"
    assert result.data_source == peeked_file


def test_null_target_column_is_reported_as_missing_not_as_invented(
    make_fake_client, peeked_file
):
    """The reason the backstop is guarded on target_column being non-null.

    A null target is a MISSING value, not a wrong one. Without the guard,
    `None not in real_columns` is True and the user would be told
    "'None' isn't a column in this dataset" — technically accurate,
    actively unhelpful. It must fall through to the missing-fields path
    instead."""
    fake_client = make_fake_client(_extraction(target_column=None))

    result = RequirementAgent(client=fake_client).run(
        "Predict something from this file.", file_path=peeked_file
    )

    assert isinstance(result, ClarificationNeeded)
    assert "target_column" in result.missing_fields
    assert "isn't a column in this dataset" not in result.question_for_user
    assert "None" not in result.question_for_user


def test_backstop_is_skipped_entirely_when_there_is_no_file(make_fake_client):
    """No peek means no real column names to check against, so the
    backstop must not run at all — the text-only path is unchanged by this
    fix. Here the model's own unclear_or_missing is the only signal
    available, and it must still be honoured."""
    canned = _extraction(
        target_column="anything_at_all",
        data_source="builtin:breast_cancer",
        unclear_or_missing=["target_column"],
        clarifying_question="Which column should I predict?",
    )
    fake_client = make_fake_client(canned)

    result = RequirementAgent(client=fake_client).run("Predict something.")

    assert isinstance(result, ClarificationNeeded)
    # With no peek, the model's question is the correct thing to surface.
    assert result.question_for_user == "Which column should I predict?"


# --------------------------------------------------------------------------
# Supporting behaviour, so the ordering tests cannot pass for stale reasons
# --------------------------------------------------------------------------

def test_peek_runs_before_the_llm_and_grounds_the_prompt(make_fake_client, peeked_file):
    """The agent's stated design: the deterministic tool call happens
    FIRST, and its output is what the model reasons over. Asserting the
    real column names reach the system prompt is what proves the peek is
    not merely computed and discarded."""
    fake_client = make_fake_client(_extraction())

    RequirementAgent(client=fake_client).run("Predict churn.", file_path=peeked_file)

    assert fake_client.call_count == 1
    system_msg = fake_client.last_messages[0]
    assert system_msg["role"] == "system"
    for real_column in REAL_COLUMNS:
        assert real_column in system_msg["content"]


def test_unsupported_file_type_raises_before_any_llm_call(make_fake_client, tmp_path):
    """peek_dataset_tool rejects unsupported extensions, and it runs before
    the LLM — so a bad file must cost nothing. Same short-circuit
    discipline as every other agent's PipelineBlockedError path."""
    bad = tmp_path / "data.txt"
    bad.write_text("not a real dataset")
    fake_client = make_fake_client(_extraction())

    with pytest.raises(ValueError):
        RequirementAgent(client=fake_client).run("Predict churn.", file_path=str(bad))

    assert fake_client.call_count == 0


def test_invalid_metric_for_task_type_is_caught_and_explained(
    make_fake_client, peeked_file
):
    """ProblemSpec's _metric_matches_task validator is the last line of
    defence, and structured outputs cannot enforce it — success_metric is a
    plain string in _Extraction, so the model can legally propose a
    regression metric for a classification task. That must become a
    ClarificationNeeded, not an unhandled ValidationError."""
    fake_client = make_fake_client(
        _extraction(task_type="classification", success_metric="rmse")
    )

    result = RequirementAgent(client=fake_client).run(
        "Predict churn.", file_path=peeked_file
    )

    assert isinstance(result, ClarificationNeeded)
    assert "rmse" in result.question_for_user
