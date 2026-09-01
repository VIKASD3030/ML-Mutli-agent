"""
tests/unit/test_data_stage.py

No mocking needed here — pipeline/data_stage.py has zero LLM calls, so
these tests run against the real functions directly, same as every
manual test_*.py script you've already written, just formalized as
pytest assertions instead of print statements you eyeball.
"""

import pandas as pd
import pytest

from pipeline.data_stage import clean_and_profile
from schemas.problem_spec import ProblemSpec

@pytest.fixture
def classification_spec() -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="diagnosis",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
    )

def test_clean_and_profile_missing_target_column_b(classification_spec):
    """Regression test for the guard we built way back — if the target
    column doesn't exist, blocking_issue must be set, immediately, with
    no further processing attempted."""
    df = pd.DataFrame({"some_other_column": [1,2,3]})

    _cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert profile.blocking_issue is not None
    assert "diagnosis" in profile. blocking_issue
    assert profile.is_usable is False

def test_clean_and_profile_imputes_missing_numericvalue(classification_spec):
    """Regression test for median imputation — this is the exact
    behavior we walked through line-by-line early in this project."""
    df = pd.DataFrame({
        "diagnosis": [0,1,0,1],
        "feature_a": [10.0, None, 30.0, 40.0],
    })

    cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert profile.blocking_issue is None
    assert cleaned_df["feature_a"].isna().sum() == 0
    expected_median = pd.Series([10.0, 30.0, 40.0]).median()
    assert cleaned_df["feature_a"].iloc[1] == expected_median
    assert len(profile.cleaning_actions_taken) == 1
    assert profile.cleaning_actions_taken[0].action == "median_imputation"

def test_clean_and_profile_drops_exact_duplicate_rows(classification_spec):
    df = pd.DataFrame({
        "diagnosis": [0,0,1],
        "feature_a": [5.0, 5.0, 10.0],
    })

    cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert len(cleaned_df) == 2
    assert any(a.action == "dropped_duplicate_rows" for a in profile.cleaning_actions_taken)
    

def test_clean_and_profile_flags_severe_class_imbalance(classification_spec):
    """91 rows of class 0, 9 rows of class 1 — minority share (9%) is
    below the 10% threshold data_stage.py itself defines as 'severe'."""
    df = pd.DataFrame({
        "diagnosis": [0] *91 + [1] * 9,
        "feature_a": list(range(100)),
    })

    _cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert profile.warnings
    assert any("imbalance" in w.lower() for w in profile.warnings)