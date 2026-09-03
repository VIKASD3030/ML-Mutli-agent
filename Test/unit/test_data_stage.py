"""
tests/unit/test_data_stage.py

No mocking needed here — pipeline/data_stage.py has zero LLM calls, so
these tests run against the real functions directly, same as every
manual test_*.py script you've already written, just formalized as
pytest assertions instead of print statements you eyeball.
"""

import pandas as pd
import pytest

from pipeline.data_stage import (
    clean_and_profile,
    is_categorical_column,
    is_datetime_column,
)
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

# --------------------------------------------------------------------------
# Categorical and datetime handling
# --------------------------------------------------------------------------

def test_clean_and_profile_mode_imputes_missing_categorical(classification_spec):
    """Categorical counterpart to the median-imputation test above. A median
    is undefined for strings, so the most frequent value stands in — recorded
    as its own action name so a reader can tell WHICH strategy was applied,
    not merely that something was imputed."""
    df = pd.DataFrame({
        "diagnosis": [0, 1, 0, 1, 0],
        "city": ["NY", "NY", "LA", None, "NY"],
    })

    cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert cleaned_df["city"].isna().sum() == 0
    assert cleaned_df["city"].iloc[3] == "NY"  # the mode

    actions = [a for a in profile.cleaning_actions_taken if a.action == "mode_imputation"]
    assert len(actions) == 1
    assert actions[0].column == "city"
    assert "mode" in actions[0].rationale.lower()
    # Must NOT be misreported as the numeric strategy.
    assert not any(
        a.action == "median_imputation" and a.column == "city"
        for a in profile.cleaning_actions_taken
    )


def test_clean_and_profile_leaves_datetime_columns_untouched(classification_spec):
    """Datetime columns get no missing-value strategy at this stage — they are
    decomposed in engineer_features() instead, so imputing here would invent a
    timestamp nobody asked for. The column must survive unchanged, NaT and
    all, with no CleaningAction claiming otherwise."""
    df = pd.DataFrame({
        "diagnosis": [0, 1, 0, 1],
        "signup_date": pd.to_datetime(
            ["2024-01-01", None, "2024-03-01", "2024-04-01"]
        ),
    })

    cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["signup_date"])
    assert cleaned_df["signup_date"].isna().sum() == 1  # NaT preserved
    assert not any(a.column == "signup_date" for a in profile.cleaning_actions_taken)


def test_all_categorical_column_is_left_alone_rather_than_invented(classification_spec):
    """An all-missing categorical has no mode to impute with. Leaving it
    untouched and recording nothing is the honest outcome — the alternative
    is fabricating a category. The run continues rather than raising."""
    # The extra unique column keeps every row distinct — without it
    # drop_duplicates() collapses (0, None) and (1, None) down to two rows
    # and the count below would be measuring deduplication, not imputation.
    df = pd.DataFrame({
        "diagnosis": [0, 1, 0, 1],
        "row_id": [1.0, 2.0, 3.0, 4.0],
        "empty_cat": [None, None, None, None],
    })

    cleaned_df, profile = clean_and_profile(df, classification_spec)

    assert len(cleaned_df) == 4
    assert cleaned_df["empty_cat"].isna().sum() == 4
    assert not any(
        a.action == "mode_imputation" for a in profile.cleaning_actions_taken
    )


def test_column_type_predicates_classify_correctly():
    """Direct test of the shared predicates, since feature_stage.py imports
    them and both stages depend on them agreeing.

    The ordering assertion is the important one: datetime is checked first,
    so a datetime column is never also reported as categorical. Without that
    short-circuit a timestamp column could be one-hot encoded into one column
    per distinct instant."""
    dt = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    text = pd.Series(["a", "b"])                      # StringDtype on pandas 3.x
    obj = pd.Series(["a", 1], dtype=object)           # genuinely mixed -> object
    cat = pd.Series(["a", "b"], dtype="category")
    num = pd.Series([1.0, 2.0])

    assert is_datetime_column(dt) is True
    assert is_categorical_column(dt) is False  # datetime wins, never both

    # All three text-ish representations must be caught. `text` is the one
    # that matters most in practice and the one an object-only check would
    # miss entirely on pandas 3.x, where plain strings infer as StringDtype.
    assert is_categorical_column(text) is True
    assert is_categorical_column(obj) is True
    assert is_categorical_column(cat) is True
    assert is_datetime_column(text) is False

    assert is_categorical_column(num) is False
    assert is_datetime_column(num) is False


def test_date_like_strings_are_not_treated_as_datetimes():
    """Explicitly pinned scope limit: only genuinely datetime64-typed columns
    get datetime handling. Auto-parsing date-looking strings is a fuzzier
    problem with real false-positive risk (a zero-padded ID reading as a
    date), so an object column of date strings stays categorical."""
    date_strings = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"])

    assert is_datetime_column(date_strings) is False
    assert is_categorical_column(date_strings) is True
