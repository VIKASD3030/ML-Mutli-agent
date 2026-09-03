"""
tests/unit/test_feature_stage.py

No mocking — engineer_features() is pure pandas/numpy, same "test the
real deterministic function directly" philosophy as test_data_stage.py.
"""

import pandas as pd
import pytest

from pipeline.data_stage import clean_and_profile
from pipeline.feature_stage import MAX_CATEGORICAL_CARDINALITY, engineer_features
from schemas.data_profile import DataProfile
from schemas.problem_spec import ProblemSpec


@pytest.fixture
def classification_spec() -> ProblemSpec:
    return ProblemSpec(
        task_type="classification",
        target_column="target",
        success_metric="f1",
        metric_threshold=0.90,
        data_source="builtin:breast_cancer",
    )


def _minimal_profile(row_count: int) -> DataProfile:
    """A hand-built DataProfile, not a real clean_and_profile() output —
    engineer_features() only reads profile.class_balance for its notes
    string, so a minimal stand-in is enough and keeps this test fast and
    fully isolated from data_stage.py."""
    return DataProfile(
        row_count=row_count,
        column_count=0,
        column_schema=[],
        class_balance={"0": 0.5, "1": 0.5},
    )


def test_engineer_features_drops_near_zero_variance_column(classification_spec):
    """Regression test for the exact check we walked through line-by-line:
    a constant column should be dropped for LOW VARIANCE, not for low
    signal — the rationale text should say so specifically."""
    df = pd.DataFrame({
        "target": [0, 1, 0, 1, 0, 1],
        "informative": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "constant_col": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # zero variance
    })

    _X, _y, report = engineer_features(df, _minimal_profile(len(df)), classification_spec)

    dropped_names = [d.name for d in report.dropped_features]
    assert "constant_col" in dropped_names
    dropped_entry = next(d for d in report.dropped_features if d.name == "constant_col")
    assert "variance" in dropped_entry.rationale.lower()
    assert "constant_col" not in report.final_feature_names


def test_engineer_features_drops_low_signal_column(classification_spec):
    """A column with essentially zero correlation to the target should be
    dropped for LOW SIGNAL, distinct from the low-variance case above.

    The uncorrelated column is CONSTRUCTED, not sampled. Random noise was
    the obvious choice but does not work here: the drop threshold is
    0.01, while a random column of length n has a sample correlation of
    roughly 1/sqrt(n) with anything — about 0.07 at n=200, seven times
    over the threshold. Making the test reliable by sampling would need
    n in the tens of thousands and still be a coin flip.

    Instead: the target alternates 0,1,0,1... and 'noise' is constant
    within each adjacent pair (0,0,1,1,2,2,...). Every value of noise is
    paired with exactly one 0 and one 1, so the covariance is exactly
    zero by construction — deterministically, at any n — while the
    column still has plenty of variance and so cannot be confused with
    the near-zero-variance drop tested above."""
    n = 200
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "noise": [float(i // 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    dropped_names = [d.name for d in report.dropped_features]
    assert "noise" in dropped_names
    dropped_entry = next(d for d in report.dropped_features if d.name == "noise")
    assert "signal threshold" in dropped_entry.rationale.lower()


def test_engineer_features_flags_leakage_but_does_not_drop_it(classification_spec):
    """THE core behavior this whole stage exists for: a feature that's
    suspiciously (near-perfectly) correlated with the target must be
    FLAGGED in leakage_warnings, but must still survive into
    final_feature_names — asymmetric handling, not silent removal."""
    n = 100
    target = [i % 2 for i in range(n)]
    df = pd.DataFrame({
        "target": target,
        # Deterministically identical to the target — as close to a
        # perfect leak as possible, well above the 0.98 threshold.
        "leaked_column": target,
        "normal_feature": [i * 0.37 % 5 for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert len(report.leakage_warnings) >= 1
    assert any("leaked_column" in w for w in report.leakage_warnings)
    # The critical assertion: flagged, but NOT dropped.
    assert "leaked_column" in report.final_feature_names
    assert not any(d.name == "leaked_column" for d in report.dropped_features)


def test_engineer_features_creates_interaction_of_top_two_correlated(classification_spec):
    """Regression test for the interaction-feature mechanism — should
    combine whichever two SURVIVING columns have the highest correlation,
    named exactly '{a}__x__{b}'."""
    n = 100
    target = [i % 2 for i in range(n)]
    df = pd.DataFrame({
        "target": target,
        "strong_a": [t + 0.1 * (i % 3) for i, t in enumerate(target)],
        "strong_b": [t + 0.2 * (i % 3) for i, t in enumerate(target)],
        "weak": [(i * 7) % 11 for i in range(n)],  # deliberately uncorrelated
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert len(report.engineered_features) == 1
    engineered_name = report.engineered_features[0].name
    assert "__x__" in engineered_name
    assert engineered_name in report.final_feature_names

# --------------------------------------------------------------------------
# Categorical and datetime handling
# --------------------------------------------------------------------------

def test_low_cardinality_categorical_is_one_hot_encoded(classification_spec):
    """A small categorical column must be expanded into binary columns that
    then go through the ORDINARY filtering path — not a parallel one. The
    expansion itself is recorded as a single engineered_features entry naming
    the source column, so a reader can trace where the dummies came from."""
    n = 60
    target = [i % 2 for i in range(n)]
    df = pd.DataFrame({
        "target": target,
        # Perfectly aligned with the target, so the dummies carry real signal
        # and survive the low-signal filter.
        "grade": ["high" if t else "low" for t in target],
        "numeric_noise": [float(i % 5) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    expansion = [e for e in report.engineered_features if e.name == "grade__onehot"]
    assert len(expansion) == 1
    assert "one-hot" in expansion[0].rationale.lower()
    assert "2 binary columns" in expansion[0].rationale

    # The dummy columns themselves reached the feature matrix. The "__x__"
    # exclusion matters: the interaction term built from the two surviving
    # dummies is also named with a "grade_" prefix, but it is an ENGINEERED
    # product column, not a one-hot dummy, so it has no entry of its own in
    # correlation_summary (correlations are scored before it exists).
    dummy_names = [
        c for c in report.final_feature_names
        if c.startswith("grade_") and "__x__" not in c
    ]
    assert dummy_names
    # The raw categorical column never survives as itself.
    assert "grade" not in report.final_feature_names
    # And they were scored by the same correlation pass as everything else.
    assert all(name in report.correlation_summary for name in dummy_names)


def test_one_hot_dummies_go_through_the_existing_filters(classification_spec):
    """The dummies are not privileged. A single-category column expands into
    one all-ones dummy with zero variance, which the EXISTING low-variance
    rule must drop — proving the encoded columns are filtered by the same
    logic as native numeric ones rather than force-kept."""
    n = 40
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "constant_cat": ["only_value"] * n,
        "signal": [float(i % 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    # The expansion still happened and was recorded...
    assert any(e.name == "constant_cat__onehot" for e in report.engineered_features)
    # ...but the resulting dummy was dropped by the ordinary variance rule.
    dropped_names = [d.name for d in report.dropped_features]
    assert "constant_cat_only_value" in dropped_names
    dropped_entry = next(
        d for d in report.dropped_features if d.name == "constant_cat_only_value"
    )
    assert "variance" in dropped_entry.rationale.lower()
    assert "constant_cat_only_value" not in report.final_feature_names


def test_high_cardinality_categorical_is_dropped_not_expanded(classification_spec):
    """Above the cardinality limit the column is dropped outright. The
    assertion that matters most is the negative one: NO dummy columns appear.
    A 50-value column silently becoming 50 features is exactly the
    dimensionality explosion the limit exists to prevent."""
    n = 50
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "user_id": [f"user_{i}" for i in range(n)],  # 50 uniques > limit of 20
        "signal": [float(i % 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    dropped_entry = next(d for d in report.dropped_features if d.name == "user_id")
    assert "high-cardinality" in dropped_entry.rationale.lower()
    assert str(MAX_CATEGORICAL_CARDINALITY) in dropped_entry.rationale
    assert "50" in dropped_entry.rationale

    # Not expanded, not recorded as an expansion, not present anywhere.
    assert not any(c.startswith("user_id_") for c in report.final_feature_names)
    assert not any(e.name == "user_id__onehot" for e in report.engineered_features)


def test_cardinality_limit_is_inclusive_at_the_boundary(classification_spec):
    """`unique_count <= MAX_CATEGORICAL_CARDINALITY` is encoded, so exactly 20
    distinct values must still be one-hot encoded rather than dropped. Pinned
    because an off-by-one here changes behaviour silently."""
    n = 80
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "bucket": [f"b{i % MAX_CATEGORICAL_CARDINALITY}" for i in range(n)],
    })
    assert df["bucket"].nunique() == MAX_CATEGORICAL_CARDINALITY

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert any(e.name == "bucket__onehot" for e in report.engineered_features)
    assert not any(d.name == "bucket" for d in report.dropped_features)


def test_datetime_column_is_decomposed_into_derived_parts(classification_spec):
    """A raw datetime64 column cannot be fed to a RandomForest, so it is
    decomposed into four numeric parts and the original is dropped with a
    rationale naming what replaced it. Every derived part is recorded as an
    engineered feature so the expansion is traceable."""
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "signup_date": dates,
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    expected = {
        "signup_date__year",
        "signup_date__month",
        "signup_date__day_of_week",
        "signup_date__is_weekend",
    }
    engineered_names = {e.name for e in report.engineered_features}
    assert expected <= engineered_names

    # The raw column is gone from the feature matrix and recorded as dropped.
    assert "signup_date" not in report.final_feature_names
    raw_drop = next(d for d in report.dropped_features if d.name == "signup_date")
    assert "decomposed" in raw_drop.rationale.lower()
    assert "day_of_week" in raw_drop.rationale

    # The derived parts were scored by the same correlation pass, and each
    # either survived or was dropped by the ordinary rules — never ignored.
    accounted = set(report.final_feature_names) | {d.name for d in report.dropped_features}
    for name in expected:
        assert name in report.correlation_summary
        assert name in accounted


def test_datetime_derived_parts_are_filtered_by_the_existing_rules(classification_spec):
    """Derived datetime parts get no special protection either. With all dates
    inside one year, `year` is constant and must be dropped by the SAME
    low-variance rule that drops any other constant column."""
    n = 40
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    year_drop = next(
        d for d in report.dropped_features if d.name == "signup_date__year"
    )
    assert "variance" in year_drop.rationale.lower()
    assert "signup_date__year" not in report.final_feature_names


# --------------------------------------------------------------------------
# End-to-end: no column disappears without a trace
# --------------------------------------------------------------------------

def test_every_original_column_fate_is_traceable_end_to_end():
    """THE guarantee this whole feature exists for.

    Before this change, categorical and datetime columns were invisible to
    select_dtypes(include="number") and vanished with NO record anywhere —
    the EDAReport simply never mentioned them. This runs a mixed-type frame
    through the real clean_and_profile() -> engineer_features() path and
    asserts that EVERY original column can be accounted for: it either
    survived into final_feature_names, or was explicitly dropped, or was
    explicitly transformed into something that did.
    """
    n = 60
    rows = range(n)
    target = [i % 2 for i in rows]
    raw = pd.DataFrame({
        "outcome": target,
        "spend": [float(i) * 1.5 for i in rows],
        "tenure_months": [float(i % 12) for i in rows],
        "plan": ["pro" if t else "basic" for t in target],   # low-cardinality
        "account_ref": [f"acct_{i}" for i in rows],          # high-cardinality
        "signed_up": pd.date_range("2021-01-01", periods=n, freq="17D"),
    })

    spec = ProblemSpec(
        task_type="classification",
        target_column="outcome",
        success_metric="f1",
        metric_threshold=0.9,
        data_source="builtin:breast_cancer",  # unused; nothing loads here
    )

    cleaned_df, profile = clean_and_profile(raw, spec)
    X, y, report = engineer_features(cleaned_df, profile, spec)

    kept = set(report.final_feature_names)
    dropped = {d.name for d in report.dropped_features}
    engineered = {e.name for e in report.engineered_features}

    for column in raw.columns:
        if column == spec.target_column:
            continue
        accounted_directly = column in kept or column in dropped
        # A transformed column is accounted for by its offspring: the one-hot
        # marker, or any derived/dummy column carrying its name as a prefix.
        accounted_by_expansion = (
            f"{column}__onehot" in engineered
            or any(name.startswith(f"{column}__") for name in engineered)
            or any(name.startswith(f"{column}_") for name in kept | dropped)
        )
        assert accounted_directly or accounted_by_expansion, (
            f"Column '{column}' vanished without a trace - it appears in "
            f"neither final_feature_names, dropped_features, nor "
            f"engineered_features."
        )

    # Spot-check each type reached its intended handling.
    assert "spend" in kept or "spend" in dropped                  # numeric
    assert any(name.startswith("plan_") for name in kept | dropped)  # one-hot
    assert "account_ref" in dropped                               # high-cardinality
    assert "signed_up" in dropped                                 # raw datetime
    assert "signed_up__month" in engineered                       # decomposed

    # The returned matrix must be entirely numeric — the point of all this.
    assert all(
        pd.api.types.is_numeric_dtype(X[c]) for c in X.columns
    ), "engineer_features returned a non-numeric column; RandomForest would reject it"
    assert list(X.columns) == report.final_feature_names


def test_bool_feature_columns_are_cast_to_numeric_and_kept(classification_spec):
    """Regression test for a gap this suite previously PINNED as broken.

    A bool column is neither object/string/category nor picked up by
    select_dtypes(include="number"), so it used to fall through every branch
    and vanish with no record anywhere — the same silent-drop failure that was
    fixed for text and datetime columns. It is now cast to 0/1 and treated as
    an ordinary numeric feature.

    No one-hot encoding: a bool is ALREADY binary, so expanding it would only
    produce two perfectly anti-correlated columns carrying the same single
    bit. And no engineered_features entry, deliberately — a dtype cast
    produces no new feature, the column keeps its own name, and its fate is
    already traceable through final_feature_names/dropped_features."""
    n = 40
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        # Perfectly aligned with the target, so it carries real signal and
        # survives the low-signal filter on its merits.
        "is_active": [bool(i % 2) for i in range(n)],
        "noise": [float((i * 7) % 11) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert "is_active" in report.final_feature_names
    assert not any(d.name == "is_active" for d in report.dropped_features)
    # Scored by the same correlation pass as every other numeric column.
    assert "is_active" in report.correlation_summary
    # Not one-hot expanded into is_active_True / is_active_False. Checked by
    # exact dummy names rather than a "is_active_" prefix scan, because the
    # interaction term built from the surviving features is legitimately
    # named "is_active__x__<other>" and shares that prefix.
    assert "is_active_True" not in report.final_feature_names
    assert "is_active_False" not in report.final_feature_names
    assert not any(e.name == "is_active__onehot" for e in report.engineered_features)


def test_bool_column_reaches_the_feature_matrix_as_zeros_and_ones(classification_spec):
    """The cast has to be real, not nominal. The returned X must hold numeric
    0/1 values — a bool dtype surviving into the matrix would be exactly the
    thing a RandomForest chokes on, which is what this whole path exists to
    prevent."""
    n = 40
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "is_active": [bool(i % 2) for i in range(n)],
        "noise": [float((i * 7) % 11) for i in range(n)],
    })

    X, _y, _report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert pd.api.types.is_numeric_dtype(X["is_active"])
    assert not pd.api.types.is_bool_dtype(X["is_active"])
    assert set(X["is_active"].unique()) == {0, 1}


def test_constant_bool_column_is_dropped_for_low_variance(classification_spec):
    """Bools get no special protection once cast. An all-True column has zero
    variance and must be dropped by the EXISTING low-variance rule — the same
    rule that drops a constant numeric column or a single-category dummy —
    proving the bool path feeds the shared filter rather than bypassing it."""
    n = 40
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "always_true": [True] * n,
        "signal": [float(i % 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    dropped_entry = next(d for d in report.dropped_features if d.name == "always_true")
    assert "variance" in dropped_entry.rationale.lower()
    assert "always_true" not in report.final_feature_names


def test_all_false_bool_column_is_also_dropped_for_low_variance(classification_spec):
    """The mirror case. All-False casts to all-zeros, which is just as
    constant as all-ones — asserted separately because a naive truthiness
    check somewhere in the cast could treat the two ends differently."""
    n = 40
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "always_false": [False] * n,
        "signal": [float(i % 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    dropped_entry = next(d for d in report.dropped_features if d.name == "always_false")
    assert "variance" in dropped_entry.rationale.lower()
    assert "always_false" not in report.final_feature_names


# --------------------------------------------------------------------------
# Missing datetimes are surfaced, not silently propagated
# --------------------------------------------------------------------------

def test_missing_datetime_produces_a_data_quality_warning(classification_spec):
    """NaT becomes NaN in every derived column, and RandomForest rejects NaN
    outright — so without this warning the run dies several stages later,
    inside sklearn, with an error naming neither this column nor this stage.

    The column is still decomposed as normal (imputing a timestamp would be
    inventing data); what changes is that the problem is stated up front,
    naming the column and what it will break. Assertions check for the column
    name and a NaT/missing mention rather than exact wording, so rephrasing
    the message does not break the test."""
    n = 12
    dates = list(pd.date_range("2023-01-01", periods=n - 1, freq="37D")) + [pd.NaT]
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "signup_date": pd.to_datetime(pd.Series(dates)),
        "signal": [float(i % 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert report.data_quality_warnings, "expected a data-quality warning"
    warning = " ".join(report.data_quality_warnings)
    assert "signup_date" in warning
    assert "nat" in warning.lower() or "missing" in warning.lower()
    assert report.has_data_quality_risk is True

    # Decomposition still happened — the warning informs, it does not skip.
    assert any(e.name == "signup_date__month" for e in report.engineered_features)


def test_clean_datetime_produces_no_data_quality_warning(classification_spec):
    """The negative case, and the one that stops the warning becoming noise.
    A complete datetime column must produce NO warning at all — a check that
    fires on every dataset teaches users to ignore it."""
    n = 30
    df = pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="37D"),
        "signal": [float(i % 2) for i in range(n)],
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert report.data_quality_warnings == []
    assert report.has_data_quality_risk is False


def test_data_quality_warnings_are_kept_separate_from_leakage_warnings(
    classification_spec,
):
    """The two lists answer different questions and must not be conflated:
    leakage is 'can this result be trusted', data quality is 'will this run
    complete at all'. A dataset with both a leaking feature and a missing date
    must report each in its own list, so an agent weighing them is not forced
    to compare a trust problem against a crash on one axis."""
    n = 20
    target = [i % 2 for i in range(n)]
    dates = list(pd.date_range("2023-01-01", periods=n - 1, freq="37D")) + [pd.NaT]
    df = pd.DataFrame({
        "target": target,
        "leaked": [float(t) for t in target],  # near-perfect correlation
        "signup_date": pd.to_datetime(pd.Series(dates)),
    })

    _X, _y, report = engineer_features(df, _minimal_profile(n), classification_spec)

    assert any("leaked" in w for w in report.leakage_warnings)
    assert any("signup_date" in w for w in report.data_quality_warnings)
    # Neither list has absorbed the other's contents.
    assert not any("signup_date" in w for w in report.leakage_warnings)
    assert not any("leaked" in w for w in report.data_quality_warnings)
