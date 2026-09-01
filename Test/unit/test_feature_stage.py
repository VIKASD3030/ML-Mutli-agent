"""
tests/unit/test_feature_stage.py

No mocking — engineer_features() is pure pandas/numpy, same "test the
real deterministic function directly" philosophy as test_data_stage.py.
"""

import pandas as pd
import pytest

from pipeline.feature_stage import engineer_features
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