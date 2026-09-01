# Phase 1 - deterministic Feature Engineering / EDA Stage. NO LLM here.

from __future__ import annotations

import pandas as pd

from schemas.data_profile import DataProfile
from schemas.eda_report import EDAReport, FeatureDecision
from schemas.problem_spec import ProblemSpec

Leakage_correlation_threshold = 0.98

Low_signal_correlation_threshold = 0.01

Low_variance_threshold = 1e-8

def engineer_features(
    df: pd.DataFrame, profile: DataProfile, spec: ProblemSpec
) -> tuple[pd.DataFrame, pd.Series, EDAReport]:
    """Compute correlations, drop low-signal/near-zero-variance columns,
    flag leakage-risk columns, and return the final feature matrix + target.

    Parameters
    ----------
    df: pd.DataFrame
        The CLEANED dataframe from clean_and_profile() — not raw data.
    profile: DataProfile
        Passed through so this stage can reference cleaning context (e.g.
        class_balance) when writing the EDAReport's notes — it doesn't
        redo any of clean_and_profile()'s work.
    spec: ProblemSpec
        Needed for target_column (to separate X from y) and task_type
        (to decide whether class_balance is relevant to the report).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, EDAReport]
        X (final feature matrix), y (target Series), and the EDAReport
        describing every keep/drop/leakage decision made.
    """
    y = df[spec.target_column]
    X = df.drop(columns=[spec.target_column])

    # Only numeric columns get correlation-based treatment in Phase 1 —
    # categorical feature handling (encoding) is out of scope for now,
    # same "don't generalize early" principle as data_stage.py's CSV support.
    numeric_X = X.select_dtypes(include="number")

    correlation: dict[str, float] = {}
    for col in numeric_X.columns:
        try:
            corr = numeric_X[col].corr(y.astype(float))
        except (TypeError, ValueError):
            corr = 0.0
        # pandas' .corr() can return NaN (e.g. constant column) — coerce
        # to 0.0 so downstream comparisons never have to handle NaN.
        correlation[col] = 0.0 if pd.isna(corr) else round(abs(float(corr)), 4)

    dropped: list[FeatureDecision] = []
    engineered: list[FeatureDecision] = []
    leakage_warnings: list[str] = []
    keep_cols: list[str] = []

    for col in numeric_X.columns:
        variance = numeric_X[col].var()
        corr = correlation.get(col, 0.0)

        if variance is not None and variance < Low_variance_threshold:
            dropped.append(
                FeatureDecision(
                    name=col,
                    rationale=f"Near-zero Variance ({variance:.2e}) - carries no signal.",
                )
            )
            continue  # skip this column entirely - never reaches keep_cols

        if corr >= Leakage_correlation_threshold:
            leakage_warnings.append(
                f"'{col}' correlates {corr:.3f} with target — investigate for leakage "
                "before trusting model performance."
            )
            # Deliberately NOT `continue`-ing here — the column is still
            # kept below. Flagging + keeping, not flagging + dropping, is
            # the asymmetric handling described above.

        if corr < Low_signal_correlation_threshold:
            dropped.append(
                FeatureDecision(
                    name=col,
                    rationale=f"Correlation with target is {corr:.4f} — below the "
                    f"{Low_signal_correlation_threshold} signal threshold.",
                )
            )
            continue

        keep_cols.append(col)

    # Simple demonstration of "engineering": multiply the two most
    # correlated surviving features together as an interaction term.
    ranked = sorted(keep_cols, key=lambda c: correlation[c], reverse=True)
    final_X = numeric_X[keep_cols].copy()
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        interaction_name = f"{a}__x__{b}"
        final_X[interaction_name] = numeric_X[a] * numeric_X[b]
        engineered.append(
            FeatureDecision(
                name=interaction_name,
                rationale=f"Interaction of the two highest-correlation features "
                f"({a}: {correlation[a]:.3f}, {b}: {correlation[b]:.3f}).",
            )
        )

    target_note = (
        f"Target '{spec.target_column}' — task_type={spec.task_type}, "
        f"{len(y)} labeled rows."
    )

    if spec.task_type == "classification" and profile.class_balance:
        target_note += f" Class Balance: {profile.class_balance}"

    report = EDAReport(
        correlation_summary=correlation,
        engineered_features=engineered,
        dropped_features=dropped,
        leakage_warnings=leakage_warnings,
        target_distribution_notes=target_note,
        final_feature_names=list(final_X.columns),
    )
    return final_X, y, report