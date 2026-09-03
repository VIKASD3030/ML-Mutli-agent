# Phase 1 - deterministic Feature Engineering / EDA Stage. NO LLM here.

from __future__ import annotations

import pandas as pd

from pipeline.data_stage import is_categorical_column, is_datetime_column
from schemas.data_profile import DataProfile
from schemas.eda_report import EDAReport, FeatureDecision
from schemas.problem_spec import ProblemSpec

Leakage_correlation_threshold = 0.98

Low_signal_correlation_threshold = 0.01

Low_variance_threshold = 1e-8

# Above this many distinct values, one-hot encoding stops being useful and
# starts being a dimensionality problem: a 500-value ID column would add 500
# almost-all-zero features, each individually uninformative, and swamp the
# genuinely predictive columns. Such columns are dropped outright for now —
# target/hash encoding is deliberately future scope, not silently improvised
# here.
MAX_CATEGORICAL_CARDINALITY = 20

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

    dropped: list[FeatureDecision] = []
    engineered: list[FeatureDecision] = []
    leakage_warnings: list[str] = []
    data_quality_warnings: list[str] = []
    keep_cols: list[str] = []

    # --- Convert non-numeric columns into numeric candidates -----------------
    #
    # Categorical and datetime columns used to be discarded here in silence:
    # select_dtypes(include="number") simply did not see them, so nothing in
    # the EDAReport recorded that they had ever existed. Every column now
    # leaves a trace — as surviving features, as engineered_features, or as
    # dropped_features with a reason.
    #
    # Type detection reads the DATAFRAME, not profile.column_schema. The
    # schema's dtype string would technically be enough to branch on, but the
    # real dtypes are right here and are the ground truth the encoding
    # actually operates on; going through a stringified copy would add a way
    # for the two to disagree without adding any information. No schema change
    # was needed for this feature.
    #
    # Everything produced below is plain numeric and is then fed through the
    # SAME correlation/variance/leakage filter as the original numeric
    # columns — there is no parallel filtering path.
    numeric_X = X.select_dtypes(include="number")
    candidate_frames: list[pd.DataFrame] = [numeric_X]

    for col in X.columns:
        series = X[col]

        # Bools are checked FIRST. They are already binary, so one-hot
        # encoding would only produce two perfectly anti-correlated columns
        # carrying the same single bit — strictly worse than the original.
        # A plain 0/1 cast is all they need, after which they are ordinary
        # numeric columns and go through the same filter as everything else.
        # (They need handling at all because they fall through BOTH existing
        # paths: select_dtypes(include="number") excludes bool, and bool is
        # not object/string/category either, so they used to vanish silently.)
        if pd.api.types.is_bool_dtype(series):
            candidate_frames.append(series.astype(int).to_frame(name=col))
            continue

        if is_datetime_column(series):
            # Missing dates are surfaced, not silently repaired. NaT becomes
            # NaN in every derived column below, and RandomForest rejects NaN
            # outright — so without this the run dies several stages later
            # inside sklearn, with an error naming neither this column nor
            # this stage. Imputing a timestamp instead would be inventing
            # data, so the warning is the honest option: decompose as normal,
            # and say plainly what is wrong and what it will break.
            missing_count = int(series.isna().sum())

            # A RandomForest cannot consume datetime64, so the column is
            # decomposed rather than encoded. These four parts are where the
            # predictive signal in a timestamp usually lives (seasonality,
            # weekly rhythm) without exploding into one feature per date.
            derived = pd.DataFrame(
                {
                    f"{col}__year": series.dt.year,
                    f"{col}__month": series.dt.month,
                    f"{col}__day_of_week": series.dt.dayofweek,
                    f"{col}__is_weekend": (series.dt.dayofweek >= 5),
                },
                index=X.index,
            ).astype(float)

            if missing_count:
                data_quality_warnings.append(
                    f"'{col}' contains {missing_count} missing datetime value(s) "
                    f"(NaT). The derived features {sorted(derived.columns)} will "
                    "therefore contain missing values, which a RandomForest "
                    "cannot consume — model training will fail unless these rows "
                    "are filled or dropped first."
                )

            candidate_frames.append(derived)
            for derived_name in derived.columns:
                engineered.append(
                    FeatureDecision(
                        name=derived_name,
                        rationale=f"Extracted from datetime column '{col}'.",
                    )
                )
            dropped.append(
                FeatureDecision(
                    name=col,
                    rationale=f"Raw datetime column decomposed into "
                    f"{sorted(derived.columns)}; a RandomForest cannot consume a "
                    "datetime64 dtype directly.",
                )
            )
            continue

        if is_categorical_column(series):
            unique_count = int(series.nunique(dropna=True))

            if unique_count > MAX_CATEGORICAL_CARDINALITY:
                dropped.append(
                    FeatureDecision(
                        name=col,
                        rationale=f"High-cardinality categorical column "
                        f"({unique_count} unique values, above the "
                        f"{MAX_CATEGORICAL_CARDINALITY} limit) — dropped to avoid a "
                        "dimensionality explosion from one-hot encoding.",
                    )
                )
                continue

            # dtype=float, not the pandas-2 default of bool: the filtering
            # below calls .var() and .corr(), and a bool column would need
            # coercing at every one of those call sites instead of once here.
            dummies = pd.get_dummies(series, prefix=col, dtype=float)
            candidate_frames.append(dummies)
            engineered.append(
                FeatureDecision(
                    name=f"{col}__onehot",
                    rationale=f"Categorical column '{col}' ({unique_count} unique "
                    f"values) one-hot expanded into {len(dummies.columns)} binary "
                    f"columns: {sorted(dummies.columns)}.",
                )
            )
            continue

    # One frame of purely numeric candidates: original numerics, one-hot
    # dummies, and datetime parts, all treated identically from here on.
    candidate_X = pd.concat(candidate_frames, axis=1)

    correlation: dict[str, float] = {}
    for col in candidate_X.columns:
        try:
            corr = candidate_X[col].corr(y.astype(float))
        except (TypeError, ValueError):
            corr = 0.0
        # pandas' .corr() can return NaN (e.g. constant column) — coerce
        # to 0.0 so downstream comparisons never have to handle NaN.
        correlation[col] = 0.0 if pd.isna(corr) else round(abs(float(corr)), 4)

    for col in candidate_X.columns:
        variance = candidate_X[col].var()
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
    final_X = candidate_X[keep_cols].copy()
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        interaction_name = f"{a}__x__{b}"
        final_X[interaction_name] = candidate_X[a] * candidate_X[b]
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
        data_quality_warnings=data_quality_warnings,
        target_distribution_notes=target_note,
        final_feature_names=list(final_X.columns),
    )
    return final_X, y, report