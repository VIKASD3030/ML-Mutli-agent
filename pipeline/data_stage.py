from __future__ import annotations

import pandas as pd
from schemas.problem_spec import ProblemSpec
from schemas.data_profile import ColumnSchema, DataProfile, CleaningAction

Missing_blocking_threshold = 0.5


def load_data(spec: ProblemSpec) -> pd.DataFrame:
    """Load raw data from the source named in the ProblemSpec.

    Parameters
    ----------
    spec: ProblemSpec
        We take the whole spec (not loose strings) so every stage function
        in this pipeline shares one consistent signature. If load_data()
        took (data_source: str, target_column: str) while every other
        stage took spec, calling code would need to remember two different
        calling conventions, and adding a new field this function needs
        later (say, a "sheet_name" for Excel sources) would mean changing
        this function's signature — and every call site — instead of just
        reading a new field off the spec that's already being passed around.

    Returns
    -------
    pd.DataFrame
        Raw data. Returned as a plain DataFrame, not a Pydantic model,
        because Pydantic models describe *decisions* about data
        (DataProfile, EDAReport) — they aren't built to hold arbitrary
        tabular data efficiently. The DataFrame itself isn't a hand-off
        decision object, so it doesn't get wrapped.
    """
    if spec.data_source == "builtin:breast_cancer":
        from sklearn.datasets import load_breast_cancer
        raw = load_breast_cancer(as_frame=True)
        df = raw.frame.copy()

        df = df.rename(columns={"target": spec.target_column})
        return df

    raise ValueError(f"Unsupported data source: {spec.data_source!r}")

def clean_and_profile(df:pd.DataFrame, spec: ProblemSpec) -> tuple[pd.DataFrame, DataProfile]:
    """Clean the dataframe and produce a DataProfile describing what was done.

    Parameters
    ----------
    df: pd.DataFrame
        Raw data, as returned by load_data(). Not mutated in place — a
        cleaned copy is returned, matching the .copy() discipline from
        load_data() so callers never have to worry about aliasing.
    spec: ProblemSpec
        Needed for spec.target_column (which column is the label) and
        spec.task_type (whether to compute class_balance).

    Returns
    -------
    tuple[pd.DataFrame, DataProfile]
        The cleaned DataFrame, and a DataProfile describing every action
        taken. Returned as a tuple rather than two separate calls because
        they're produced together — there's no world where you'd want
        the cleaned data without also knowing what was done to it.
    """
    cleaning_actions: list[CleaningAction] = []
    warnings : list[str] = []
    blocking_issue: str | None = None

    # Checks if we have Target column even exists in the DataFrame
    if spec.target_column not in df.columns:
        blocking_issue = f"Target column '{spec.target_column}' not present in data."
        profile = DataProfile(
            row_count = len(df),
            column_count = df.shape[1],
            column_schema = [],
            blocking_issue = blocking_issue,
        )

        return df, profile
    
    # Checls whether the target missingness is Valid or not(should we hault or not?)
    target_missing_pct = df[spec.target_column].isna().mean()
    if target_missing_pct > Missing_blocking_threshold:
        blocking_issue = (
            f"{target_missing_pct:.0%} of target column '{spec.target_column}' is "
            f"missing — exceeds the {Missing_blocking_threshold:.0%} usability threshold"
        )

    # Drops the rows with missing Target as we can't train/evaluate on unlabeled rows, so these get dropped either way.
    before = len(df)
    df = df.dropna(subset = [spec.target_column]).reset_index(drop=True)
    if len(df) < before:
        cleaning_actions.append(
            CleaningAction(
                column = spec.target_column,
                action = "dropped_rows_missing_target",
                rationale = f"Dropped {before - len(df)} rows with missing target; "
                "cannot train or evaluate on unlabeled rows.",
            )
        )
    
    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        cleaning_actions.append(
            CleaningAction(
                column="<all>",
                action= "dropped_duplicate_rows",
                rationale = f"Removed {before - len(df)} exact-duplicate rows.",

            )
        )
    
    column_schema: list[ColumnSchema] = []
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        is_target = col == spec.target_column

        if missing_pct > 0 and not is_target and pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            cleaning_actions.append(
                CleaningAction(
                column = col,
                action = "median_imputation",
                rationale = f"{missing_pct: .1f}% missing: used median ({median_val: .3f})"
                " as a robust default  given no distribution info suggesting otherwise"
            )
        )

        # Record this column's schema info regardless of whether it needed
        # imputation - DataProfile.column_schema should describe
        # every column, not just the ones we touched.
        column_schema.append(
            ColumnSchema(
                name=col,
                dtype=str(df[col].dtype),
                missing_pct=round(missing_pct, 2),
                is_target=is_target,
            )
        )

    # --Class balance(classification only)--
    class_balance = None
    if spec.task_type == "classification":
        vc = df[spec.target_column].value_counts(normalize=True)
        class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}
        minority_share = min(class_balance.values())
        if minority_share < 0.1:
            warnings.append(
                f"Severe class imbalance detected: minority class share = "
                f"{minority_share:.1%}. Consider class weighting or resampling."
            )

    # --Outlier flagging(numeric columns, IQR method) --
    outlier_flags: list[str] = []
    for col in df.select_dtypes(include="number").columns:
        if col == spec.target_column:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 -q1
        if iqr == 0:
            continue
        lower , upper = q1 -3 * iqr, q3 + 3 * iqr
        outlier_share = ((df[col] < lower) | (df[col] > upper)).mean()
        if outlier_share > 0.02:
            outlier_flags.append(col)

    # -- Assemble the final profile --
    profile = DataProfile(
        row_count = len(df),
        column_count = df.shape[1],
        column_schema = column_schema,
        class_balance = class_balance,
        outlier_flags = outlier_flags,
        cleaning_actions_taken = cleaning_actions,
        warnings = warnings,
        blocking_issue = blocking_issue,
    )
    return df, profile