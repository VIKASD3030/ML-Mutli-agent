from __future__ import annotations

import pandas as pd
from schemas.problem_spec import ProblemSpec
from schemas.data_profile import ColumnSchema, DataProfile, CleaningAction

Missing_blocking_threshold = 0.5


# --- Column-type predicates -------------------------------------------------
# These live here, not in feature_stage.py, because this is already the module
# that owns dtype-based policy (it is what decides median-vs-mode imputation).
# feature_stage.py imports them rather than re-deriving the same rules, so the
# two stages can never disagree about what counts as "categorical" — a drift
# that would silently mean a column gets mode-imputed here and then handled as
# something else there. data_stage does not import feature_stage, so there is
# no cycle.

def is_datetime_column(series: pd.Series) -> bool:
    """True only for genuinely datetime64-typed columns.

    Deliberately does NOT try to sniff date-like strings out of object
    columns. That is a much fuzzier problem with real false-positive risk —
    a zero-padded numeric ID or a version string can parse as a date — and
    misclassifying an identifier as a timestamp would silently produce
    meaningless year/month features. Only columns something upstream already
    typed as datetimes get datetime treatment.
    """
    return bool(pd.api.types.is_datetime64_any_dtype(series))


def is_categorical_column(series: pd.Series) -> bool:
    """True for object/string/category columns that are not datetimes.

    The datetime check comes FIRST and short-circuits: a column can be
    datetime-typed while still failing a naive "is it numeric?" test, and
    treating it as categorical would one-hot encode every distinct timestamp.
    Datetime handling wins wherever the two could both apply.

    StringDtype is included alongside object, and that is load-bearing rather
    than belt-and-braces. As of pandas 3.0 a plain column of text is inferred
    as StringDtype ("str"), NOT object — pd.api.types.is_object_dtype() returns
    False for it. Checking object alone would therefore have matched almost
    nothing on this project's pinned pandas (3.0.5): every ordinary text column
    would have kept sailing past both this and the numeric branch, and the
    categorical handling would have been dead code that looked correct. object
    is still checked because genuinely mixed-type columns remain object-dtyped.
    """
    if is_datetime_column(series):
        return False
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype) or isinstance(dtype, pd.StringDtype):
        return True
    return bool(pd.api.types.is_object_dtype(series))


def load_data(spec: ProblemSpec) -> pd.DataFrame:
    """Load raw data from the source named in the ProblemSpec.

    ... (existing docstring for spec/return unchanged) ...
    """
    if spec.data_source == "builtin:breast_cancer":
        from sklearn.datasets import load_breast_cancer

        raw = load_breast_cancer(as_frame=True)
        df = raw.frame.copy()
        df = df.rename(columns={"target": spec.target_column})
        return df

    # NEW — real CSV/parquet support. No renaming here: RequirementAgent's
    # target_column is already grounded against this exact file's real
    # column names (via peek_dataset_tool), so the column is already
    # correctly named on disk — unlike the builtin dataset's generic
    # "target" column, which always needs remapping.
    if spec.data_source.endswith(".csv"):
        try:
            return pd.read_csv(spec.data_source)
        except FileNotFoundError:
            # Re-raised as ValueError, not left as FileNotFoundError —
            # matches the "unsupported source" failure below as ONE
            # consistent exception type callers of this function need to
            # handle, rather than two different ones for two different
            # loading failures.
            raise ValueError(f"CSV file not found: {spec.data_source!r}") from None
        except Exception as e:
            # Any other read failure (malformed CSV, encoding issue,
            # etc.) — surfaced with the real file path and underlying
            # error rather than a bare traceback the caller has to dig
            # into. Same "fail loud, with context" discipline as every
            # other explicit error in this file.
            raise ValueError(f"Failed to read CSV {spec.data_source!r}: {e}") from e

    if spec.data_source.endswith(".parquet"):
        try:
            return pd.read_parquet(spec.data_source)
        except FileNotFoundError:
            raise ValueError(f"Parquet file not found: {spec.data_source!r}") from None
        except Exception as e:
            raise ValueError(f"Failed to read parquet {spec.data_source!r}: {e}") from e

    raise ValueError(f"Unsupported data_source: {spec.data_source!r}")

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

        # Categorical counterpart to the median branch above. A median is
        # undefined for non-numeric data, so the most frequent value stands in
        # as the equivalent "unremarkable default" — same intent, same
        # CleaningAction shape, different statistic.
        #
        # Datetime columns fall through both branches on purpose: they are
        # left untouched here and decomposed into numeric parts in
        # engineer_features() instead, so imputing them at this stage would
        # be inventing a timestamp nobody asked for.
        elif missing_pct > 0 and not is_target and is_categorical_column(df[col]):
            mode_values = df[col].mode(dropna=True)
            # .mode() returns an EMPTY series when every value is NaN — there
            # is no most-frequent value to impute with, so leave the column
            # alone rather than inventing one. Recording no action is honest:
            # nothing was done.
            if not mode_values.empty:
                mode_val = mode_values.iloc[0]
                df[col] = df[col].fillna(mode_val)
                cleaning_actions.append(
                    CleaningAction(
                        column=col,
                        action="mode_imputation",
                        rationale=f"{missing_pct: .1f}% missing: used mode ('{mode_val}')"
                        " as the most frequent category, since a median is undefined"
                        " for non-numeric data",
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