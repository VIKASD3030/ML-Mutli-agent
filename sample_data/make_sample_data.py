"""
sample_data/make_sample_data.py

Generates datasets that deliberately exercise every branch of the pipeline.
Regenerate with:  .venv/Scripts/python.exe sample_data/make_sample_data.py

Why these specific columns: each one targets a code path that is otherwise
never hit by a "normal" dataset, so a single upload smoke-tests the whole
feature stage rather than just the happy path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
N = 400
rng = np.random.default_rng(42)


def build(n: int = N) -> pd.DataFrame:
    # Target FIRST, so every other column can be built relative to it.
    # NUMERIC 0/1, not "yes"/"no" — feature_stage.py computes correlation via
    # y.astype(float); a string target makes that raise, every correlation
    # falls back to 0.0, and EVERY feature is dropped as low-signal, leaving
    # a zero-column matrix that training cannot fit.
    churned = rng.integers(0, 2, n)

    return pd.DataFrame({
        "churned": churned,

        # --- ordinary numeric: real signal, should survive ---
        "monthly_spend": churned * 40 + rng.normal(100, 15, n),
        "support_tickets": churned * 3 + rng.poisson(2, n),

        # --- numeric with missing values -> median_imputation ---
        "account_age_days": np.where(
            rng.random(n) < 0.15, np.nan, rng.integers(30, 2000, n)
        ),

        # --- low-signal numeric -> dropped below the 0.01 correlation floor ---
        "random_noise": rng.normal(0, 1, n),

        # --- zero variance -> dropped by the low-variance rule ---
        "constant_flag": np.ones(n),

        # --- bool -> cast to 0/1 and filtered like any numeric column ---
        "is_premium": (churned == 1) & (rng.random(n) < 0.8),

        # --- low-cardinality categorical -> one-hot encoded (3 uniques) ---
        "plan_tier": rng.choice(["basic", "pro", "enterprise"], n),

        # --- categorical WITH missing values -> mode_imputation ---
        "region": np.where(
            rng.random(n) < 0.12, None, rng.choice(["north", "south", "east", "west"], n)
        ),

        # --- high-cardinality categorical -> dropped to avoid a
        #     dimensionality explosion (400 uniques, well over the limit of 20) ---
        "customer_ref": [f"CUST-{i:05d}" for i in range(n)],

        # --- datetime -> decomposed into year/month/day_of_week/is_weekend.
        #     NOTE: only survives as a real datetime64 in PARQUET. A CSV
        #     round-trip turns this back into a string, after which it is
        #     treated as a high-cardinality categorical and dropped. ---
        "signed_up": pd.to_datetime("2021-01-01")
        + pd.to_timedelta(rng.integers(0, 1200, n), unit="D"),
    })


def build_leaky(n: int = N) -> pd.DataFrame:
    """Same shape, plus a column that is essentially a copy of the target.
    Exercises leakage_warnings, which is the single most important signal
    FeatureAgent is prompted to reason about — and which a clean dataset
    will never trigger."""
    df = build(n)
    # ~0.99 correlation with the target: flagged as leakage but deliberately
    # NOT dropped (flag-and-keep is the documented asymmetric behaviour).
    noise = rng.random(n) < 0.005
    df["cancellation_logged"] = np.where(noise, 1 - df["churned"], df["churned"])
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)

    clean = build()
    clean.to_csv(OUT / "customers.csv", index=False)
    clean.to_parquet(OUT / "customers.parquet", index=False)

    leaky = build_leaky()
    leaky.to_csv(OUT / "customers_leaky.csv", index=False)

    for name in ("customers.csv", "customers.parquet", "customers_leaky.csv"):
        p = OUT / name
        print(f"  {name:26} {p.stat().st_size / 1024:7.1f} KB")
    print(f"\n  rows={len(clean)}  target='churned'  balance={clean.churned.mean():.2%} positive")
