"""
Global LightGBM forecaster with quantile regression for uncertainty.

Why a single global model across all countries?
- ML benefits from cross-country knowledge transfer (similar outbreak shapes).
- Per-country models are starved on countries with short histories.
- Country identity is captured through static covariates + lag features.

Why quantile regression?
- Point predictions tell you what to expect; quantile predictions tell you how
  uncertain you are. For pandemic forecasting the uncertainty matters as much
  as the central forecast — health systems need to plan for the upper tail.
- We fit three separate models for q=0.1, 0.5, 0.9. The 80% prediction interval
  is [q10, q90].

What we predict
---------------
Target: `new_cases_smoothed`, the 7-day rolling mean of daily new cases.
We predict H days ahead, where H is supplied at runtime (default 14).

How we structure the training data
----------------------------------
At training time, for each (country, date) in the training set:
  - Features = values known up to `date`
  - Target   = value at `date + H` (so we're predicting the future)

This means a single trained model handles all horizons up to H by varying
the lag features used at inference time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.evaluate import Split, iter_split_data

log = logging.getLogger(__name__)


FEATURES: list[str] = [
    # Lag features
    "new_cases_smoothed_lag1",
    "new_cases_smoothed_lag3",
    "new_cases_smoothed_lag7",
    "new_cases_smoothed_lag14",
    "new_cases_smoothed_lag21",
    "new_cases_smoothed_lag28",
    # Epi state
    "growth_rate_7d",
    "r_effective_approx",
    "cfr_rolling_28d",
    "cases_per_million",
    "days_since_outbreak",
    "vaccination_coverage",
    # Static
    "population",
    # Calendar
    "dow",
    "month",
    "is_weekend",
]

TARGET_COL = "new_cases_smoothed"


@dataclass
class LightGBMConfig:
    horizon_days: int = 14
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20


def _build_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create the shifted target column: y(t+H) for each row.

    Rows where the target is NaN (last H days of each country) are dropped
    at training time; for the test set we keep them aligned with the actual
    target dates."""
    df = df.sort_values(["country", "date"]).copy()
    df["y_target"] = df.groupby("country")[TARGET_COL].shift(-horizon)
    return df


def run_lightgbm(
    features_df: pd.DataFrame,
    splits: list[Split],
    config: LightGBMConfig | None = None,
) -> pd.DataFrame:
    """Train + predict across rolling-origin splits.

    Returns
    -------
    DataFrame with columns:
        model, split, country, date, y_true, y_pred, y_pred_q10, y_pred_q90
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("lightgbm not installed: pip install lightgbm") from e

    cfg = config or LightGBMConfig()
    df = _build_target(features_df, cfg.horizon_days)
    # Drop rows where features are entirely NaN (very early in each country's
    # series, before lag features are filled)
    df = df.dropna(subset=FEATURES, how="all").copy()

    log.info(
        "LightGBM: %d obs, %d countries, %d features",
        len(df), df["country"].nunique(), len(FEATURES),
    )

    all_rows: list[dict] = []

    for split_idx, split, train, test in iter_split_data(df, splits):
        log.info("  Split %d: %s", split_idx + 1, split)

        # The "current row" features are at the date the forecast is MADE.
        # The target (y_target) is the value H days later. So we train on
        # rows whose target is observed (i.e. the H-days-later date is also
        # within the training window).
        train_for_fit = train[train["y_target"].notna()].copy()
        if len(train_for_fit) < 100:
            log.warning("    Too few training rows (%d), skipping", len(train_for_fit))
            continue

        # For prediction, we want to forecast targets in the test window. The
        # "forecast made on date d predicts date d+H". We need the rows where
        # date + H is within the test window — those are in the *train* set
        # with d in [test_start - H, test_end - H].
        forecast_window_start = split.test_start - pd.Timedelta(days=cfg.horizon_days)
        forecast_window_end = split.test_end - pd.Timedelta(days=cfg.horizon_days)
        predict_from = train[
            (train["date"] >= forecast_window_start)
            & (train["date"] <= forecast_window_end)
        ].copy()
        if predict_from.empty:
            continue

        X_train = train_for_fit[FEATURES].astype(float).fillna(0)
        y_train = train_for_fit["y_target"].astype(float).values

        X_predict = predict_from[FEATURES].astype(float).fillna(0)

        # Fit one model per quantile
        preds_per_q: dict[float, np.ndarray] = {}
        for q in cfg.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_child_samples,
                verbosity=-1,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds_per_q[q] = model.predict(X_predict)

        # Assemble predictions, mapping back to the *target* date (forecast_date + H)
        for i, (_, row) in enumerate(predict_from.iterrows()):
            target_date = row["date"] + pd.Timedelta(days=cfg.horizon_days)
            # Find the true value at target_date (might be NaN at the edges)
            mask = (df["country"] == row["country"]) & (df["date"] == target_date)
            true_match = df.loc[mask, TARGET_COL]
            if true_match.empty:
                continue
            y_true = float(true_match.iloc[0])
            if not np.isfinite(y_true):
                continue
            all_rows.append({
                "model": "lightgbm",
                "split": split_idx,
                "country": row["country"],
                "date": target_date,
                "y_true": y_true,
                "y_pred": float(preds_per_q[0.5][i]),
                "y_pred_q10": float(preds_per_q[0.1][i]),
                "y_pred_q90": float(preds_per_q[0.9][i]),
            })

    return pd.DataFrame(all_rows)
