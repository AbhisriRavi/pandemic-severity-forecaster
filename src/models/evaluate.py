"""
Shared evaluation utilities: metrics + rolling-origin cross-validation.

These are used by every model script so we evaluate apples-to-apples.

Time-series cross-validation
============================
NEVER use random train/test splits on time series — that leaks future
information into training. We use rolling-origin (also called expanding-window
or forward-chaining) CV:

    Split 1: train [day 0   → day T-3h], test [day T-3h+1 → day T-2h]
    Split 2: train [day 0   → day T-2h], test [day T-2h+1 → day T-h]
    Split 3: train [day 0   → day T-h],   test [day T-h+1   → day T]

where h is the forecast horizon and T is the most recent date.

Subgroup analysis
=================
After computing overall metrics, we re-evaluate separately by country income
group (World Bank classification) and continent — to surface whether the
model generalises equally well across groups.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error, with a denominator floor at 1.

    The floor prevents divide-by-zero when true value is 0 (which happens at
    the start/end of an outbreak), at the cost of slight pessimism on near-zero
    values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q: float,
) -> float:
    """Quantile (pinball) loss. Measures calibration of a single quantile q."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
) -> float:
    """Fraction of true values that fell inside the [lower, upper] interval.

    For a well-calibrated 80% interval (q10–q90), this should be ~0.80."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(y_lower)) & (y_true <= np.asarray(y_upper))))


# -----------------------------------------------------------------------------
# Rolling-origin splits
# -----------------------------------------------------------------------------

@dataclass
class Split:
    """One train/test split."""
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Split(train≤{self.train_end.date()}, "
            f"test {self.test_start.date()}→{self.test_end.date()})"
        )


def rolling_origin_splits(
    dates: pd.Series | pd.DatetimeIndex,
    n_splits: int = 4,
    horizon_days: int = 14,
    min_train_days: int = 90,
) -> list[Split]:
    """Build chronologically-ordered expanding-window CV splits."""
    sorted_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))
    n = len(sorted_dates)
    if n < min_train_days + horizon_days:
        raise ValueError(
            f"Not enough dates ({n}) for min_train={min_train_days} + "
            f"horizon={horizon_days}"
        )
    splits: list[Split] = []
    for i in range(n_splits):
        test_end_idx = n - 1 - i * horizon_days
        test_start_idx = test_end_idx - horizon_days + 1
        train_end_idx = test_start_idx - 1
        if train_end_idx < min_train_days - 1:
            break
        splits.append(Split(
            train_end=sorted_dates[train_end_idx],
            test_start=sorted_dates[test_start_idx],
            test_end=sorted_dates[test_end_idx],
        ))
    return list(reversed(splits))  # chronological order


def iter_split_data(
    df: pd.DataFrame,
    splits: list[Split],
    date_col: str = "date",
) -> Iterator[tuple[int, Split, pd.DataFrame, pd.DataFrame]]:
    """Yield (split_index, split, train_df, test_df)."""
    for i, s in enumerate(splits):
        train = df[df[date_col] <= s.train_end]
        test = df[(df[date_col] >= s.test_start) & (df[date_col] <= s.test_end)]
        yield i, s, train, test


# -----------------------------------------------------------------------------
# Evaluation result containers
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Standardised evaluation result, JSON-serialisable."""
    model: str
    n_predictions: int
    overall_mape: float
    overall_mae: float
    overall_rmse: float
    by_horizon_day: dict = field(default_factory=dict)
    by_continent: dict = field(default_factory=dict)
    pinball_q10: float | None = None
    pinball_q90: float | None = None
    interval_coverage_80: float | None = None


def evaluate_predictions(
    predictions: pd.DataFrame,
    panel: pd.DataFrame | None = None,
) -> EvalResult:
    """Compute headline + subgroup metrics.

    `predictions` must have columns: model, country, date, y_true, y_pred,
    and optionally y_pred_q10, y_pred_q90 for interval metrics.

    If `panel` is provided with a 'continent' column, we add a continent-level
    breakdown.
    """
    df = predictions.copy()
    model_name = df["model"].iloc[0]

    res = EvalResult(
        model=model_name,
        n_predictions=len(df),
        overall_mape=mape(df["y_true"], df["y_pred"]),
        overall_mae=mae(df["y_true"], df["y_pred"]),
        overall_rmse=rmse(df["y_true"], df["y_pred"]),
    )

    # Breakdown by step-ahead horizon (1, 2, ..., H days)
    if "split" in df.columns:
        df = df.sort_values(["split", "country", "date"])
        df["horizon_day"] = df.groupby(["split", "country"]).cumcount() + 1
        for h, grp in df.groupby("horizon_day"):
            res.by_horizon_day[int(h)] = {
                "mape": mape(grp["y_true"], grp["y_pred"]),
                "mae": mae(grp["y_true"], grp["y_pred"]),
                "n": len(grp),
            }

    # Breakdown by continent
    if panel is not None and "continent" in panel.columns:
        joined = df.merge(
            panel[["country", "continent"]].drop_duplicates(),
            on="country",
            how="left",
        )
        for continent, grp in joined.groupby("continent"):
            if len(grp) < 20:  # skip tiny continents (Australia/Oceania often)
                continue
            res.by_continent[str(continent)] = {
                "mape": mape(grp["y_true"], grp["y_pred"]),
                "mae": mae(grp["y_true"], grp["y_pred"]),
                "n": len(grp),
            }

    # Interval metrics (if available)
    if "y_pred_q10" in df.columns and "y_pred_q90" in df.columns:
        res.pinball_q10 = pinball_loss(df["y_true"], df["y_pred_q10"], 0.1)
        res.pinball_q90 = pinball_loss(df["y_true"], df["y_pred_q90"], 0.9)
        res.interval_coverage_80 = interval_coverage(
            df["y_true"], df["y_pred_q10"], df["y_pred_q90"]
        )

    return res
