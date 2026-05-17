"""
Per-country Prophet forecasts.

Prophet decomposes a series into trend + yearly + weekly seasonality plus
optional regressors. We add `vaccination_coverage` as an additive regressor
where available.

Per-country (not global) because Prophet:
  - has no native multi-series mode (you'd fit N independent models anyway)
  - is fast enough to fit one per country
  - works on whichever countries have enough history
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.models.evaluate import Split, iter_split_data

log = logging.getLogger(__name__)


def run_prophet(
    features_df: pd.DataFrame,
    splits: list[Split],
    horizon_days: int = 14,
) -> pd.DataFrame:
    """Train + forecast per country across splits."""
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("prophet not installed — skipping Prophet runs")
        return pd.DataFrame()

    all_rows: list[dict] = []
    target = "new_cases_smoothed"

    for split_idx, split, train, test in iter_split_data(features_df, splits):
        log.info("  Split %d: %s", split_idx + 1, split)

        for country, sub in train.groupby("country"):
            sub = sub.dropna(subset=[target]).sort_values("date")
            if len(sub) < 90:
                continue

            hist = sub[["date", target, "vaccination_coverage"]].rename(
                columns={"date": "ds", target: "y"}
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_mode="additive",
                    )
                    if hist["vaccination_coverage"].notna().any():
                        m.add_regressor("vaccination_coverage")
                    m.fit(hist[["ds", "y", "vaccination_coverage"]])

                    test_sub = test[test["country"] == country].sort_values("date")
                    if test_sub.empty:
                        continue
                    future = test_sub[["date", "vaccination_coverage"]].rename(
                        columns={"date": "ds"}
                    )
                    future["vaccination_coverage"] = (
                        future["vaccination_coverage"].fillna(
                            hist["vaccination_coverage"].iloc[-1] if hist["vaccination_coverage"].notna().any() else 0
                        )
                    )
                    forecast = m.predict(future)
                except Exception as e:
                    log.debug("Prophet failed for %s: %s", country, e)
                    continue

            test_sub = test_sub.set_index("date")
            for date, yhat in zip(forecast["ds"], forecast["yhat"]):
                if date not in test_sub.index:
                    continue
                y_true = float(test_sub.loc[date, target])
                if not np.isfinite(y_true):
                    continue
                all_rows.append({
                    "model": "prophet",
                    "split": split_idx,
                    "country": country,
                    "date": date,
                    "y_true": y_true,
                    "y_pred": float(max(yhat, 0)),
                })
    return pd.DataFrame(all_rows)
