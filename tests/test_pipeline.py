"""Tests for the pandemic-forecaster pipeline.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.client import CountrySnapshot
from src.data.fetch import _parse_timeline
from src.features.build import build_country_features, SERIAL_INTERVAL_DAYS
from src.models.evaluate import (
    mape, mae, rmse, pinball_loss, interval_coverage,
    rolling_origin_splits, evaluate_predictions,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_country() -> tuple[pd.DataFrame, int]:
    """A single country's cumulative time series with three waves."""
    np.random.seed(42)
    dates = pd.date_range("2020-03-01", "2022-06-30", freq="D")
    n = len(dates)
    new_cases = np.zeros(n)
    for wave_start, peak_day, peak_height in [(0, 60, 1000), (200, 280, 3000), (500, 580, 5000)]:
        for i in range(n):
            if i >= wave_start:
                t = i - peak_day
                new_cases[i] += peak_height * (np.exp(0.1 * t) if t < 0 else np.exp(-0.04 * t))
    new_cases *= np.random.lognormal(0, 0.05, n)
    cumulative = np.cumsum(new_cases).astype(int)
    df = pd.DataFrame({
        "date": dates,
        "cases": cumulative,
        "deaths": (cumulative * 0.01).astype(int),
        "recovered": (cumulative * 0.9).astype(int),
    })
    return df, 100_000_000


@pytest.fixture
def multi_country_features() -> pd.DataFrame:
    """Multi-country panel for end-to-end tests."""
    np.random.seed(7)
    rows = []
    for ci in range(4):
        pop = int(np.random.lognormal(17, 1))
        dates = pd.date_range("2020-03-01", "2022-06-30", freq="D")
        n = len(dates)
        nc = np.zeros(n)
        for ws, pd_, ph in np.random.uniform([0, 60, 1000], [200, 700, 5000], (2, 3)):
            for i in range(n):
                if i >= ws:
                    t = i - pd_
                    nc[i] += ph * (np.exp(0.08 * t) if t < 0 else np.exp(-0.04 * t))
        nc *= np.random.lognormal(0, 0.05, n)
        cum = np.cumsum(nc).astype(int)
        raw = pd.DataFrame({
            "date": dates,
            "cases": cum,
            "deaths": (cum * 0.01).astype(int),
            "recovered": (cum * 0.9).astype(int),
        })
        f = build_country_features(raw, population=pop)
        f["country"] = f"C{ci}"
        f["continent"] = ["Asia", "Europe", "Africa", "Americas"][ci]
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# Data layer tests
# ============================================================================

def test_country_snapshot_from_api_full():
    d = {
        "country": "India",
        "countryInfo": {"iso2": "IN", "iso3": "IND"},
        "continent": "Asia",
        "population": 1_400_000_000,
        "cases": 1_000_000, "deaths": 10_000,
        "recovered": 900_000, "active": 90_000,
    }
    s = CountrySnapshot.from_api(d)
    assert s.country == "India"
    assert s.iso3 == "IND"
    assert s.population == 1_400_000_000


def test_country_snapshot_handles_missing_fields():
    d = {"country": "X", "cases": 1, "deaths": 0, "recovered": 0, "active": 1}
    s = CountrySnapshot.from_api(d)
    assert s.population is None
    assert s.iso2 is None


def test_parse_timeline_basic():
    df = _parse_timeline({"1/22/20": 1, "3/1/20": 100, "6/15/24": 5000})
    assert len(df) == 3
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-22")
    assert df["date"].iloc[-1] == pd.Timestamp("2024-06-15")


def test_parse_timeline_empty():
    assert _parse_timeline({}).empty


# ============================================================================
# Feature engineering tests
# ============================================================================

def test_feature_columns_present(synthetic_country):
    df, pop = synthetic_country
    feats = build_country_features(df, population=pop)
    required = {
        "new_cases", "new_cases_smoothed", "growth_rate_7d",
        "r_effective_approx", "cfr_rolling_28d", "cases_per_million",
        "days_since_outbreak", "vaccination_coverage",
        "new_cases_smoothed_lag7", "new_cases_smoothed_lag14",
    }
    assert required.issubset(set(feats.columns))


def test_growth_rate_signs(synthetic_country):
    df, pop = synthetic_country
    feats = build_country_features(df, population=pop)
    # Around the first peak (day 60), growth rate should be positive before
    # and negative after
    assert feats["growth_rate_7d"].iloc[45] > 0
    assert feats["growth_rate_7d"].iloc[90] < 0


def test_r_effective_above_one_during_growth(synthetic_country):
    df, pop = synthetic_country
    feats = build_country_features(df, population=pop)
    assert feats["r_effective_approx"].iloc[45] > 1
    assert feats["r_effective_approx"].iloc[90] < 1


def test_cfr_bounded(synthetic_country):
    df, pop = synthetic_country
    feats = build_country_features(df, population=pop)
    cfr = feats["cfr_rolling_28d"].dropna()
    assert (cfr >= 0).all()
    assert (cfr <= 1).all()


def test_days_since_outbreak_monotonic(synthetic_country):
    df, pop = synthetic_country
    feats = build_country_features(df, population=pop)
    diffs = feats["days_since_outbreak"].diff().dropna()
    assert (diffs >= 0).all()


def test_negative_new_cases_clipped():
    # Cumulative count that decreases (data revision) — should clip to 0
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "cases": [100, 200, 300, 250, 400, 500, 600, 700, 800, 900],
        # ↑ that drop at index 3 simulates a revision
        "deaths": [1] * 10,
        "recovered": [10] * 10,
    })
    feats = build_country_features(df, population=1_000_000)
    assert (feats["new_cases"].dropna() >= 0).all()


def test_lag_features_correctly_offset(synthetic_country):
    df, pop = synthetic_country
    feats = build_country_features(df, population=pop)
    # lag-7 at index 100 should equal new_cases_smoothed at index 93
    assert feats["new_cases_smoothed_lag7"].iloc[100] == feats["new_cases_smoothed"].iloc[93]


# ============================================================================
# Metrics tests
# ============================================================================

def test_mape_perfect_prediction():
    assert mape([100, 200], [100, 200]) == 0


def test_mape_known_value():
    # 10% absolute pct error each → 10% mean
    assert abs(mape([100, 100], [110, 90]) - 10.0) < 0.001


def test_mae_basic():
    assert mae([100, 200], [110, 180]) == 15


def test_rmse_basic():
    # |10|, |20| → rmse = sqrt((100+400)/2) ≈ 15.81
    assert abs(rmse([100, 200], [110, 180]) - 15.811) < 0.01


def test_pinball_loss_at_truth():
    assert pinball_loss([100, 200], [100, 200], q=0.5) == 0


def test_interval_coverage():
    # 4/5 are inside [80, 220]
    cov = interval_coverage([90, 100, 150, 200, 250], [80] * 5, [220] * 5)
    assert cov == 0.8


# ============================================================================
# Rolling-origin splits
# ============================================================================

def test_rolling_splits_count_and_order():
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    splits = rolling_origin_splits(dates, n_splits=4, horizon_days=14)
    assert len(splits) == 4
    # Chronological
    for i in range(len(splits) - 1):
        assert splits[i].train_end < splits[i + 1].train_end


def test_rolling_splits_no_overlap_between_train_and_test():
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    splits = rolling_origin_splits(dates, n_splits=3, horizon_days=14)
    for s in splits:
        assert s.test_start > s.train_end
        assert s.test_end >= s.test_start


def test_rolling_splits_raises_if_too_short():
    short_dates = pd.date_range("2020-01-01", periods=30, freq="D")
    with pytest.raises(ValueError):
        rolling_origin_splits(short_dates, n_splits=4, horizon_days=14, min_train_days=90)


# ============================================================================
# End-to-end modelling tests
# ============================================================================

def test_lightgbm_runs(multi_country_features):
    from src.models.gbm import run_lightgbm, LightGBMConfig
    splits = rolling_origin_splits(multi_country_features["date"], n_splits=2, horizon_days=14)
    preds = run_lightgbm(multi_country_features, splits, LightGBMConfig(horizon_days=14, n_estimators=50))
    assert not preds.empty
    assert {"model", "country", "date", "y_true", "y_pred", "y_pred_q10", "y_pred_q90"}.issubset(preds.columns)
    assert (preds["y_pred"] >= 0).all()


def test_evaluate_lightgbm_output(multi_country_features):
    from src.models.gbm import run_lightgbm, LightGBMConfig
    splits = rolling_origin_splits(multi_country_features["date"], n_splits=2, horizon_days=14)
    preds = run_lightgbm(multi_country_features, splits, LightGBMConfig(horizon_days=14, n_estimators=50))
    ev = evaluate_predictions(preds, panel=multi_country_features)
    assert ev.overall_mape > 0
    assert ev.overall_mae > 0
    assert len(ev.by_continent) >= 2
    assert ev.interval_coverage_80 is not None


def test_lstm_runs(multi_country_features):
    from src.models.lstm_model import run_lstm, LSTMConfig
    splits = rolling_origin_splits(multi_country_features["date"], n_splits=2, horizon_days=14)
    preds = run_lstm(
        multi_country_features, splits,
        LSTMConfig(horizon_days=14, n_epochs=2, batch_size=64),
    )
    assert not preds.empty
    assert (preds["y_pred"] >= 0).all()
