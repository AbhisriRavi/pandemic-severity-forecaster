"""
Build modelling-ready features from disease.sh raw data.

Epidemiological background — what we compute and why
====================================================

The raw historical endpoint gives us *cumulative* case/death/recovery counts.
For forecasting we want *new cases per day* plus features that capture the
epidemiological state of the outbreak. The features below are designed to be
interpretable to a public-health audience and well-suited to ML models:

1. **new_cases_smoothed (target)**: 7-day rolling mean of daily new cases.
   Smoothing removes weekend/holiday reporting effects that obscure the
   underlying epidemic trend.

2. **growth_rate_7d**: log(cases_today / cases_7_days_ago) / 7. A positive
   value means the outbreak is growing exponentially; ~0 means plateaued;
   negative means declining. This is far more stable than raw percent change.

3. **r_effective_approx**: a simplified approximation of the effective
   reproduction number using the ratio of new cases over consecutive
   serial-interval windows. We use a fixed serial interval τ = 5 days
   (well-established for COVID-19 from Du et al. 2020, EID).

        R_eff(t) ≈ N(t) / N(t - τ)

   This is NOT the formal Wallinga-Teunis or Cori estimate (which need
   proper Bayesian inference over the serial-interval distribution); it's
   a useful proxy feature that captures the same signal.

4. **cfr_rolling_28d**: case fatality rate — deaths / cases, with a 28-day
   lag on cases to better reflect when those cases were actually diagnosed.
   Captures outbreak severity, healthcare strain, and demographics.

5. **cases_per_million**: scales raw counts by population for cross-country
   comparison.

6. **days_since_outbreak**: count of days since the country first crossed
   100 cumulative cases (a common outbreak-onset definition).

7. **vaccination_coverage**: cumulative doses / population (proxy; doesn't
   account for booster vs first dose).

8. **Static covariates** (per country): population, population density
   approximation (cases per million as a weak proxy when density data
   absent), continent.

Output schema
=============
data/processed/features.parquet — one row per (country, date) with:
    country (str), date (datetime),
    new_cases (int), new_cases_smoothed (float),    ← target & raw signal
    new_deaths (int), new_deaths_smoothed (float),
    growth_rate_7d (float), r_effective_approx (float),
    cfr_rolling_28d (float), cases_per_million (float),
    days_since_outbreak (int), vaccination_coverage (float),
    population (int), continent (str),
    # Lag features for ML
    new_cases_smoothed_lag1, _lag3, _lag7, _lag14, _lag21, _lag28,
    # Calendar
    dow (int), month (int), is_weekend (int)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Epidemiological constants
SERIAL_INTERVAL_DAYS = 5         # COVID-19 mean serial interval
OUTBREAK_THRESHOLD_CASES = 100   # threshold for "outbreak started"
DEATH_LAG_DAYS = 28              # typical case-to-death lag for CFR calculation


# -----------------------------------------------------------------------------
# Per-country feature computation
# -----------------------------------------------------------------------------

def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """log(a / b) where both sides are non-negative, NaN-safe.

    Returns NaN where either side is 0 or NaN (log of zero is undefined and
    would propagate as -inf, polluting downstream features)."""
    mask = (a > 0) & (b > 0)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out.loc[mask] = np.log(a.loc[mask] / b.loc[mask])
    return out


def _diff_clip0(series: pd.Series) -> pd.Series:
    """Daily diff with negative values clipped to 0.

    Cumulative case counts occasionally decrease (revisions, data corrections).
    A negative 'new cases' is meaningless for epidemiology and breaks log-based
    features, so we floor to zero.
    """
    return series.diff().clip(lower=0)


def build_country_features(
    df: pd.DataFrame,
    population: int | float | None = None,
    vaccines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build features for a single country.

    Parameters
    ----------
    df : DataFrame with columns [date, cases, deaths, recovered]
         (cumulative counts, sorted ascending by date)
    population : int, optional
        Country population for per-capita features.
    vaccines : DataFrame [date, cumulative_doses], optional
        Vaccine timeline; merged on date.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Daily new counts (clip negative revisions to 0)
    df["new_cases"] = _diff_clip0(df["cases"])
    df["new_deaths"] = _diff_clip0(df["deaths"])

    # 7-day smoothed series (centred=False so we never use future data)
    df["new_cases_smoothed"] = df["new_cases"].rolling(7, min_periods=3).mean()
    df["new_deaths_smoothed"] = df["new_deaths"].rolling(7, min_periods=3).mean()

    # Growth rate: log-ratio over 7 days, then divided by 7 to get per-day rate
    df["growth_rate_7d"] = _safe_log_ratio(
        df["new_cases_smoothed"],
        df["new_cases_smoothed"].shift(7),
    ) / 7

    # R-effective approximation (simplified Wallinga-Teunis-style)
    df["r_effective_approx"] = (
        df["new_cases_smoothed"] / df["new_cases_smoothed"].shift(SERIAL_INTERVAL_DAYS)
    ).replace([np.inf, -np.inf], np.nan)

    # Rolling CFR with 28-day lag on cases (deaths today are from cases ~28 days ago)
    deaths_28d = df["new_deaths"].rolling(28, min_periods=7).sum()
    cases_lagged_28d = df["new_cases"].shift(DEATH_LAG_DAYS).rolling(28, min_periods=7).sum()
    df["cfr_rolling_28d"] = (deaths_28d / cases_lagged_28d).replace(
        [np.inf, -np.inf], np.nan
    )
    df["cfr_rolling_28d"] = df["cfr_rolling_28d"].clip(0, 1)  # CFR must be in [0, 1]

    # Per-million features (where population is known)
    if population and population > 0:
        df["cases_per_million"] = df["new_cases_smoothed"] / population * 1e6
        df["population"] = int(population)
    else:
        df["cases_per_million"] = np.nan
        df["population"] = pd.NA

    # Days since outbreak start (cumulative cases > threshold)
    above = df["cases"] >= OUTBREAK_THRESHOLD_CASES
    if above.any():
        outbreak_start = df.loc[above, "date"].iloc[0]
        df["days_since_outbreak"] = (df["date"] - outbreak_start).dt.days
        df.loc[df["days_since_outbreak"] < 0, "days_since_outbreak"] = 0
    else:
        df["days_since_outbreak"] = 0

    # Vaccination coverage
    df["vaccination_coverage"] = 0.0
    if vaccines is not None and not vaccines.empty and population and population > 0:
        vmerged = df[["date"]].merge(
            vaccines[["date", "cumulative_doses"]],
            on="date",
            how="left",
        )
        # Forward-fill — once doses are reported, they don't disappear
        vmerged["cumulative_doses"] = vmerged["cumulative_doses"].ffill().fillna(0)
        df["vaccination_coverage"] = (vmerged["cumulative_doses"] / population).clip(0, None)

    # Calendar
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Lag features for ML — built on the smoothed target so they capture trend
    for lag in (1, 3, 7, 14, 21, 28):
        df[f"new_cases_smoothed_lag{lag}"] = df["new_cases_smoothed"].shift(lag)

    return df


# -----------------------------------------------------------------------------
# Batch processing across countries
# -----------------------------------------------------------------------------

def build_all_features(
    historical: pd.DataFrame,
    snapshots: pd.DataFrame,
    vaccines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply per-country feature build to every country, concat, return."""
    out = []
    pop_lookup = snapshots.set_index("country")["population"].to_dict()
    cont_lookup = snapshots.set_index("country")["continent"].to_dict()

    countries = historical["country"].unique()
    log.info("Building features for %d countries", len(countries))

    for country in countries:
        sub = historical[historical["country"] == country].copy()
        if len(sub) < 30:
            log.debug("Skipping %s: only %d days of data", country, len(sub))
            continue

        v_sub = None
        if vaccines is not None and not vaccines.empty:
            v_sub = vaccines[vaccines["country"] == country]

        feats = build_country_features(
            sub[["date", "cases", "deaths", "recovered"]],
            population=pop_lookup.get(country),
            vaccines=v_sub,
        )
        feats["country"] = country
        feats["continent"] = cont_lookup.get(country)
        out.append(feats)

    if not out:
        return pd.DataFrame()

    combined = pd.concat(out, ignore_index=True)
    log.info(
        "Combined features: %d rows × %d cols (%d countries, %s → %s)",
        len(combined),
        combined.shape[1],
        combined["country"].nunique(),
        combined["date"].min().date(),
        combined["date"].max().date(),
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features.parquet"))
    args = parser.parse_args()

    historical_path = args.in_dir / "historical.parquet"
    snapshots_path = args.in_dir / "snapshots.parquet"
    vaccines_path = args.in_dir / "vaccines.parquet"

    if not historical_path.exists():
        log.error("Missing %s. Run `python -m src.data.fetch` first.", historical_path)
        return 1

    historical = pd.read_parquet(historical_path)
    historical["date"] = pd.to_datetime(historical["date"])
    snapshots = pd.read_parquet(snapshots_path)
    vaccines = pd.read_parquet(vaccines_path) if vaccines_path.exists() else None
    if vaccines is not None:
        vaccines["date"] = pd.to_datetime(vaccines["date"])

    features = build_all_features(historical, snapshots, vaccines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
