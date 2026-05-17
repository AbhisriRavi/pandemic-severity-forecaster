"""
Fetch data from disease.sh for a configurable set of countries.

Usage
-----
    # Default: top 100 most-populous countries, full history
    python -m src.data.fetch

    # All countries (slower, ~5 minutes)
    python -m src.data.fetch --all

    # Specific list
    python -m src.data.fetch --countries India UK Germany Japan

Output
------
data/raw/disease_sh/                   (cached JSON)
data/processed/historical.parquet      (tidy long: country, date, cases, deaths, recovered)
data/processed/snapshots.parquet       (country-level metadata)
data/processed/vaccines.parquet        (country, date, cumulative_doses)
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.client import DiseaseAPIClient, DiseaseAPIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_timeline(timeline: dict[str, int]) -> pd.DataFrame:
    """Convert {'M/D/YY': cumulative_count} → DataFrame[date, value]."""
    if not timeline:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(
        [(pd.to_datetime(k, format="%m/%d/%y"), v) for k, v in timeline.items()],
        columns=["date", "value"],
    )
    return df.sort_values("date").reset_index(drop=True)


def pick_countries(
    client: DiseaseAPIClient,
    explicit: list[str] | None,
    top_n: int | None,
) -> list[str]:
    """Decide which countries to fetch.

    If `explicit` is given, use that list verbatim.
    Otherwise, pull all countries and (optionally) take the top-N by population.
    """
    if explicit:
        return explicit
    snapshots = client.all_countries()
    df = pd.DataFrame(snapshots)
    # Filter out entries with missing population (a handful of dependencies)
    df = df[df["population"].notna() & (df["population"] > 0)]
    if top_n is not None:
        df = df.nlargest(top_n, "population")
    return df["country"].tolist()


def fetch_all(
    countries: list[str],
    client: DiseaseAPIClient,
    lastdays: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch historical, snapshots, vaccines for a list of countries.

    Returns
    -------
    historical_long : DataFrame [country, date, cases, deaths, recovered]
    snapshots       : DataFrame [country, iso2, iso3, continent, population, ...]
    vaccines_long   : DataFrame [country, date, cumulative_doses]
    """
    historical_rows: list[pd.DataFrame] = []
    snapshot_rows: list[dict] = []
    vaccine_rows: list[pd.DataFrame] = []

    failed_country: list[tuple[str, str]] = []

    for country in tqdm(countries, desc="Fetching", unit="country"):
        try:
            snap = client.country_snapshot(country)
            snapshot_rows.append(asdict(snap))
        except DiseaseAPIError as e:
            failed_country.append((country, f"snapshot: {e}"))
            continue

        try:
            timeline = client.country_historical(country, lastdays=lastdays)
        except DiseaseAPIError as e:
            failed_country.append((country, f"historical: {e}"))
            continue

        # Build long-format country time series
        cases = _parse_timeline(timeline.get("cases", {})).rename(columns={"value": "cases"})
        deaths = _parse_timeline(timeline.get("deaths", {})).rename(columns={"value": "deaths"})
        recovered = _parse_timeline(timeline.get("recovered", {})).rename(columns={"value": "recovered"})

        if cases.empty:
            failed_country.append((country, "empty historical"))
            continue

        merged = cases.merge(deaths, on="date", how="outer").merge(
            recovered, on="date", how="outer"
        )
        merged["country"] = country
        historical_rows.append(merged)

        # Vaccines (not every country reports)
        try:
            vaccine_timeline = client.country_vaccine(country, lastdays=lastdays)
            if vaccine_timeline:
                vdf = _parse_timeline(vaccine_timeline).rename(
                    columns={"value": "cumulative_doses"}
                )
                vdf["country"] = country
                vaccine_rows.append(vdf)
        except DiseaseAPIError as e:
            log.debug("No vaccine data for %s: %s", country, e)

    if failed_country:
        log.warning("Failed for %d countries:", len(failed_country))
        for c, reason in failed_country[:5]:
            log.warning("  %s — %s", c, reason)

    historical_df = (
        pd.concat(historical_rows, ignore_index=True)
        if historical_rows
        else pd.DataFrame(columns=["country", "date", "cases", "deaths", "recovered"])
    )
    snapshots_df = pd.DataFrame(snapshot_rows)
    vaccines_df = (
        pd.concat(vaccine_rows, ignore_index=True)
        if vaccine_rows
        else pd.DataFrame(columns=["country", "date", "cumulative_doses"])
    )

    return historical_df, snapshots_df, vaccines_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Specific country names. Default: top 100 by population.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch every country (slower, ~5 min for ~220 countries).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Fetch top-N countries by population (default 100). Ignored if --all or --countries.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/raw/disease_sh"),
        help="JSON cache directory.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk caching (every run hits the API).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else args.cache
    client = DiseaseAPIClient(cache_dir=cache_dir)

    top_n = None if args.all else args.top
    countries = pick_countries(client, args.countries, top_n)
    log.info("Will fetch %d countries", len(countries))

    historical, snapshots, vaccines = fetch_all(countries, client)

    log.info(
        "Historical: %d rows × %d countries, %s → %s",
        len(historical),
        historical["country"].nunique(),
        historical["date"].min().date() if not historical.empty else "—",
        historical["date"].max().date() if not historical.empty else "—",
    )
    log.info("Snapshots: %d countries", len(snapshots))
    log.info("Vaccines: %d rows × %d countries",
             len(vaccines), vaccines["country"].nunique() if not vaccines.empty else 0)

    historical.to_parquet(args.out / "historical.parquet", index=False)
    snapshots.to_parquet(args.out / "snapshots.parquet", index=False)
    if not vaccines.empty:
        vaccines.to_parquet(args.out / "vaccines.parquet", index=False)
    log.info("Wrote outputs to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
