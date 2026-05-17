"""
disease.sh API client.

Wraps the [disease.sh](https://disease.sh/) free, open, no-auth API.

Why a client class rather than ad-hoc requests:
  - Centralised retry + backoff (disease.sh occasionally 502s under load)
  - On-disk JSON cache (so we don't hammer the API during development)
  - Single place for the base URL if it ever moves
  - Typed return shapes so the rest of the codebase doesn't drown in dicts

Endpoints we use (all GET, all return JSON):
  /v3/covid-19/historical/{country}?lastdays={n|all}
      → {country, province, timeline: {cases, deaths, recovered}}
  /v3/covid-19/countries/{country}
      → snapshot incl. population, continent, todayCases, ...
  /v3/covid-19/countries
      → list of all countries (use this for the master list)
  /v3/covid-19/vaccine/coverage/countries/{country}?lastdays={n|all}
      → {country, timeline: {date: dose_count}}

Schema reference: https://disease.sh/docs/
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://disease.sh"


@dataclass
class CountrySnapshot:
    """Current-state country record from /v3/covid-19/countries/{country}."""
    country: str
    iso2: str | None
    iso3: str | None
    continent: str | None
    population: int | None
    cases: int
    deaths: int
    recovered: int
    active: int
    tests: int | None
    cases_per_million: float | None
    deaths_per_million: float | None
    tests_per_million: float | None

    @classmethod
    def from_api(cls, d: dict) -> "CountrySnapshot":
        info = d.get("countryInfo", {}) or {}
        return cls(
            country=d.get("country", ""),
            iso2=info.get("iso2"),
            iso3=info.get("iso3"),
            continent=d.get("continent"),
            population=d.get("population"),
            cases=int(d.get("cases", 0)),
            deaths=int(d.get("deaths", 0)),
            recovered=int(d.get("recovered", 0)),
            active=int(d.get("active", 0)),
            tests=d.get("tests"),
            cases_per_million=d.get("casesPerOneMillion"),
            deaths_per_million=d.get("deathsPerOneMillion"),
            tests_per_million=d.get("testsPerOneMillion"),
        )


class DiseaseAPIError(RuntimeError):
    """Raised when disease.sh returns a non-recoverable error (404, 400, etc)."""


class DiseaseAPIClient:
    """Thin client with on-disk JSON caching.

    Example
    -------
    >>> client = DiseaseAPIClient(cache_dir=Path("data/raw/disease_sh"))
    >>> snap = client.country_snapshot("India")
    >>> ts = client.country_historical("India", lastdays="all")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | None = None,
        rate_limit_seconds: float = 0.2,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self._last_request_at = 0.0
        self._session = requests.Session()
        # Polite UA — disease.sh logs traffic and may rate-limit anonymous bots
        self._session.headers.update({
            "User-Agent": "pandemic-forecaster-research/0.1 (educational)",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------ low-level

    def _cache_path(self, path: str, params: dict | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = path + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        # Human-readable prefix from the path
        prefix = path.strip("/").replace("/", "_")[:80]
        return self.cache_dir / f"{prefix}__{digest}.json"

    def _throttle(self) -> None:
        """Sleep to keep us under disease.sh's polite limits."""
        gap = time.monotonic() - self._last_request_at
        if gap < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - gap)
        self._last_request_at = time.monotonic()

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _get_raw(self, path: str, params: dict | None = None) -> Any:
        """GET with retries on transient network errors. 4xx errors are not
        retried (they won't fix themselves)."""
        self._throttle()
        url = f"{self.base_url}{path}"
        log.debug("GET %s params=%s", url, params)
        resp = self._session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 404:
            raise DiseaseAPIError(f"Not found: {url}")
        if 400 <= resp.status_code < 500:
            raise DiseaseAPIError(
                f"Client error {resp.status_code} on {url}: {resp.text[:200]}"
            )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> Any:
        """GET with on-disk caching. Cached results are returned indefinitely
        unless `force_refresh` was set at construction — for this project we
        accept staleness up to one day (the user re-runs `fetch.py` daily)."""
        cache_path = self._cache_path(path, params)
        if cache_path is not None and cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                log.warning("Corrupted cache %s, refetching", cache_path.name)

        data = self._get_raw(path, params)

        if cache_path is not None:
            cache_path.write_text(json.dumps(data))
        return data

    # ------------------------------------------------------------------ public API

    def all_countries(self) -> list[dict]:
        """All countries with current-state snapshots. Useful for picking the
        master list of countries to backfill."""
        return self._get("/v3/covid-19/countries")

    def country_snapshot(self, country: str) -> CountrySnapshot:
        """Current state for a single country (today's totals + metadata)."""
        d = self._get(f"/v3/covid-19/countries/{country}", {"strict": "true"})
        return CountrySnapshot.from_api(d)

    def country_historical(
        self,
        country: str,
        lastdays: str | int = "all",
    ) -> dict[str, dict[str, int]]:
        """Daily cumulative time series for one country.

        Returns
        -------
        {
            'cases':     {'M/D/YY': cumulative_count, ...},
            'deaths':    {'M/D/YY': cumulative_count, ...},
            'recovered': {'M/D/YY': cumulative_count, ...},
        }
        """
        d = self._get(
            f"/v3/covid-19/historical/{country}",
            {"lastdays": str(lastdays)},
        )
        # The endpoint sometimes returns the timeline at the top level,
        # sometimes nested under 'timeline' depending on whether the country
        # has provincial breakdowns. Normalise.
        if "timeline" in d:
            return d["timeline"]
        return {k: d[k] for k in ("cases", "deaths", "recovered") if k in d}

    def country_vaccine(
        self,
        country: str,
        lastdays: str | int = "all",
    ) -> dict[str, int]:
        """Cumulative vaccine doses by date for one country.

        Returns
        -------
        {'M/D/YY': cumulative_doses, ...}
        """
        d = self._get(
            f"/v3/covid-19/vaccine/coverage/countries/{country}",
            {"lastdays": str(lastdays), "fullData": "false"},
        )
        return d.get("timeline", {}) if isinstance(d, dict) else {}
