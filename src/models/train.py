"""
Train + evaluate any/all models. Saves predictions and a metrics summary.

Usage
-----
    python -m src.models.train --models all
    python -m src.models.train --models lightgbm lstm
    python -m src.models.train --models lightgbm --countries India UK Germany Brazil
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.models.evaluate import evaluate_predictions, rolling_origin_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/forecasts"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lightgbm", "prophet", "lstm", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Limit to specific countries (e.g. for faster debugging).",
    )
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--n_splits", type=int, default=4)
    args = parser.parse_args()

    if not args.features.exists():
        log.error(
            "Features file not found: %s. Run `python -m src.features.build` first.",
            args.features,
        )
        return 1

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])

    if args.countries:
        df = df[df["country"].isin(args.countries)]
        log.info("Filtered to %d countries", df["country"].nunique())

    args.out.mkdir(parents=True, exist_ok=True)

    splits = rolling_origin_splits(
        df["date"], n_splits=args.n_splits, horizon_days=args.horizon
    )
    log.info("Rolling-origin splits: %d", len(splits))
    for s in splits:
        log.info("  %s", s)

    requested = (
        ["lightgbm", "prophet", "lstm"] if "all" in args.models else args.models
    )

    summary = {}
    for name in requested:
        log.info("=" * 60)
        log.info("Running %s", name.upper())
        log.info("=" * 60)
        try:
            if name == "lightgbm":
                from src.models.gbm import run_lightgbm, LightGBMConfig
                preds = run_lightgbm(df, splits, LightGBMConfig(horizon_days=args.horizon))
            elif name == "prophet":
                from src.models.prophet_model import run_prophet
                preds = run_prophet(df, splits, horizon_days=args.horizon)
            elif name == "lstm":
                from src.models.lstm_model import run_lstm, LSTMConfig
                preds = run_lstm(df, splits, LSTMConfig(horizon_days=args.horizon))
            else:
                continue
        except Exception as e:
            log.error("%s failed: %s", name, e)
            continue

        if preds.empty:
            log.warning("%s produced no predictions, skipping", name)
            continue

        preds.to_parquet(args.out / f"{name}_predictions.parquet", index=False)
        result = evaluate_predictions(preds, panel=df)
        summary[name] = asdict(result)
        log.info("%s: MAPE=%.2f%%, MAE=%.1f, n=%d",
                 name, result.overall_mape, result.overall_mae, result.n_predictions)
        if result.interval_coverage_80 is not None:
            log.info("  80%% PI coverage: %.1f%%", result.interval_coverage_80 * 100)

    with open(args.out / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote summary → %s", args.out / "summary_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
