"""Run the walk-forward backtest and persist results to data/backtests/.

Usage:
    python scripts/run_backtest.py [--horizon 30] [--step 7] [--max-folds 52]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from forecasting.timesfm_service import GoldForecastService
from ingestion.fetch_gold_prices import load_canonical, update_canonical
from logging_setup import setup_logging

logger = logging.getLogger("gold_forecast.backtest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--max-folds", type=int, default=52)
    parser.add_argument("--suffix", default="", help="filename suffix, e.g. '-1y'")
    parser.add_argument("--refresh", action="store_true", help="re-fetch data first")
    args = parser.parse_args()

    setup_logging(settings.log_level)
    if args.refresh:
        import datetime as dt

        update_canonical(
            dt.date.fromisoformat(settings.history_start), dt.date.today()
        )
    try:
        history = load_canonical()
    except FileNotFoundError:
        import datetime as dt

        history = update_canonical(
            dt.date.fromisoformat(settings.history_start), dt.date.today()
        )
    service = GoldForecastService(
        model_version=settings.model_version, context_length=settings.context_length
    )
    service.load_model()
    results = service.backtest(
        history,
        horizon_days=args.horizon,
        step_days=args.step,
        max_folds=args.max_folds,
    )
    service.persist_backtest(results, suffix=args.suffix)
    for name, res in results.items():
        print(f"{name:>18}: {res.summary}")


if __name__ == "__main__":
    main()
