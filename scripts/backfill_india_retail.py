"""One-time (rerunnable) backfill of Indian city retail history.

Groww only publishes ~10 days; MCX is bot-blocked; so historical retail is
ESTIMATED: bullion COMEX close (per-date FX) x the city's current live retail
premium, tagged `comex_converted`. Live `groww_live` rows always win.

Usage:
    python scripts/backfill_india_retail.py [--city pune]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=settings.digest_city)
    args = parser.parse_args()
    setup_logging(settings.log_level)

    from ingestion.india_rates import backfill_from_comex

    result = backfill_from_comex(city_slug=args.city)
    print(f"city={args.city} backfilled={result['backfilled']} gaps={len(result['gaps'])}")
    if result.get("gaps"):
        print("gap dates (not filled):", ", ".join(result["gaps"]))


if __name__ == "__main__":
    main()
