"""Retail–COMEX gold premium series in ₹/g, per karat.

premium(t) = groww_retail(t) − bullion_INR(t)

where bullion_INR(t) = COMEX close(t) × USD/INR(t) / 31.1035 × purity(karat).

Data-quality split (the spec's core honesty requirement):
  * quality="real"      — the date has a live Groww retail quote, so the
                          premium reflects ACTUAL retail/dealer movement.
  * quality="estimated" — the date only exists as `comex_converted` retail
                          (bullion × current premium ratio). Its premium is
                          therefore bullion × (ratio − 1): structurally tied
                          to bullion price, not independent movement. Kept so
                          the series has history, but tagged so confidence and
                          any backtest treat it separately from real rows.

The forecast + accuracy tracker only promise evidence on *real* out-of-sample
points that arrive AFTER live rows exist (those actuals are real by
construction). Estimated history is context, not proof.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import settings
from conversion import PURITY_FACTORS, TROY_OZ_TO_GRAM
from ingestion.india_rates import (
    _CARAT_COLS,
    _LIVE_SOURCES,
    load_cache as load_india_cache,
)

logger = logging.getLogger("gold_forecast.ingestion")

PREMIUM_FILENAME_FMT = "premium_{city}_{karat}.parquet"
SERIES_PREFIX = "premium:"


def series_id(city: str, karat: str) -> str:
    return f"{SERIES_PREFIX}{city}:{karat}"


def premium_filename(city: str, karat: str) -> str:
    return PREMIUM_FILENAME_FMT.format(city=city, karat=karat)


def premium_path(city: str, karat: str, data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir or settings.data_dir)
    return data_dir / "processed" / premium_filename(city, karat)


def _bullion_per_gram(karat: str) -> pd.DataFrame:
    """Date-indexed bullion close in INR/g at the karat's purity."""
    from ingestion.fetch_gold_prices import load_canonical
    from ingestion.usdinr import load_usdinr

    purity = PURITY_FACTORS[karat.lower()]
    gold = pd.DataFrame(
        {
            "date": pd.to_datetime(load_canonical()["date"]).dt.normalize().dt.as_unit("us"),
            "close": load_canonical()["close"].astype(float),
        }
    )
    fx = pd.DataFrame(
        {
            "date": pd.to_datetime(load_usdinr()["date"]).dt.normalize().dt.as_unit("us"),
            "usd_inr_rate": load_usdinr()["close"].astype(float),
        }
    ).sort_values("date")
    merged = pd.merge_asof(gold, fx, on="date", direction="backward")
    merged["bullion_pg"] = (
        merged["close"] * merged["usd_inr_rate"] / TROY_OZ_TO_GRAM * purity
    )
    return merged[["date", "bullion_pg"]]


def build_premium_series(
    city: str = "pune",
    karat: str = "22k",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Recompute (and persist) the premium series for one city/karat.

    Every India cache row is paired with the bullion equivalent on the same
    date. Live rows become quality="real"; `comex_converted` rows become
    quality="estimated". Dates with neither retail nor bullion are skipped.
    """
    karat = karat.lower()
    col = _CARAT_COLS[karat]
    retail = load_india_cache(data_dir)
    retail["date"] = pd.to_datetime(retail["date"]).dt.normalize().dt.as_unit("us")
    city_rows = retail[retail["city_slug"] == city][["date", col, "source"]].copy()
    city_rows = city_rows.rename(columns={col: "retail_pg"})
    if city_rows.empty:
        raise RuntimeError(f"No India retail cache rows for city={city!r}")

    bullion = _bullion_per_gram(karat)
    merged = pd.merge_asof(
        city_rows.sort_values("date"),
        bullion.sort_values("date"),
        on="date",
        direction="backward",
    ).sort_values("date")
    merged = merged.dropna(subset=["retail_pg", "bullion_pg"])
    merged["premium_pg"] = merged["retail_pg"] - merged["bullion_pg"]
    merged["quality"] = merged["source"].isin(_LIVE_SOURCES).map(
        {True: "real", False: "estimated"}
    )
    merged = merged[["date", "retail_pg", "bullion_pg", "premium_pg", "quality"]]

    out_path = premium_path(city, karat, data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    logger.info(
        "premium series %s:%s rebuilt (%d rows, real=%d, estimated=%d)",
        city,
        karat,
        len(merged),
        int((merged["quality"] == "real").sum()),
        int((merged["quality"] == "estimated").sum()),
    )
    return merged


def load_premium_series(
    city: str = "pune",
    karat: str = "22k",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    path = premium_path(city, karat, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"premium series not found at {path} — run build_premium_series first"
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def premium_forecast_input(
    city: str = "pune",
    karat: str = "22k",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """DataFrame in the exact shape the forecast service expects."""
    series = load_premium_series(city, karat, data_dir)
    return pd.DataFrame(
        {
            "date": series["date"],
            "close": series["premium_pg"].astype(float),
        }
    )


def premium_actuals(
    city: str = "pune",
    karat: str = "22k",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Realized premium per day, for scoring logged premium forecasts."""
    series = load_premium_series(city, karat, data_dir)
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(series["date"]).dt.normalize(),
            "actual": series["premium_pg"].astype(float),
        }
    ).drop_duplicates(subset="target_date", keep="last")