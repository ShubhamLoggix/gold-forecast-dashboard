"""USD/INR exchange rate ingestion (yfinance `INR=X` primary).

Follows the same fetch -> validate -> cache patterns as the gold ingestion.
Cached canonical series: data/processed/usdinr_daily.parquet
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from config import settings
from ingestion.validation import validate_and_clean

logger = logging.getLogger("gold_forecast.ingestion")

USDINR_TICKER = "INR=X"
USDINR_CANONICAL_FILENAME = "usdinr_daily.parquet"


def fetch_usdinr_rate(
    start_date: dt.date,
    end_date: dt.date,
    force_refresh: bool = False,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch the USD/INR daily FX close series, cached and validated.

    Returns columns date/open/high/low/close/volume/source (business days).
    Missing FX days are NOT interpolated; use `resolve_rate_for_date` at
    lookup time to fall back to the most recent prior available rate.
    """
    data_dir = Path(data_dir or settings.data_dir)
    out_path = data_dir / "processed" / USDINR_CANONICAL_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force_refresh:
        cached = pd.read_parquet(out_path)
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        cov_start, cov_end = cached["date"].min().date(), cached["date"].max().date()
        if cov_start <= start_date and end_date <= cov_end:
            logger.info("USDINR cache hit [%s, %s].", start_date, end_date)
            return _slice(cached, start_date, end_date)

    fresh = _fetch_yfinance_fx(start_date, end_date)
    clean = validate_and_clean(fresh)
    clean["date"] = pd.to_datetime(clean["date"]).dt.normalize()

    if out_path.exists():
        cached = pd.read_parquet(out_path)
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        combined = pd.concat([cached, clean], ignore_index=True)
        combined = (
            combined.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = clean
    combined.to_parquet(out_path, index=False)
    logger.info(
        "USDINR series written to %s (%d rows, %s -> %s).",
        out_path, len(combined),
        combined["date"].min().date(), combined["date"].max().date(),
    )
    return _slice(combined, start_date, end_date)


def load_usdinr(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the cached USDINR canonical series (raises if missing)."""
    data_dir = Path(data_dir or settings.data_dir)
    path = data_dir / "processed" / USDINR_CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"USDINR series not found at {path}. Run fetch_usdinr_rate() first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def resolve_rate_for_date(
    target_date: dt.date,
    rates: pd.DataFrame | None = None,
    data_dir: Path | None = None,
    max_stale_days: int = 4,
) -> dict:
    """Resolve the USDINR rate for a date.

    Uses the rate on `target_date` if present, else the most recent prior
    available rate (NOT interpolated). Returns:
      {"rate": float, "rate_date": date, "rate_may_be_stale": bool}
    `rate_may_be_stale` is True when the fallback reaches back more than
    `max_stale_days` days before the target date (weekends/holidays excluded).
    """
    rates = rates if rates is not None else load_usdinr(data_dir)
    rates = rates.sort_values("date")
    if rates.empty:
        raise ValueError("USDINR rate series is empty")
    dates = pd.to_datetime(rates["date"])
    target = pd.Timestamp(target_date)
    usable = rates[dates <= target]
    if usable.empty:
        raise ValueError(f"No USDINR rate on or before {target_date}")
    row = usable.iloc[-1]
    rate_date = pd.Timestamp(row["date"]).date()
    gap_days = (target - pd.Timestamp(row["date"])).days
    return {
        "rate": float(row["close"]),
        "rate_date": pd.Timestamp(row["date"]).date(),
        "rate_may_be_stale": gap_days > max_stale_days,
    }


def latest_rate(
    rates: pd.DataFrame | None = None,
    data_dir: Path | None = None,
    reference_date: dt.date | None = None,
    max_stale_days: int = 4,
) -> dict:
    """Latest available USDINR rate (used for chart conversion), with staleness flag.

    `reference_date` lets callers express staleness relative to the newest gold
    date rather than today.
    """
    rates = rates if rates is not None else load_usdinr(data_dir)
    if rates.empty:
        raise ValueError("USDINR rate series is empty")
    rates = rates.sort_values("date")
    row = rates.iloc[-1]
    rate_date = pd.Timestamp(row["date"]).date()
    ref = pd.Timestamp(reference_date) if reference_date else pd.Timestamp(dt.date.today())
    gap_days = (ref - pd.Timestamp(row["date"])).days
    return {
        "rate": float(row["close"]),
        "rate_date": rate_date,
        "rate_may_be_stale": gap_days > max_stale_days,
    }


def rate_frame_for_dates(
    dates: pd.Series,
    rates: pd.DataFrame | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Per-date USDINR rates via backward as-of join (no interpolation).

    Each date gets the most recent available rate on or before it; a date with
    no prior rate at all gets NaN (caller decides: skip/stale-flag/raise).
    Returns a frame aligned 1:1 with `dates` (any index) with columns
    ["usd_inr_rate", "usd_inr_rate_date"].
    """
    rates = rates if rates is not None else load_usdinr(data_dir)
    if rates.empty:
        raise ValueError("USDINR rate series is empty")
    lookup = pd.DataFrame(
        {
            "date": pd.to_datetime(rates["date"]).dt.normalize(),
            "usd_inr_rate": rates["close"].astype(float),
            "usd_inr_rate_date": pd.to_datetime(rates["date"]).dt.normalize(),
        }
    ).sort_values("date")
    left = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).dt.normalize()})
    merged = pd.merge_asof(left, lookup, on="date", direction="backward")
    # Dates before the FX series begins have no backward rate: fall back to the
    # earliest available rate (NOT 0/NaN) — callers already flag staleness via
    # the latest_rate()/resolve_rate_for_date() helpers.
    if merged["usd_inr_rate"].isna().any():
        first_rate = float(lookup["usd_inr_rate"].iloc[0])
        merged["usd_inr_rate"] = merged["usd_inr_rate"].fillna(first_rate)
        merged["usd_inr_rate_date"] = merged["usd_inr_rate_date"].fillna(lookup["date"].iloc[0])
    return merged[["usd_inr_rate", "usd_inr_rate_date"]]


def _slice(df: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    out = df.loc[mask].reset_index(drop=True)
    return out[["date", "open", "high", "low", "close", "volume", "source"]]


def _fetch_yfinance_fx(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    import yfinance as yf

    logger.info("Fetching %s from yfinance [%s, %s].", USDINR_TICKER, start_date, end_date)
    raw = yf.download(
        USDINR_TICKER,
        start=start_date,
        end=end_date + dt.timedelta(days=1),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {USDINR_TICKER}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"]),
            "open": raw.get("Open"),
            "high": raw.get("High"),
            "low": raw.get("Low"),
            "close": raw.get("Close"),
            "volume": raw.get("Volume"),
            "source": "yfinance:INR=X",
        }
    )
