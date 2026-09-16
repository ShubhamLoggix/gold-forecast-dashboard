"""COMEX Silver (SI=F) ingestion — mirrors the gold ingestion patterns.

yfinance-first (SI=F, continuous COMEX silver futures, USD/oz). Silver has no
LBMA/metals-api fallback equivalents wired, so the chain is yfinance-only for
now; the cache/validation/canonical patterns are identical to gold's.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from config import settings
from ingestion.validation import REQUIRED_COLUMNS, validate_and_clean

logger = logging.getLogger("gold_forecast.ingestion")

SILVER_YFINANCE_TICKER = "SI=F"  # COMEX Silver Futures (continuous), USD/oz
SILVER_CACHE_TEMPLATE = "silver_prices_{source}.parquet"
SILVER_CANONICAL_FILENAME = "silver_prices_daily.parquet"


def fetch_history(
    start_date: dt.date,
    end_date: dt.date,
    source: str = "auto",
    force_refresh: bool = False,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch historical COMEX silver prices (daily) from yfinance (cited)."""
    data_dir = Path(data_dir or settings.data_dir)
    order = [source] if source != "auto" else ["yfinance"]
    errors: list[str] = []
    for src in order:
        try:
            df = _fetch_cached(
                start_date, end_date, src, force_refresh=force_refresh, data_dir=data_dir
            )
            if df.empty:
                raise RuntimeError("source returned 0 rows")
            return df
        except Exception as exc:  # noqa: BLE001 - fall through source list
            errors.append(f"{src}: {exc}")
            logger.warning("Silver source %s failed (%s).", src, exc)
    raise RuntimeError("All silver sources failed: " + " | ".join(errors))


def update_canonical(
    start_date: dt.date,
    end_date: dt.date,
    source: str = "auto",
    force_refresh: bool = False,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch, validate, and write the canonical silver daily series."""
    data_dir = Path(data_dir or settings.data_dir)
    raw = fetch_history(start_date, end_date, source=source,
                        force_refresh=force_refresh, data_dir=data_dir)
    clean = validate_and_clean(raw)
    out_path = data_dir / "processed" / SILVER_CANONICAL_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    logger.info(
        "Silver canonical series written to %s (%d rows, %s -> %s).",
        out_path, len(clean), clean["date"].min().date(), clean["date"].max().date(),
    )
    return clean


def load_canonical(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the canonical clean silver daily series (raises if missing)."""
    data_dir = Path(data_dir or settings.data_dir)
    path = data_dir / "processed" / SILVER_CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Silver canonical series not found at {path}. Run update_canonical() first."
        )
    return pd.read_parquet(path)


def _fetch_cached(
    start_date: dt.date,
    end_date: dt.date,
    source: str,
    force_refresh: bool,
    data_dir: Path,
) -> pd.DataFrame:
    cache_path = data_dir / "raw" / SILVER_CACHE_TEMPLATE.format(source=source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        cov_start = cached["date"].min().date()
        cov_end = cached["date"].max().date()
        if cov_start <= start_date and end_date <= cov_end:
            return _slice_range(cached, start_date, end_date)

    fresh = _fetch_yfinance(start_date, end_date)
    if cache_path.exists():
        old = pd.read_parquet(cache_path)
        old["date"] = pd.to_datetime(old["date"])
        fresh = _merge_frames(old, fresh)
    _save_cache(fresh, cache_path)
    return _slice_range(fresh, start_date, end_date)


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    df.to_parquet(cache_path, index=False)


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([old, new], ignore_index=True)
    return (
        combined.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _slice_range(df: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    from ingestion.fetch_gold_prices import _slice_range as gold_slice

    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    out = df.loc[mask].copy().reset_index(drop=True)
    return out[REQUIRED_COLUMNS] if not out.empty else out


def _fetch_yfinance(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    import yfinance as yf

    logger.info(
        "Fetching %s from yfinance [%s, %s].", SILVER_YFINANCE_TICKER, start_date, end_date
    )
    raw = yf.download(
        SILVER_YFINANCE_TICKER,
        start=start_date,
        end=end_date + dt.timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {SILVER_YFINANCE_TICKER}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"]),
            "open": raw.get("Open"),
            "high": raw.get("High"),
            "low": raw.get("Low"),
            "close": raw.get("Close"),
            "volume": raw.get("Volume"),
            "source": "yfinance:SI=F",
        }
    )
    return df
