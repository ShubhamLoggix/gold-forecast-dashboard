"""Gold price ingestion from real, cited sources (yfinance GC=F primary).

No fabricated data in production paths: every price comes from a cited source
(`source` column records which one). Tests use clearly-labeled fixtures and
never hit the network.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from config import settings
from ingestion.validation import REQUIRED_COLUMNS, validate_and_clean

logger = logging.getLogger("gold_forecast.ingestion")

YFINANCE_TICKER = "GC=F"  # COMEX Gold Futures (continuous), USD/oz
LBMA_JSON_URL = "https://prices.lbma.org.uk/json/gold_pm.json"
METALS_API_BASE = "https://api.metals-api.com/v1/timeseries"

CACHE_TEMPLATE = "gold_prices_{source}.parquet"
CANONICAL_FILENAME = "gold_prices_daily.parquet"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fetch_history(
    start_date: dt.date,
    end_date: dt.date,
    source: str = "auto",
    force_refresh: bool = False,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch historical gold prices (daily) from a real, cited source.

    Args:
        start_date: inclusive start.
        end_date: inclusive end.
        source: "yfinance" | "lbma" | "metals_api" | "auto" (priority order).
        force_refresh: bypass the local cache and re-fetch.
        data_dir: override base data dir (defaults to settings.data_dir).

    Returns:
        DataFrame with columns date, open, high, low, close, volume, source.
    """
    data_dir = Path(data_dir or settings.data_dir)
    order = (
        [source]
        if source != "auto"
        else ["yfinance", "lbma", "metals_api"] if settings.metals_api_key else ["yfinance", "lbma"]
    )
    errors: list[str] = []
    for src in order:
        try:
            df = _fetch_from_source_cached(
                start_date, end_date, src, force_refresh=force_refresh, data_dir=data_dir
            )
            if df.empty:
                raise RuntimeError("source returned 0 rows")
            return df
        except Exception as exc:  # noqa: BLE001 - fall down the source list
            errors.append(f"{src}: {exc}")
            logger.warning("Source %s failed (%s); trying next source.", src, exc)
    raise RuntimeError("All sources failed: " + " | ".join(errors))


def update_canonical(
    start_date: dt.date,
    end_date: dt.date,
    source: str = "auto",
    force_refresh: bool = False,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch, validate, and write the canonical processed daily series.

    Returns the validated canonical DataFrame (close prices, business days).
    """
    data_dir = Path(data_dir or settings.data_dir)
    raw = fetch_history(start_date, end_date, source=source, force_refresh=force_refresh,
                        data_dir=data_dir)
    clean = validate_and_clean(raw)
    out_path = data_dir / "processed" / CANONICAL_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    logger.info("Canonical series written to %s (%d rows, %s -> %s).",
                out_path, len(clean), clean["date"].min().date(), clean["date"].max().date())
    return clean


def load_canonical(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the canonical clean daily series (raises if missing)."""
    data_dir = Path(data_dir or settings.data_dir)
    path = data_dir / "processed" / CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical series not found at {path}. Run update_canonical() first."
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Raw cache layer
# --------------------------------------------------------------------------- #
def _fetch_from_source_cached(
    start_date: dt.date,
    end_date: dt.date,
    source: str,
    force_refresh: bool,
    data_dir: Path,
) -> pd.DataFrame:
    cache_path = data_dir / "raw" / CACHE_TEMPLATE.format(source=source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_cache(cache_path)
    if cached is not None and not force_refresh:
        cov_start, cov_end = cached["date"].min().date(), cached["date"].max().date()
        if cov_start <= start_date and end_date <= cov_end:
            logger.info("Cache hit for %s [%s, %s].", source, start_date, end_date)
            return _slice_range(cached, start_date, end_date)

    fresh = _fetchers[source](start_date, end_date)
    if cached is None:
        merged = fresh
    else:
        merged = _merge_frames(cached, fresh)
    _save_cache(merged, cache_path)
    return _slice_range(merged, start_date, end_date)


def _load_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001 - corrupt cache should not kill ingestion
        logger.warning("Cache file %s unreadable (%s); refetching.", cache_path, exc)
        return None


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    df.to_parquet(cache_path, index=False)
    logger.info("Wrote %d cached rows to %s.", len(df), cache_path)


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([old, new], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    # New rows win over stale cached rows for the same date.
    combined = combined.sort_values(["date", "source"]).drop_duplicates(
        subset=["date"], keep="last"
    )
    return combined.sort_values("date").reset_index(drop=True)


def _slice_range(df: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    out = df.loc[mask].copy().reset_index(drop=True)
    return out[REQUIRED_COLUMNS] if not out.empty else out


# --------------------------------------------------------------------------- #
# Source adapters (each returns the standard schema)
# --------------------------------------------------------------------------- #
def _fetch_yfinance(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    import yfinance as yf

    logger.info("Fetching %s from yfinance [%s, %s].", YFINANCE_TICKER, start_date, end_date)
    raw = yf.download(
        YFINANCE_TICKER,
        start=start_date,
        end=end_date + dt.timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {YFINANCE_TICKER}")

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
            "source": "yfinance:GC=F",
        }
    )
    return df


def _fetch_lbma(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    import urllib.request
    import json as _json

    logger.info("Fetching LBMA PM gold from %s [%s, %s].", LBMA_JSON_URL, start_date, end_date)
    with urllib.request.urlopen(LBMA_JSON_URL, timeout=30) as resp:  # noqa: S310 - fixed https URL
        payload = _json.loads(resp.read().decode("utf-8"))
    rows = []
    for entry in payload:
        if isinstance(entry, dict):
            d = str(entry.get("d", ""))[:10]
            usd = entry.get("usd", entry.get("v", [None, None, None])[0])
        else:
            d, usd = str(entry[0])[:10], entry[1]
        try:
            date = dt.date.fromisoformat(d)
        except ValueError:
            continue
        if usd is None:
            continue
        rows.append(
            {
                "date": date,
                "open": None,
                "high": None,
                "low": None,
                "close": float(usd),
                "volume": None,
                "source": "lbma:gold_pm",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("LBMA feed returned no parsable rows")
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def _fetch_metals_api(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if not settings.metals_api_key:
        raise RuntimeError("METALS_API_KEY not set; metals_api source disabled")
    import urllib.request
    import json as _json

    url = (
        f"{METALS_API_BASE}?start_date={start_date.isoformat()}"
        f"&end_date={end_date}&access_key={settings.metals_api_key}"
    )
    logger.info("Fetching metals-api timeseries [%s, %s].", start_date, end_date)
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - https API
        payload = _json.loads(resp.read().decode("utf-8"))
    rates = payload.get("rates", {})
    rows = []
    for day, values in sorted(rates.items()):
        xau = values.get("XAU")
        if not xau:
            continue
        usd = float(xau)
        usd = 1.0 / usd if usd < 1.0 else usd  # metals-api: USD->XAU rate; invert
        rows.append(
            {
                "date": dt.date.fromisoformat(day),
                "open": None,
                "high": None,
                "low": None,
                "close": usd,
                "volume": None,
                "source": "metals_api:XAU",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("metals-api returned no XAU rates")
    return df


_fetchers = {
    "yfinance": _fetch_yfinance,
    "lbma": _fetch_lbma,
    "metals_api": _fetch_metals_api,
}
