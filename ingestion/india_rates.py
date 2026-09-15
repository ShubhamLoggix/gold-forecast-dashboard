"""Indian city-wise retail gold rate ingestion (source: Groww gold-rates pages).

The hub page (https://groww.in/gold-rates) embeds today's per-gram retail rates
for ~350 Indian cities/territories inside its Next.js `__NEXT_DATA__` JSON.
City pages (gold-rate-today-in-<slug>) additionally carry a ~10-day daily
series for that city.

These are PUBLISHED RETAIL rates — import duty, GST, and local dealer premium
included — unlike the theoretical bullion-equivalent conversion elsewhere in
this project. Cached canonical frame: data/processed/india_gold_rates.parquet
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

import pandas as pd

from config import settings

logger = logging.getLogger("gold_forecast.ingestion")

GROWW_GOLD_RATES_URL = "https://groww.in/gold-rates"
GROWW_CITY_URL_FMT = "https://groww.in/gold-rates/gold-rate-today-in-{slug}"
INDIA_RATES_FILENAME = "india_gold_rates.parquet"
SOURCE_NAME = "groww:gold-rates"

_HUB_TTL_SECONDS = 900.0
_hub_cache: dict | None = None
_hub_cached_at = 0.0
_hub_lock = threading.Lock()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

_CARAT_KEYS = {"24k": "TWENTY_FOUR", "22k": "TWENTY_TWO", "18k": "EIGHTEEN"}
_CARAT_COLS = {
    "24k": "price_24k_pg",
    "22k": "price_22k_pg",
    "18k": "price_18k_pg",
}

_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def _fetch_gold_rate_data(url: str) -> dict:
    import requests

    resp = requests.get(url, headers=_HEADERS, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Groww gold-rates request failed: HTTP {resp.status_code}")
    match = _NEXT_DATA_RE.search(resp.text)
    if not match:
        raise RuntimeError("Groww page did not contain __NEXT_DATA__ (layout changed?)")
    data = json.loads(match.group(1))
    gold_rate_data = data["props"]["pageProps"].get("goldRateData")
    if not gold_rate_data:
        raise RuntimeError("Groww __NEXT_DATA__ missing goldRateData (layout changed?)")
    return gold_rate_data


def _fetch_hub() -> dict:
    global _hub_cache, _hub_cached_at
    with _hub_lock:
        if _hub_cache is not None and time.monotonic() - _hub_cached_at < _HUB_TTL_SECONDS:
            return _hub_cache
        hub = _fetch_gold_rate_data(GROWW_GOLD_RATES_URL)
        _hub_cache = hub
        _hub_cached_at = time.monotonic()
        return hub


def _record_to_row(slug: str, rec: dict) -> dict | None:
    price = rec.get("price") or {}
    date = rec.get("date")
    if not date or not price:
        return None
    row = {
        "date": pd.Timestamp(date).normalize(),
        "city_slug": slug,
        "city": rec.get("priceLocation") or slug,
        "source": SOURCE_NAME,
    }
    for karat, key in _CARAT_KEYS.items():
        value = price.get(key)
        row[_CARAT_COLS[karat]] = float(value) if value is not None else float("nan")
    return row


def snapshot_frame(grd: dict | None = None) -> pd.DataFrame:
    """All locations' latest-day per-gram rates from the hub page."""
    grd = grd if grd is not None else _fetch_hub()
    rows = []
    for slug, rec in (grd.get("physicalGoldRate") or {}).items():
        row = _record_to_row(slug, rec)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def city_history_frame(grd: dict, slug: str) -> pd.DataFrame:
    """Per-gram daily series for `slug` from a page's daySummary block."""
    summary = (grd.get("daySummary") or {}).get(slug) or {}
    rows = []
    for date, rec in sorted(summary.items()):
        row = _record_to_row(slug, rec)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_city_history(slug: str) -> pd.DataFrame:
    """~10-day per-gram series for one city from its dedicated Groww page."""
    grd = _fetch_gold_rate_data(GROWW_CITY_URL_FMT.format(slug=slug))
    frame = city_history_frame(grd, slug)
    if frame.empty:
        raise RuntimeError(f"Groww city page for {slug!r} has no daySummary series")
    return frame


def _cache_path(data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir or settings.data_dir)
    return data_dir / "processed" / INDIA_RATES_FILENAME


def _merge_save(frame: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    out_path = _cache_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        cached = pd.read_parquet(out_path)
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        frame = pd.concat([cached, frame], ignore_index=True)
    frame = (
        frame.sort_values(["city_slug", "date"])
        .drop_duplicates(subset=["city_slug", "date"], keep="last")
        .reset_index(drop=True)
    )
    frame.to_parquet(out_path, index=False)
    return frame


def load_cache(data_dir: Path | None = None) -> pd.DataFrame:
    path = _cache_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"India rates cache not found at {path}")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def cities_catalog(grd: dict | None = None) -> list[dict]:
    """City catalog from the hub page locations block."""
    grd = grd if grd is not None else _fetch_hub()
    locations = grd.get("locations") or []
    return [
        {
            "slug": loc.get("slug") or "",
            "name": loc.get("name") or "",
            "type": loc.get("type") or "city",
            "state_name": loc.get("stateName"),
        }
        for loc in locations
        if loc.get("slug")
    ]


def refresh_india_rates(
    history_cities: tuple[str, ...] = ("pune", "mumbai", "delhi"),
    data_dir: Path | None = None,
) -> int:
    """Scheduled/manual refresh: store today's snapshot for ALL cities plus the
    10-day series for the given cities. Returns total rows now in cache."""
    total = snapshot_frame()
    merged = _merge_save(total, data_dir)
    for slug in history_cities:
        try:
            merged = _merge_save(fetch_city_history(slug), data_dir)
        except Exception as exc:  # noqa: BLE001 - one city must not kill the refresh
            logger.warning("India city history fetch failed for %s: %s", slug, exc)
    logger.info("India retail rates cached (%d rows).", len(merged))
    return len(merged)


def get_city_rates(slug: str, data_dir: Path | None = None) -> dict:
    """Today's rate + recent history for one city, for the API layer.

    History comes from the cache when it already has >= 3 distinct dates for
    the city; otherwise the city page is fetched and merged in.
    """
    hub = _fetch_hub()
    records = hub.get("physicalGoldRate") or {}
    if slug not in records:
        raise KeyError(f"unknown city slug: {slug!r}")
    today_row = _record_to_row(slug, records[slug])
    if today_row is None:
        raise RuntimeError(f"Groww returned no usable rate record for {slug!r}")

    try:
        cached = load_cache(data_dir)
    except FileNotFoundError:
        cached = pd.DataFrame(
            columns=[
                "date",
                "city_slug",
                "city",
                "price_24k_pg",
                "price_22k_pg",
                "price_18k_pg",
                "source",
            ]
        )
    city_rows = cached[cached["city_slug"] == slug].sort_values("date")
    if city_rows["date"].nunique() < 3:
        city_rows = _merge_save(fetch_city_history(slug), data_dir)
        city_rows = city_rows[city_rows["city_slug"] == slug].sort_values("date")

    per_gram = {karat: today_row[_CARAT_COLS[karat]] for karat in _CARAT_KEYS}
    pct_change = None
    change = (records[slug].get("percentageChange") or {})
    mapped = {
        karat: change.get(key) for karat, key in _CARAT_KEYS.items() if change.get(key) is not None
    }
    if mapped:
        pct_change = mapped

    history = [
        {
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "price_24k_pg": float(row["price_24k_pg"]),
            "price_22k_pg": float(row["price_22k_pg"]),
            "price_18k_pg": float(row["price_18k_pg"]),
        }
        for _, row in city_rows.iterrows()
    ]
    return {
        "city_slug": slug,
        "city": today_row["city"],
        "date": pd.Timestamp(today_row["date"]).date(),
        "per_gram": per_gram,
        "pct_change": pct_change,
        "history": history,
    }
