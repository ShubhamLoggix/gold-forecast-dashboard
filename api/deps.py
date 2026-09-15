"""Shared singletons and caches for the API layer."""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Any, Callable, Hashable

import pandas as pd

from config import settings
from forecasting.timesfm_service import GoldForecastService
from ingestion.fetch_gold_prices import load_canonical

logger = logging.getLogger("gold_forecast.api")

_service: GoldForecastService | None = None
_service_lock = threading.Lock()

# Response cache keyed by (kind, args...); invalidated on data refresh.
_response_cache: dict[tuple, object] = {}
_response_cache_lock = threading.Lock()

# Health bookkeeping for the scheduler.
last_refresh_ok = True
last_refresh_attempt: float | None = None


def get_history() -> pd.DataFrame:
    return load_canonical()


def get_service() -> GoldForecastService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                service = GoldForecastService(
                    model_version=settings.model_version,
                    context_length=settings.context_length,
                    device=settings.device,
                )
                service.load_model()
                _service = service
    return _service


def cache_get(key: tuple) -> object | None:
    with _response_cache_lock:
        return _response_cache.get(key)


def cache_set(key: tuple, value: object) -> None:
    with _response_cache_lock:
        _response_cache[key] = value


def invalidate_response_cache() -> None:
    with _response_cache_lock:
        _response_cache.clear()
    logger.info("Response cache invalidated (%d entries dropped).", 0)


class RateLimiter:
    """Tiny in-memory token bucket (per key). Good enough for a demo API."""

    def __init__(self, max_calls: int = 1, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str = "default") -> bool:
        now = time.monotonic()
        with self.lock:
            bucket = [t for t in self.calls.get(key, []) if now - t < self.period]
            if len(bucket) >= self.max_calls:
                self.calls[key] = bucket
                return False
            bucket.append(now)
            self.calls[key] = bucket
            return True


def data_freshness(path: dt.datetime | float | None = None) -> tuple[dt.date | None, str]:
    """Return (last_date, freshness_label) from the canonical series."""
    try:
        df = load_canonical()
        last_date = pd.Timestamp(df["date"].iloc[-1]).date()
        mtime = dt.datetime.fromtimestamp(
            (settings.processed_dir / "gold_prices_daily.parquet").stat().st_mtime
        )
        age_days = (dt.datetime.now() - mtime).days
        if age_days > 4:
            return last_date, "stale"
        if dt.datetime.now() - mtime < dt.timedelta(minutes=30):
            return last_date, "fresh"
        return last_date, "cached"
    except Exception:  # noqa: BLE001 - health must not crash
        return None, "missing"


def is_stale() -> bool:
    _, freshness = data_freshness()
    return freshness == "stale"
