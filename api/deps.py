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

# Health bookkeeping for the scheduler + startup bootstrap.
last_refresh_ok = True
last_refresh_attempt: float | None = None
_bootstrap_thread: threading.Thread | None = None
_bootstrap_error: str | None = None


def get_history(metal: str = "gold") -> pd.DataFrame:
    if metal == "silver":
        from ingestion.fetch_silver_prices import load_canonical as load_silver

        return load_silver()
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


def service_is_loaded() -> bool:
    """Report model state WITHOUT triggering a load (safe for /health)."""
    return _service is not None and _service.is_loaded


def bootstrap_if_needed() -> threading.Thread | None:
    """Start a background thread that seeds data + loads the model once.

    Keeps the HTTP server responsive while the (potentially minutes-long)
    first-time model download/ingestion happens; /health reports progress.
    """
    global _bootstrap_thread
    with _service_lock:
        if _bootstrap_thread is not None:
            return None
        _bootstrap_thread = threading.Thread(
            target=_bootstrap, name="bootstrap", daemon=True
        )
        _bootstrap_thread.start()
        return _bootstrap_thread


def _bootstrap() -> None:
    global _bootstrap_error
    try:
        import datetime as dt

        from ingestion.fetch_gold_prices import load_canonical, update_canonical
        from ingestion.usdinr import fetch_usdinr_rate

        try:
            load_canonical()
        except FileNotFoundError:
            logger.info("No canonical data; seeding from yfinance...")
            update_canonical(
                dt.date.fromisoformat(settings.history_start), dt.date.today()
            )
        try:
            from ingestion.fetch_silver_prices import load_canonical as load_silver

            load_silver()
        except FileNotFoundError:
            try:
                from ingestion.fetch_silver_prices import update_canonical as update_silver

                update_silver(
                    dt.date.fromisoformat(settings.history_start), dt.date.today()
                )
            except Exception as ag_exc:  # noqa: BLE001 - gold view must not depend on silver
                logger.warning("Silver seeding failed: %s", ag_exc)
        # FX series for the INR view. Full-window request is cache-aware:
        # covered caches are skipped, short ones are self-healed.
        try:
            from ingestion.usdinr import fetch_usdinr_rate

            fetch_usdinr_rate(
                dt.date.fromisoformat(settings.history_start), dt.date.today()
            )
        except Exception as fx_exc:  # noqa: BLE001 - gold view must not depend on FX
            logger.warning("USDINR seeding failed: %s", fx_exc)
        try:
            from ingestion.india_rates import refresh_india_rates

            refresh_india_rates()
        except Exception as india_exc:  # noqa: BLE001 - retail view is auxiliary
            logger.warning("India retail rates seeding failed: %s", india_exc)
        try:
            from forecasting.accountability import log_daily_forecasts
            from ingestion.fetch_gold_prices import load_canonical
            from ingestion.premium_series import build_premium_series, premium_forecast_input

            log_daily_forecasts(get_service(), load_canonical())
            try:
                build_premium_series(settings.digest_city)
                premium_input = premium_forecast_input(settings.digest_city)
                if not premium_input.empty and len(premium_input) >= 32:
                    log_daily_forecasts(
                        get_service(),
                        premium_input,
                        series=f"premium:{settings.digest_city}:22k",
                    )
            except Exception as prem_exc:  # noqa: BLE001 - premium is auxiliary
                logger.warning("premium accountability logging failed: %s", prem_exc)
        except Exception as acc_exc:  # noqa: BLE001 - accountability is auxiliary
            logger.warning("accountability logging failed: %s", acc_exc)
        get_service()
    except Exception as exc:  # noqa: BLE001 - bootstrap failures surface via /health
        _bootstrap_error = str(exc)
        logger.exception("Startup bootstrap failed: %s", exc)


def bootstrap_error() -> str | None:
    return _bootstrap_error


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
