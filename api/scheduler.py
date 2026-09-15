"""APScheduler wiring: daily data refresh after market close, weekly backtest.

All jobs are wrapped so a failure logs and flips the health flag instead of
crashing the API; the API keeps serving last-known-good cached data.
"""

from __future__ import annotations

import datetime as dt
import logging
import traceback

from config import settings
from logging_setup import setup_logging

setup_logging(settings.log_level, json_logs=True)
logger = logging.getLogger("gold_forecast.scheduler")

_scheduler = None


def start() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("APScheduler not installed; scheduling disabled.")
        return

    _scheduler = AsyncIOScheduler(timezone="UTC")
    _scheduler.add_job(
        daily_refresh_job,
        CronTrigger(hour=22, minute=10, day_of_week="mon-fri"),
        id="daily_data_refresh",
        replace_existing=True,
    )
    _scheduler.add_job(
        weekly_backtest_job,
        CronTrigger(day_of_week="mon", hour=6, minute=0),
        id="weekly_backtest",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Scheduler started (daily refresh 22:10 UTC, weekly backtest Mon 00:00 UTC).")


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped.")


def run_now() -> None:
    if _scheduler is not None:
        _scheduler.get_job("daily_data_refresh").modify(next_run_time=dt.datetime.now(dt.timezone.utc))


def _refresh_india_rates_job() -> None:
    """City-wise Indian retail rates (Groww). Auxiliary: failures never flip
    the main health flag."""
    try:
        from ingestion.india_rates import refresh_india_rates

        rows = refresh_india_rates()
        logger.info("India retail rate refresh done (rows=%d).", rows)
    except Exception:  # noqa: BLE001
        logger.warning("India retail rate refresh FAILED:\n%s", traceback.format_exc())


def daily_refresh_job() -> None:
    logger.info("Scheduled data refresh starting...")
    try:
        import api.deps as deps

        from ingestion.fetch_gold_prices import update_canonical
        from ingestion.usdinr import fetch_usdinr_rate

        df = update_canonical(
            dt.date.fromisoformat(settings.history_start), dt.date.today(),
            force_refresh=False,
        )
        # Refresh the USD/INR FX series on the same schedule. Full-window fetch
        # if the canonical file doesn't exist yet, else a recent top-up.
        fx_start = dt.date.today() - dt.timedelta(days=14)
        try:
            from ingestion.usdinr import load_usdinr

            load_usdinr()
        except FileNotFoundError:
            fx_start = dt.date.fromisoformat(settings.history_start)
        fetch_usdinr_rate(fx_start, dt.date.today())
        deps.invalidate_response_cache()
        deps.last_refresh_ok = True
        logger.info("Scheduled data refresh done (gold rows=%d).", len(df))
    except Exception:  # noqa: BLE001 - scheduled jobs must not crash the API
        deps.last_refresh_ok = False
        logger.error("Scheduled data refresh FAILED:\n%s", traceback.format_exc())
    _refresh_india_rates_job()


def weekly_backtest_job() -> None:
    logger.info("Scheduled weekly backtest starting...")
    try:
        from api import deps

        from forecasting.timesfm_service import GoldForecastService
        from ingestion.fetch_gold_prices import load_canonical

        history = load_canonical()
        service = GoldForecastService(
            model_version=settings.model_version, context_length=settings.context_length
        )
        service.load_model()
        results = service.backtest(history, horizon_days=30, step_days=7, max_folds=26)
        service.persist_backtest(results)
        deps.invalidate_response_cache()
        logger.info("Scheduled weekly backtest done (30d).")
        # Long-horizon trust score (overlapping quarterly folds).
        try:
            results_1y = service.backtest(
                history, horizon_days=252, step_days=63, max_folds=8
            )
            service.persist_backtest(results_1y, suffix="-1y")
            logger.info("Scheduled weekly backtest done (1y).")
        except Exception:  # noqa: BLE001 - 1y score is supplementary
            logger.warning("1y backtest FAILED:\n%s", traceback.format_exc())
    except Exception:  # noqa: BLE001
        logger.error("Scheduled weekly backtest FAILED:\n%s", traceback.format_exc())
