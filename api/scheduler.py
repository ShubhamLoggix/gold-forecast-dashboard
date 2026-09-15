"""APScheduler wiring: daily data refresh after market close, weekly backtest.

All jobs are wrapped so a failure logs and flips the health flag instead of
crashing the API; the API keeps serving last-known-good cached data.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import traceback
from pathlib import Path

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
    _accountability_and_alerts_job()


def _accountability_and_alerts_job() -> None:
    try:
        import api.deps as deps
        from forecasting.accountability import log_daily_forecasts
        from ingestion.fetch_gold_prices import load_canonical

        history = load_canonical()
        log_daily_forecasts(deps.get_service(), history)
    except Exception:  # noqa: BLE001
        logger.warning("accountability logging FAILED:\n%s", traceback.format_exc())
    _send_digest_and_check_alerts()


def _latest_drift_report() -> dict | None:
    try:
        files = sorted((Path(settings.data_dir) / "backtests").glob("backtest_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text(encoding="utf-8")).get("model_drift")
    except Exception:  # noqa: BLE001
        return None


def _send_digest_and_check_alerts() -> None:
    if not settings.alerts_enabled:
        return
    try:
        import api.deps as deps
        from alerts.telegram import daily_digest, evaluate_targets, list_targets
        from conversion import convert_series
        from forecasting.timesfm_service import HORIZON_PRESETS
        from ingestion.fetch_gold_prices import load_canonical
        from ingestion.india_rates import get_city_rates
        from ingestion.usdinr import latest_rate

        history = load_canonical()
        rate = latest_rate()
        service = deps.get_service()
        city = settings.digest_city

        try:
            retail_data = get_city_rates(city)
        except Exception:  # noqa: BLE001 - retail line is optional
            retail_data = None

        bullion = service.forecast(history, HORIZON_PRESETS["1m"], quantiles=False)
        rate_val = rate["rate"]
        bullion_22k_10g_today = float(
            convert_series(float(history["close"].iloc[-1]), rate_val, "22k", "10gram")
        )
        est_22k_10g = float(
            convert_series(float(bullion.point[-1]), rate_val, "22k", "10gram")
        )
        ratio = None
        pct = None
        if retail_data and bullion_22k_10g_today > 0:
            ratio = retail_data["per_10g"]["22k"] / bullion_22k_10g_today
            est_22k_10g *= ratio
            pct = (est_22k_10g / retail_data["per_10g"]["22k"] - 1) * 100
        forecast_info = {
            "horizon": "1m",
            "median_inr_22k_10g": est_22k_10g,
            "pct": pct,
            "premium_ratio": ratio,
        }
        active = len(
            [t for t in list_targets() if not t.get("triggered_at")]
        )
        daily_digest(
            retail=retail_data,
            usd_inr_rate=rate["rate"],
            usd_inr_rate_date=rate["rate_date"],
            forecast_info=forecast_info,
            drift=_latest_drift_report(),
            active_targets=active,
        )
        evaluate_targets(history, rate["rate"])
    except Exception:  # noqa: BLE001
        logger.warning("daily digest / alert check FAILED:\n%s", traceback.format_exc())


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
