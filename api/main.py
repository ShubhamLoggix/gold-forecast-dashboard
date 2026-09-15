"""Stage B backend: FastAPI gold forecast API.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import deps, scheduler as scheduler_mod
from api.schemas import (
    BacktestModelResult,
    BacktestResponse,
    BaselineSeries,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    OhlcPoint,
    RateInfo,
    RefreshResponse,
)
from config import settings
from conversion import PURITY_FACTORS, TROY_OZ_TO_GRAM, UNIT_FACTORS, convert_series
from forecasting.baselines import naive_last_value, simple_moving_average
from forecasting.timesfm_service import HORIZON_PRESETS, GoldForecastService
from ingestion.aggregate import aggregate_history
from ingestion.fetch_gold_prices import update_canonical
from logging_setup import setup_logging

setup_logging(settings.log_level, json_logs=True)
logger = logging.getLogger("gold_forecast.api")

refresh_rate_limiter = deps.RateLimiter(max_calls=1, period_seconds=60.0)
# Public read endpoints: generous for a dashboard, strict enough to stop abuse.
public_rate_limiter = deps.RateLimiter(max_calls=60, period_seconds=60.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed data + warm the model in the background so the HTTP server (and the
    # healthcheck) stay responsive while the first-time download happens.
    deps.bootstrap_if_needed()
    if settings.enable_scheduler:
        scheduler_mod.start()
    try:
        yield
    finally:
        scheduler_mod.stop()


app = FastAPI(
    title="Gold Price Forecast API (TimesFM)",
    version="1.0.0",
    description=(
        "Forecasts COMEX gold prices (GC=F) with TimesFM 2.5 (Apache-2.0) and compares "
        "against naive baselines. NOT investment advice; forecasts are experimental and "
        "financial series are near random walks."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request ID + error envelope middleware
# --------------------------------------------------------------------------- #
MAX_BODY_BYTES = 1024 * 1024  # 1 MiB — endpoints take no large payloads


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
        return _error_response(request, 413, "payload_too_large",
                               "Request body exceeds the 1 MiB limit.")
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return _error_response(request, exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return _error_response(request, 422, "validation_error", str(exc.errors()[:3]))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    logger.exception("Unhandled error (request_id=%s): %s", request_id, exc)
    return _error_response(request, 500, "internal_error", "Internal server error.")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # Constant-time comparison to prevent timing side channels.
    if (
        not settings.api_key
        or not x_api_key
        or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode())
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _rate_limit_public(request: Request, bucket: str) -> None:
    """Per-client-IP token bucket on public GET endpoints."""
    client = request.client.host if request.client else "unknown"
    if not public_rate_limiter.allow(f"{client}:{bucket}"):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {bucket}. Slow down and retry shortly.",
        )


def _rate_model(rate_info: dict) -> RateInfo:
    """Map ingestion rate dict -> RateInfo schema."""
    return RateInfo(
        usd_inr_rate=rate_info["rate"],
        usd_inr_rate_date=rate_info["rate_date"],
        rate_may_be_stale=rate_info["rate_may_be_stale"],
    )


def _resolve_rate_info(reference_date: dt.date | None = None) -> dict:
    """Latest USDINR rate for conversion (raises 503 if the series is absent)."""
    from ingestion.usdinr import latest_rate

    try:
        return latest_rate(reference_date=reference_date)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "USDINR rate series not available yet — it refreshes with the daily "
                "ingestion job, or run fetch_usdinr_rate() manually."
            ),
        ) from exc


def _convert_values(values, rate: float, karat: str, unit: str):
    from conversion import convert_series

    return convert_series(values, rate, karat, unit).tolist()


def _history_response(
    start: dt.date | None,
    end: dt.date | None,
    granularity: str,
    currency: str = "usd",
    karat: str = "24k",
    unit: str = "10gram",
) -> HistoryResponse:
    df = deps.get_history()
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        raise HTTPException(status_code=404, detail="No data in requested range")
    agg = aggregate_history(df, granularity)

    rate_info = None
    if currency == "inr":
        # Per-date conversion: each historical point uses the USD/INR rate of
        # its own date (most recent prior available rate; never interpolated).
        # The LATEST rate (returned in `rate`) applies to the most recent point
        # and is what the frontend captions as "converted at ₹X/USD as of DATE".
        from ingestion.usdinr import rate_frame_for_dates

        rate_info = _resolve_rate_info(reference_date=pd.Timestamp(df["date"].iloc[-1]).date())
        fx = rate_frame_for_dates(agg["date"])
        factor = (
            fx["usd_inr_rate"].astype(float)
            / TROY_OZ_TO_GRAM
            * PURITY_FACTORS[karat]
            * UNIT_FACTORS[unit]
        ).to_numpy()
        for col in ("open", "high", "low", "close"):
            agg[col] = agg[col].astype(float) * factor
        # volume is unit-agnostic and stays untouched
    points = []
    for _, r in agg.iterrows():
        points.append(
            OhlcPoint(
                date=pd.Timestamp(r["date"]).date(),
                open=None if pd.isna(r["open"]) else float(r["open"]),
                high=None if pd.isna(r["high"]) else float(r["high"]),
                low=None if pd.isna(r["low"]) else float(r["low"]),
                close=float(r["close"]),
                volume=None if r["volume"] is None or pd.isna(r["volume"]) else float(r["volume"]),
            )
        )
    return HistoryResponse(
        start=pd.Timestamp(agg["date"].iloc[0]).date(),
        end=pd.Timestamp(agg["date"].iloc[-1]).date(),
        granularity=granularity,  # type: ignore[arg-type]
        source=str(df["source"].iloc[-1]),
        points=points,
        currency=currency,  # type: ignore[arg-type]
        karat=karat if currency == "inr" else None,  # type: ignore[arg-type]
        unit=unit if currency == "inr" else None,  # type: ignore[arg-type]
        rate=_rate_model(rate_info) if rate_info else None,
    )


@app.get("/api/v1/history", response_model=HistoryResponse, tags=["data"])
def get_history(
    start: dt.date | None = Query(default=None),
    end: dt.date | None = Query(default=None),
    granularity: str = Query(default="day", pattern="^(day|week|month)$"),
    currency: str = Query(default="usd", pattern="^(usd|inr)$"),
    karat: str = Query(default="24k", pattern="^(24k|22k|18k)$"),
    unit: str = Query(default="10gram", pattern="^(gram|10gram)$"),
    request: Request = None,  # type: ignore[assignment]
):
    _rate_limit_public(request, "history")
    key = ("history", start, end, granularity, currency, karat, unit)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _history_response(start, end, granularity, currency, karat, unit)
        deps.cache_set(key, cached)
    return cached


def _forecast_response(
    horizon: str, quantiles: bool, currency: str = "usd", karat: str = "24k",
    unit: str = "10gram",
) -> ForecastResponse:
    import datetime as dt_

    preset_days = HORIZON_PRESETS[horizon]
    service = deps.get_service()
    df = deps.get_history()
    result = service.forecast(df, preset_days, quantiles=quantiles)
    naive = naive_last_value(df, preset_days)
    sma = simple_moving_average(df, preset_days)

    rate_info = None
    point, q10, q50, q90 = (
        result.point, result.q10, result.q50, result.q90,
    )
    if currency == "inr":
        rate_info = _resolve_rate_info(reference_date=result.history_last_date)
        rate = rate_info["rate"]
        point = convert_series(point, rate, karat, unit)
        q10 = convert_series(q10, rate, karat, unit)
        q50 = convert_series(q50, rate, karat, unit)
        q90 = convert_series(q90, rate, karat, unit)
        naive = convert_series(naive, rate, karat, unit)
        sma = convert_series(sma, rate, karat, unit)
        history_last_close = float(result.last_close * rate
                                    / TROY_OZ_TO_GRAM * PURITY_FACTORS[karat]
                                    * UNIT_FACTORS[unit])
    else:
        history_last_close = result.last_close
    return ForecastResponse(
        horizon=horizon,  # type: ignore[arg-type]
        horizon_days=preset_days,
        model_version=result.model_version,
        generated_at=dt_.datetime.now(dt_.timezone.utc),
        latency_ms=result.latency_ms,
        history_last_date=result.history_last_date,
        history_last_close=history_last_close,
        dates=result.dates,
        point=point.tolist(),
        q10=q10.tolist(),
        q50=q50.tolist(),
        q90=q90.tolist(),
        quantiles=quantiles,
        baselines=[
            BaselineSeries(name="naive-last-value", values=naive.tolist()),
            BaselineSeries(name="sma-20", values=sma.tolist()),
        ],
        currency=currency,  # type: ignore[arg-type]
        karat=karat if currency == "inr" else None,  # type: ignore[arg-type]
        unit=unit if currency == "inr" else None,  # type: ignore[arg-type]
        rate=_rate_model(rate_info) if rate_info else None,
    )


@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["forecast"])
def get_forecast(
    horizon: str = Query(default="1m", pattern="^(1w|1m|3m|6m|1y)$"),
    quantiles: bool = Query(default=True),
    currency: str = Query(default="usd", pattern="^(usd|inr)$"),
    karat: str = Query(default="24k", pattern="^(24k|22k|18k)$"),
    unit: str = Query(default="10gram", pattern="^(gram|10gram)$"),
    request: Request = None,  # type: ignore[assignment]
):
    _rate_limit_public(request, "forecast")
    key = ("forecast", horizon, quantiles, settings.model_version, currency, karat, unit)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _forecast_response(horizon, quantiles, currency, karat, unit)
        deps.cache_set(key, cached)
    return cached


@app.get("/api/v1/backtest/latest", response_model=BacktestResponse, tags=["backtest"])
def get_latest_backtest(request: Request = None):  # type: ignore[assignment]
    _rate_limit_public(request, "backtest")
    out_dir = Path(settings.data_dir) / "backtests"
    files = sorted(out_dir.glob("backtest_*.json"))
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No backtest report found. Run scripts/run_backtest.py first.",
        )
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    return BacktestResponse(
        generated_at=payload["generated_at"],
        model_version=payload["model_version"],
        context_length=payload["context_length"],
        results={
            name: BacktestModelResult(
                model=res["model"],
                model_version=res["model_version"],
                horizon_days=res["horizon_days"],
                step_days=res["step_days"],
                summary=res["summary"],
                folds=res["folds"],
            )
            for name, res in payload["results"].items()
        },
    )


def _latest_drift() -> dict | None:
    """Model-drift watchdog block from the most recent backtest report."""
    files = sorted((Path(settings.data_dir) / "backtests").glob("backtest_*.json"))
    if not files:
        return None
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        return payload.get("model_drift")
    except Exception:  # noqa: BLE001 - health must not crash on a bad report
        return None


@app.get("/api/v1/health", response_model=HealthResponse, tags=["ops"])
def health():
    last_date, freshness = deps.data_freshness()
    # NOTE: deliberately does NOT call get_service() — a health probe must not
    # trigger a multi-minute model download/load.
    model_loaded = deps.service_is_loaded()
    stale = freshness == "stale"
    drift = _latest_drift()
    underperforming = bool(drift and drift.get("underperforming"))
    bootstrap_err = deps.bootstrap_error()
    degraded = stale or not model_loaded or bool(bootstrap_err) or underperforming
    return HealthResponse(
        status="degraded" if degraded else "ok",
        model_loaded=model_loaded,
        model_version=settings.model_version,
        data_last_date=last_date,
        data_freshness=freshness,
        data_may_be_stale=stale,
        last_refresh_ok=deps.last_refresh_ok,
        model_underperforming_baseline=underperforming,
        bootstrap_error=bootstrap_err,
        source_status={
            "canonical": freshness,
            "ingest_backends": "yfinance,lbma,metals_api",
            "drift_delta_pp": (
                str(drift.get("delta_pp")) if drift else "no-backtest-report"
            ),
        },
    )


@app.post("/api/v1/refresh", response_model=RefreshResponse, tags=["ops"])
def refresh(
    force: bool = Query(default=False),
    _key: None = Depends(_require_api_key),
):
    if not refresh_rate_limiter.allow("refresh"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit: at most one refresh per minute. Try again shortly.",
        )
    deps.last_refresh_attempt = dt.datetime.now(dt.timezone.utc).timestamp()
    df = update_canonical(
        dt.date.fromisoformat(settings.history_start), dt.date.today(),
        force_refresh=force,
    )
    deps.invalidate_response_cache()
    last_date = pd.Timestamp(df["date"].iloc[-1]).date() if len(df) else None
    deps.last_refresh_ok = True
    logger.info("On-demand refresh complete (rows=%d last=%s).", len(df), last_date)
    return RefreshResponse(status="ok", rows=len(df), last_date=last_date,
                           cache_invalidated=True)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, log_config=None)
