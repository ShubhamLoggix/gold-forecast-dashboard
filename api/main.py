"""Stage B backend: FastAPI gold forecast API.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import datetime as dt
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
    RefreshResponse,
)
from config import settings
from forecasting.baselines import naive_last_value, simple_moving_average
from forecasting.timesfm_service import HORIZON_PRESETS, GoldForecastService
from ingestion.aggregate import aggregate_history
from ingestion.fetch_gold_prices import update_canonical
from logging_setup import setup_logging

setup_logging(settings.log_level, json_logs=True)
logger = logging.getLogger("gold_forecast.api")

refresh_rate_limiter = deps.RateLimiter(max_calls=1, period_seconds=60.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request ID + error envelope middleware
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
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
    if not settings.api_key or not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _history_response(
    start: dt.date | None, end: dt.date | None, granularity: str
) -> HistoryResponse:
    df = deps.get_history()
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        raise HTTPException(status_code=404, detail="No data in requested range")
    agg = aggregate_history(df, granularity)
    return HistoryResponse(
        start=pd.Timestamp(agg["date"].iloc[0]).date(),
        end=pd.Timestamp(agg["date"].iloc[-1]).date(),
        granularity=granularity,  # type: ignore[arg-type]
        source=str(df["source"].iloc[-1]),
        points=[
            OhlcPoint(
                date=pd.Timestamp(r["date"]).date(),
                open=None if pd.isna(r["open"]) else float(r["open"]),
                high=None if pd.isna(r["high"]) else float(r["high"]),
                low=None if pd.isna(r["low"]) else float(r["low"]),
                close=float(r["close"]),
                volume=None if r["volume"] is None or pd.isna(r["volume"]) else float(r["volume"]),
            )
            for _, r in agg.iterrows()
        ],
    )


@app.get("/api/v1/history", response_model=HistoryResponse, tags=["data"])
def get_history(
    start: dt.date | None = Query(default=None),
    end: dt.date | None = Query(default=None),
    granularity: str = Query(default="day", pattern="^(day|week|month)$"),
    request: Request = None,  # type: ignore[assignment]
):
    key = ("history", start, end, granularity)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _history_response(start, end, granularity)
        deps.cache_set(key, cached)
    return cached


def _forecast_response(horizon: str, quantiles: bool) -> ForecastResponse:
    import datetime as dt_

    preset_days = HORIZON_PRESETS[horizon]
    service = deps.get_service()
    df = deps.get_history()
    result = service.forecast(df, preset_days, quantiles=quantiles)
    naive = naive_last_value(df, preset_days)
    sma = simple_moving_average(df, preset_days)
    return ForecastResponse(
        horizon=horizon,  # type: ignore[arg-type]
        horizon_days=preset_days,
        model_version=result.model_version,
        generated_at=dt_.datetime.now(dt_.timezone.utc),
        latency_ms=result.latency_ms,
        history_last_date=result.history_last_date,
        history_last_close=result.last_close,
        dates=result.dates,
        point=result.point.tolist(),
        q10=result.q10.tolist(),
        q50=result.q50.tolist(),
        q90=result.q90.tolist(),
        quantiles=quantiles,
        baselines=[
            BaselineSeries(name="naive-last-value", values=naive.tolist()),
            BaselineSeries(name="sma-20", values=sma.tolist()),
        ],
    )


@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["forecast"])
def get_forecast(
    horizon: str = Query(default="1m", pattern="^(1w|1m|3m|6m|1y)$"),
    quantiles: bool = Query(default=True),
):
    key = ("forecast", horizon, quantiles, settings.model_version)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _forecast_response(horizon, quantiles)
        deps.cache_set(key, cached)
    return cached


@app.get("/api/v1/backtest/latest", response_model=BacktestResponse, tags=["backtest"])
def get_latest_backtest():
    out_dir = Path(settings.data_dir) / "backtests"
    files = sorted(out_dir.glob("backtest_*.json"))
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No backtest report found. Run scripts/run_backtest.py first.",
        )
    import json

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


@app.get("/api/v1/health", response_model=HealthResponse, tags=["ops"])
def health():
    last_date, freshness = deps.data_freshness()
    try:
        model_loaded = deps.get_service().is_loaded
    except Exception:  # noqa: BLE001 - health must not crash on model failure
        model_loaded = False
    stale = freshness == "stale"
    return HealthResponse(
        status="degraded" if stale or not model_loaded else "ok",
        model_loaded=model_loaded,
        model_version=settings.model_version,
        data_last_date=last_date,
        data_freshness=freshness,
        data_may_be_stale=stale,
        last_refresh_ok=deps.last_refresh_ok,
        source_status={"canonical": freshness, "ingest_backends": "yfinance,lbma,metals_api"},
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
