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
    AccountabilityHorizon,
    AccountabilityPoint,
    AccountabilityResponse,
    AlertCreatedResponse,
    AlertDeletedResponse,
    AlertTarget,
    AlertTargetIn,
    AlertTargetsResponse,
    BacktestModelResult,
    BacktestResponse,
    BacktestScoreEntry,
    BacktestScoreboard,
    BakeoffEntry,
    BakeoffResponse,
    BandCalibration,
    BaselineSeries,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    IndiaCitiesResponse,
    IndiaCity,
    IndiaForecastResponse,
    IndiaRatePoint,
    IndiaRatesResponse,
    OhlcPoint,
    RateInfo,
    RefreshResponse,
)
from config import settings
from conversion import (
    PURITY_FACTORS,
    TROY_OZ_TO_GRAM,
    UNIT_FACTORS,
    convert_silver_series,
    convert_series,
)
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


def _validate_quality(metal: str, karat: str, fineness: str, unit: str) -> tuple[str, str, str]:
    """Unit/fineness validation across metals (raises 422 on bad combos)."""
    if metal == "silver":
        if unit not in ("kg", "gram"):
            raise HTTPException(
                status_code=422, detail="unit must be 'kg' or 'gram' for silver"
            )
        if karat != "24k":  # karat is gold-only; reject non-default usage
            raise HTTPException(
                status_code=422, detail="karat applies to gold only — use fineness for silver"
            )
        return "", fineness, unit
    if unit not in ("gram", "10gram"):
        raise HTTPException(
            status_code=422, detail="unit must be 'gram' or '10gram' for gold"
        )
    if fineness != "999":
        raise HTTPException(
            status_code=422, detail="fineness applies to silver only — use karat for gold"
        )
    return karat, "", unit


def _history_response(
    start: dt.date | None,
    end: dt.date | None,
    granularity: str,
    currency: str = "usd",
    karat: str = "24k",
    unit: str = "10gram",
    metal: str = "gold",
    fineness: str = "999",
) -> HistoryResponse:
    karat, fineness, unit = _validate_quality(metal, karat, fineness, unit)
    df = deps.get_history(metal)
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
        if metal == "silver":
            from conversion import SILVER_FINENESS, SILVER_UNIT_FACTORS

            factor = (
                fx["usd_inr_rate"].astype(float)
                / TROY_OZ_TO_GRAM
                * SILVER_FINENESS[fineness]
                * SILVER_UNIT_FACTORS[unit]
            ).to_numpy()
        else:
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
        metal=metal,  # type: ignore[arg-type]
        karat=karat if (currency == "inr" and metal == "gold") else None,  # type: ignore[arg-type]
        fineness=fineness if (currency == "inr" and metal == "silver") else None,  # type: ignore[arg-type]
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
    unit: str = Query(default="10gram", pattern="^(gram|10gram|kg)$"),
    metal: str = Query(default="gold", pattern="^(gold|silver)$"),
    fineness: str = Query(default="999", pattern="^(999|958|925)$"),
    request: Request = None,  # type: ignore[assignment]
):
    _rate_limit_public(request, "history")
    key = ("history", start, end, granularity, currency, karat, unit, metal, fineness)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _history_response(
            start, end, granularity, currency, karat, unit, metal, fineness
        )
        deps.cache_set(key, cached)
    return cached


def _band_scale(horizon_days: int) -> float | None:
    """Calibrated band multiplier for a horizon, from the latest backtest that
    computed one (None when no calibration data exists yet)."""
    out_dir = Path(settings.data_dir) / "backtests"
    for path in sorted(out_dir.glob("backtest_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for res in (payload.get("results") or {}).values():
            if (
                res.get("horizon_days") == horizon_days
                and res.get("band_scale") is not None
            ):
                scale = float(res["band_scale"])
                if scale > 0:
                    return scale
    return None


def _forecast_response(
    horizon: str, quantiles: bool, currency: str = "usd", karat: str = "24k",
    unit: str = "10gram", metal: str = "gold", fineness: str = "999",
) -> ForecastResponse:
    import datetime as dt_

    karat, fineness, unit = _validate_quality(metal, karat, fineness, unit)
    preset_days = HORIZON_PRESETS[horizon]
    service = deps.get_service()
    df = deps.get_history(metal)
    result = service.forecast(df, preset_days, quantiles=quantiles)
    naive = naive_last_value(df, preset_days)
    sma = simple_moving_average(df, preset_days)

    band_calibration = None
    scale = _band_scale(preset_days)
    if scale and scale != 1.0:
        result.q10 = result.point - scale * (result.point - result.q10)
        result.q90 = result.point + scale * (result.q90 - result.point)
        band_calibration = BandCalibration(
            scale=round(scale, 3), target_coverage_pct=80.0
        )

    rate_info = None
    point, q10, q50, q90 = (
        result.point, result.q10, result.q50, result.q90,
    )
    if currency == "inr":
        rate_info = _resolve_rate_info(reference_date=result.history_last_date)
        rate = rate_info["rate"]
        if metal == "silver":
            from conversion import SILVER_FINENESS, SILVER_UNIT_FACTORS

            def _conv(values):
                return convert_silver_series(values, rate, fineness, unit)

            history_last_close = float(
                result.last_close * rate
                / TROY_OZ_TO_GRAM * SILVER_FINENESS[fineness]
                * SILVER_UNIT_FACTORS[unit]
            )
        else:
            def _conv(values):
                return convert_series(values, rate, karat, unit)

            history_last_close = float(result.last_close * rate
                                        / TROY_OZ_TO_GRAM * PURITY_FACTORS[karat]
                                        * UNIT_FACTORS[unit])
        point = _conv(point)
        q10 = _conv(q10)
        q50 = _conv(q50)
        q90 = _conv(q90)
        naive = _conv(naive)
        sma = _conv(sma)
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
        metal=metal,  # type: ignore[arg-type]
        karat=karat if (currency == "inr" and metal == "gold") else None,  # type: ignore[arg-type]
        fineness=fineness if (currency == "inr" and metal == "silver") else None,  # type: ignore[arg-type]
        unit=unit if currency == "inr" else None,  # type: ignore[arg-type]
        rate=_rate_model(rate_info) if rate_info else None,
        band_calibration=band_calibration,
    )


@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["forecast"])
def get_forecast(
    horizon: str = Query(default="1m", pattern="^(1w|1m|3m|6m|1y)$"),
    quantiles: bool = Query(default=True),
    currency: str = Query(default="usd", pattern="^(usd|inr)$"),
    karat: str = Query(default="24k", pattern="^(24k|22k|18k)$"),
    unit: str = Query(default="10gram", pattern="^(gram|10gram|kg)$"),
    metal: str = Query(default="gold", pattern="^(gold|silver)$"),
    fineness: str = Query(default="999", pattern="^(999|958|925)$"),
    request: Request = None,  # type: ignore[assignment]
):
    _rate_limit_public(request, "forecast")
    key = ("forecast", horizon, quantiles, settings.model_version, currency,
           karat, unit, metal, fineness)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _forecast_response(
            horizon, quantiles, currency, karat, unit, metal, fineness
        )
        deps.cache_set(key, cached)
    return cached


def _india_cities_response() -> IndiaCitiesResponse:
    from ingestion.india_rates import cities_catalog

    try:
        catalog = cities_catalog()
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=502, detail=f"India retail rates source unavailable: {exc}"
        ) from exc
    if not catalog:
        raise HTTPException(
            status_code=502, detail="India retail rates source returned no cities"
        )
    cities = sorted((IndiaCity(**item) for item in catalog), key=lambda c: c.name.lower())
    return IndiaCitiesResponse(cities=cities)


@app.get("/api/v1/india/cities", response_model=IndiaCitiesResponse, tags=["india"])
def list_india_cities(request: Request = None):  # type: ignore[assignment]
    _rate_limit_public(request, "india_cities")
    key = ("india_cities",)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _india_cities_response()
        deps.cache_set(key, cached)
    return cached


def _india_rates_response(city: str, unit: str) -> IndiaRatesResponse:
    from ingestion.india_rates import get_city_rates

    try:
        data = get_city_rates(city)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city!r}") from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=502, detail=f"India retail rates source unavailable: {exc}"
        ) from exc
    factor = UNIT_FACTORS[unit]
    per_gram = data["per_gram"]
    return IndiaRatesResponse(
        city_slug=data["city_slug"],
        city=data["city"],
        date=data["date"],
        per_gram=per_gram,
        per_10g={k: v * factor for k, v in per_gram.items()},
        pct_change=data["pct_change"],
        history=[IndiaRatePoint(**point) for point in data["history"]],
        source="groww:gold-rates",
    )


@app.get("/api/v1/india/rates", response_model=IndiaRatesResponse, tags=["india"])
def get_india_rates(
    city: str = Query(default="pune", pattern="^[a-z0-9-]{1,80}$"),
    unit: str = Query(default="10gram", pattern="^(gram|10gram)$"),
    request: Request = None,  # type: ignore[assignment]
):
    _rate_limit_public(request, "india_rates")
    key = ("india_rates", city, unit)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _india_rates_response(city, unit)
        deps.cache_set(key, cached)
    return cached


def _india_forecast_response(city: str, horizon: str, karat: str, unit: str) -> IndiaForecastResponse:
    """Estimated RETAIL forecast: TimesFM bullion forecast (INR, karat, unit)
    scaled by the city's current retail premium ratio."""
    from ingestion.india_rates import get_city_rates

    bullion = _forecast_response(horizon, True, currency="inr", karat=karat, unit=unit)
    try:
        data = get_city_rates(city)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city!r}") from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=502, detail=f"India retail rates source unavailable: {exc}"
        ) from exc

    retail_value = float(data["per_gram"][karat]) * UNIT_FACTORS[unit]
    bullion_value = float(bullion.history_last_close)
    if bullion_value <= 0:
        raise HTTPException(status_code=502, detail="Bullion series unavailable for premium scaling")
    ratio = retail_value / bullion_value

    def _scale(values: list[float]) -> list[float]:
        return [v * ratio for v in values]

    return IndiaForecastResponse(
        horizon=bullion.horizon,
        horizon_days=bullion.horizon_days,
        model_version=bullion.model_version,
        generated_at=bullion.generated_at,
        latency_ms=bullion.latency_ms,
        history_last_date=bullion.history_last_date,
        history_last_close=retail_value,
        dates=bullion.dates,
        point=_scale(bullion.point),
        q10=_scale(bullion.q10),
        q50=_scale(bullion.q50),
        q90=_scale(bullion.q90),
        quantiles=True,
        baselines=[
            BaselineSeries(name=b.name, values=_scale(b.values))
            for b in bullion.baselines
        ],
        currency="inr",
        karat=karat,  # type: ignore[arg-type]
        unit=unit,  # type: ignore[arg-type]
        rate=bullion.rate,
        band_calibration=bullion.band_calibration,
        premium_ratio=ratio,
        bullion_history_last_close=bullion_value,
    )


@app.get("/api/v1/india/forecast", response_model=IndiaForecastResponse, tags=["india"])
def get_india_forecast(
    city: str = Query(default="pune", pattern="^[a-z0-9-]{1,80}$"),
    horizon: str = Query(default="1m", pattern="^(1w|1m|3m|6m|1y)$"),
    karat: str = Query(default="24k", pattern="^(24k|22k|18k)$"),
    unit: str = Query(default="10gram", pattern="^(gram|10gram)$"),
    request: Request = None,  # type: ignore[assignment]
):
    _rate_limit_public(request, "india_forecast")
    key = ("india_forecast", city, horizon, karat, unit, settings.model_version)
    cached = deps.cache_get(key)
    if cached is None:
        cached = _india_forecast_response(city, horizon, karat, unit)
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


@app.get("/api/v1/backtest/scoreboard", response_model=BacktestScoreboard, tags=["backtest"])
def backtest_scoreboard(request: Request = None):  # type: ignore[assignment]
    """Per-horizon measured accuracy across all persisted backtest reports.

    This is the "how much can you trust the model" data: each entry is the
    average over real out-of-sample walk-forward folds.
    """
    _rate_limit_public(request, "scoreboard")
    out_dir = Path(settings.data_dir) / "backtests"
    files = sorted(out_dir.glob("backtest_*.json"), reverse=True)
    entries: list[BacktestScoreEntry] = []
    seen: set[int] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip unreadable reports
            continue
        results = payload.get("results") or {}
        tm = next((r for n, r in results.items() if n.startswith("timesfm-")), None)
        nv = results.get("naive-last-value")
        if not tm or not tm.get("summary"):
            continue
        summary = tm["summary"]
        horizon = int(summary.get("horizon_days") or tm.get("horizon_days") or 0)
        if not horizon or horizon in seen:
            continue
        seen.add(horizon)
        dir_acc = float(summary.get("directional_accuracy_pct", 0.0))
        nv_acc = None
        if nv and nv.get("summary", {}).get("directional_accuracy_pct") is not None:
            nv_acc = float(nv["summary"]["directional_accuracy_pct"])
        entries.append(
            BacktestScoreEntry(
                horizon_days=horizon,
                generated_at=dt.datetime.fromisoformat(payload["generated_at"]),
                model_version=str(payload.get("model_version")),
                mape_pct=float(summary.get("mape_pct", 0.0)),
                directional_accuracy_pct=dir_acc,
                naive_directional_accuracy_pct=nv_acc,
                skill_vs_naive_pp=(
                    round(dir_acc - nv_acc, 2) if nv_acc is not None else None
                ),
                band_coverage_pct=(
                    float(summary["band_coverage_pct"])
                    if summary.get("band_coverage_pct") is not None
                    else None
                ),
                n_folds=int(summary.get("n_folds", 0)),
                underperforming_naive=bool(
                    nv_acc is not None and dir_acc < nv_acc
                ),
            )
        )
    entries.sort(key=lambda e: e.horizon_days)
    return BacktestScoreboard(entries=entries)


@app.get("/api/v1/backtest/bakeoff", response_model=BakeoffResponse, tags=["backtest"])
def backtest_bakeoff(request: Request = None):  # type: ignore[assignment]
    """Multi-model leaderboard: TimesFM vs Chronos vs naive vs SMA."""
    _rate_limit_public(request, "bakeoff")
    out_dir = Path(settings.data_dir) / "backtests"
    files = sorted(out_dir.glob("bakeoff_*.json"), reverse=True)
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No bake-off report found. Run scripts/model_bakeoff.py first.",
        )
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    return BakeoffResponse(
        generated_at=dt.datetime.fromisoformat(payload["generated_at"]),
        data_range=payload["data_range"],
        entries=[BakeoffEntry(**e) for e in payload["entries"]],
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


@app.get("/api/v1/accountability", response_model=AccountabilityResponse, tags=["accountability"])
def get_accountability(request: Request = None):  # type: ignore[assignment]
    """Realized track record of logged forecasts (predicted vs actual)."""
    _rate_limit_public(request, "accountability")
    from forecasting.accountability import score_forecasts

    try:
        payload = score_forecasts(deps.get_history())
    except FileNotFoundError:
        return AccountabilityResponse(per_horizon=[], pending_counts={}, recent=[])
    except Exception as exc:  # noqa: BLE001 - report must not break the API
        raise HTTPException(status_code=502, detail=f"accountability scoring failed: {exc}") from exc
    return AccountabilityResponse(
        per_horizon=[
            AccountabilityHorizon(**e) for e in payload["per_horizon"]
        ],
        pending_counts={str(k): v for k, v in payload["pending_counts"].items()},
        recent=[AccountabilityPoint(**p) for p in payload["recent"]],
    )


def _targets_response() -> AlertTargetsResponse:
    from alerts.telegram import list_targets

    return AlertTargetsResponse(
        targets=[AlertTarget(**t) for t in list_targets()],
        alerts_enabled=settings.alerts_enabled,
    )


@app.get("/api/v1/alerts/targets", response_model=AlertTargetsResponse, tags=["alerts"])
def get_alert_targets(request: Request = None):  # type: ignore[assignment]
    _rate_limit_public(request, "alert_targets")
    return _targets_response()


@app.post("/api/v1/alerts/targets", response_model=AlertCreatedResponse, tags=["alerts"])
def create_alert_target(
    target: AlertTargetIn,
    _key: None = Depends(_require_api_key),
):
    from alerts.telegram import add_target

    created = add_target(
        karat=target.karat, unit=target.unit, currency=target.currency,
        op=target.op, price=target.price,
    )
    return AlertCreatedResponse(created=AlertTarget(**created))


@app.delete("/api/v1/alerts/targets/{target_id}", response_model=AlertDeletedResponse, tags=["alerts"])
def delete_alert_target(
    target_id: str,
    _key: None = Depends(_require_api_key),
):
    from alerts.telegram import remove_target

    deleted = remove_target(target_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown alert target: {target_id}")
    return AlertDeletedResponse(deleted=True)


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
    try:
        from ingestion.india_rates import refresh_india_rates

        rows_india = refresh_india_rates()
        logger.info("India retail rates refreshed (rows=%d).", rows_india)
    except Exception as india_exc:  # noqa: BLE001 - retail view is auxiliary
        logger.warning("India retail rate refresh failed: %s", india_exc)
    return RefreshResponse(status="ok", rows=len(df), last_date=last_date,
                           cache_invalidated=True)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, log_config=None)
