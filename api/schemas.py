"""Pydantic request/response schemas for the gold forecast API."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

Granularity = Literal["day", "week", "month"]
HorizonPreset = Literal["1w", "1m", "3m", "6m", "1y"]
Currency = Literal["usd", "inr"]
Karat = Literal["24k", "22k", "18k"]
Unit = Literal["gram", "10gram"]


class OhlcPoint(BaseModel):
    date: dt.date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None


class RateInfo(BaseModel):
    """Transparency metadata for currency conversion (present when currency=inr)."""

    usd_inr_rate: float
    usd_inr_rate_date: dt.date
    rate_may_be_stale: bool
    # Theoretical bullion-equivalent conversion; NOT Indian retail pricing.
    # Historical points use each date's own USD/INR rate (per-date, backward
    # as-of); the latest rate applies to the forecast segment.
    disclaimer: str = (
        "Converted from COMEX USD futures — theoretical bullion-equivalent "
        "price, not an Indian retail/jeweler quote (which also includes import "
        "duty, GST, and making charges). Historical points use each date's "
        "USD/INR rate; the forecast segment uses the latest rate."
    )


class IndiaRatePoint(BaseModel):
    date: dt.date
    price_24k_pg: float
    price_22k_pg: float
    price_18k_pg: float


class IndiaCity(BaseModel):
    slug: str
    name: str
    type: str
    state_name: str | None = None


class IndiaCitiesResponse(BaseModel):
    cities: list[IndiaCity]


INDIA_RETAIL_DISCLAIMER = (
    "Published city-wise RETAIL gold rates from Groww (groww.in/gold-rates) — "
    "these include import duty, GST, and local dealer premium, and therefore "
    "differ between cities and from international/bullion prices. Rates move "
    "during the day; check with your local jeweller before transacting. "
    "TimesFM forecasts are computed on the COMEX bullion series, not on these "
    "retail rates."
)


class IndiaRatesResponse(BaseModel):
    city_slug: str
    city: str
    date: dt.date
    per_gram: dict[str, float]
    per_10g: dict[str, float]
    pct_change: dict[str, float] | None = None
    history: list[IndiaRatePoint]
    source: str
    disclaimer: str = INDIA_RETAIL_DISCLAIMER


INDIA_FORECAST_DISCLAIMER = (
    "ESTIMATED retail forecast: the TimesFM COMEX bullion forecast (which is the "
    "only series with enough history to forecast) scaled by the city's current "
    "retail premium (today's published retail quote / today's bullion-equivalent). "
    "The premium varies daily and by city, so this is an approximation — not a "
    "jeweller quote."
)


class HistoryResponse(BaseModel):
    start: dt.date
    end: dt.date
    granularity: Granularity
    source: str
    points: list[OhlcPoint]
    currency: Currency = "usd"
    karat: Karat | None = None
    unit: Unit | None = None
    rate: RateInfo | None = None


class BaselineSeries(BaseModel):
    name: str
    values: list[float]


class BandCalibration(BaseModel):
    """Band scaling so the p10–p90 band honestly covers ~80% of historical
    out-of-sample outcomes (raw model bands are often too narrow)."""

    scale: float
    target_coverage_pct: float = 80.0


class ForecastResponse(BaseModel):
    horizon: HorizonPreset
    horizon_days: int
    model_version: str
    generated_at: dt.datetime
    latency_ms: float
    history_last_date: dt.date
    history_last_close: float
    dates: list[dt.date]
    point: list[float]
    q10: list[float]
    q50: list[float]
    q90: list[float]
    quantiles: bool
    baselines: list[BaselineSeries]
    currency: Currency = "usd"
    karat: Karat | None = None
    unit: Unit | None = None
    rate: RateInfo | None = None
    band_calibration: BandCalibration | None = None


class IndiaForecastResponse(ForecastResponse):
    premium_ratio: float
    bullion_history_last_close: float
    disclaimer: str = INDIA_FORECAST_DISCLAIMER


class BacktestFoldOut(BaseModel):
    origin_date: str
    horizon_days: int
    mape: float
    rmse: float
    mae: float
    directional_accuracy: float
    band_coverage: float | None = None


class BacktestModelResult(BaseModel):
    model: str
    model_version: str
    horizon_days: int
    step_days: int
    summary: dict[str, float]
    folds: list[BacktestFoldOut]


class BacktestResponse(BaseModel):
    generated_at: dt.datetime
    model_version: str
    context_length: int
    results: dict[str, BacktestModelResult]


class BacktestScoreEntry(BaseModel):
    horizon_days: int
    generated_at: dt.datetime
    model_version: str
    mape_pct: float
    directional_accuracy_pct: float
    naive_directional_accuracy_pct: float | None = None
    skill_vs_naive_pp: float | None = None
    band_coverage_pct: float | None = None
    n_folds: int
    underperforming_naive: bool


class BacktestScoreboard(BaseModel):
    entries: list[BacktestScoreEntry]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    data_last_date: dt.date | None
    data_freshness: str
    data_may_be_stale: bool
    last_refresh_ok: bool
    model_underperforming_baseline: bool = False
    bootstrap_error: str | None = None
    source_status: dict[str, str]


class RefreshResponse(BaseModel):
    status: str
    rows: int
    last_date: dt.date | None
    cache_invalidated: bool


class ErrorEnvelope(BaseModel):
    error: dict[str, str]


# --------------------------------------------------------------------------- #
# Alerts (Telegram)
# --------------------------------------------------------------------------- #
AlertOp = Literal["<=", ">="]
AlertCurrency = Literal["usd", "inr"]


class AlertTargetIn(BaseModel):
    karat: Karat = "24k"
    unit: Unit = "10gram"
    currency: AlertCurrency = "inr"
    op: AlertOp
    price: float = Field(gt=0)


class AlertTarget(BaseModel):
    id: str
    karat: Karat
    unit: Unit
    currency: AlertCurrency
    op: AlertOp
    price: float
    created_at: dt.datetime
    triggered_at: dt.datetime | None = None
    triggered_value: float | None = None


class AlertTargetsResponse(BaseModel):
    targets: list[AlertTarget]
    alerts_enabled: bool


class AlertCreatedResponse(BaseModel):
    created: AlertTarget


class AlertDeletedResponse(BaseModel):
    deleted: bool


# --------------------------------------------------------------------------- #
# Forecast accountability
# --------------------------------------------------------------------------- #
class AccountabilityPoint(BaseModel):
    origin_date: dt.date
    horizon_days: int
    target_date: dt.date
    predicted: float
    actual: float
    err_pct: float
    in_band: bool


class AccountabilityHorizon(BaseModel):
    horizon_days: int
    n_scored: int
    mape_pct: float
    band_coverage_pct: float
    n_pending: int


class AccountabilityResponse(BaseModel):
    per_horizon: list[AccountabilityHorizon]
    pending_counts: dict[str, int]
    recent: list[AccountabilityPoint]
    disclaimer: str = (
        "Realized track record of forecasts this system actually logged — "
        "scored automatically as actual closes become known. Empty until the "
        "first logged forecast matures."
    )
