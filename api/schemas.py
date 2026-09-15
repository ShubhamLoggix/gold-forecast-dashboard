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


class BacktestFoldOut(BaseModel):
    origin_date: str
    horizon_days: int
    mape: float
    rmse: float
    mae: float
    directional_accuracy: float


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
