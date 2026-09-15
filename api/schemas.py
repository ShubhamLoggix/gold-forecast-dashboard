"""Pydantic request/response schemas for the gold forecast API."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

Granularity = Literal["day", "week", "month"]
HorizonPreset = Literal["1w", "1m", "3m", "6m", "1y"]


class OhlcPoint(BaseModel):
    date: dt.date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None


class HistoryResponse(BaseModel):
    start: dt.date
    end: dt.date
    granularity: Granularity
    source: str
    points: list[OhlcPoint]


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
    source_status: dict[str, str]


class RefreshResponse(BaseModel):
    status: str
    rows: int
    last_date: dt.date | None
    cache_invalidated: bool


class ErrorEnvelope(BaseModel):
    error: dict[str, str]
