"""Model bake-off: TimesFM vs Chronos vs naive vs SMA, all dashboard horizons.

Walk-forward, out-of-sample, identical folds for every model. Writes
data/backtests/bakeoff_<date>.json and prints the leaderboard.

Usage:
    python scripts/model_bakeoff.py [--models timesfm,chronos]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from forecasting.timesfm_service import (
    GoldForecastService,
    _fold_metrics,
    _summarize,
)
from ingestion.fetch_gold_prices import load_canonical
from logging_setup import setup_logging

logger = logging.getLogger("gold_forecast.bakeoff")

HORIZONS = (
    (5, 7, 52, "1w"),
    (21, 7, 52, "1m"),
    (63, 21, 26, "3m"),
    (126, 63, 13, "6m"),
    (252, 63, 8, "1y"),
)


def walk_forward(
    history: pd.DataFrame,
    predict_fn,
    context_length: int,
    horizon_days: int,
    step_days: int,
    max_folds: int,
    model_name: str,
    model_version: str,
):
    """Identical folds to the TimesFM backtest; returns BacktestResult."""
    values = history["close"].astype(float).to_numpy()
    dates = pd.to_datetime(history["date"]).reset_index(drop=True)
    n = len(values)
    first_origin = max(512, min(context_length, n // 2))
    origins = list(range(first_origin, n - horizon_days + 1, step_days))
    if len(origins) > max_folds:
        stride = len(origins) // max_folds or 1
        origins = origins[::stride][:max_folds]
    folds = []
    cal_samples = []
    for origin in origins:
        ctx = values[max(0, origin - context_length) : origin]
        actual = values[origin : origin + horizon_days]
        if len(actual) < horizon_days:
            continue
        point, q10, q90 = predict_fn(ctx.astype(np.float32), horizon_days)
        folds.append(
            _fold_metrics(dates, origin, horizon_days, values[origin - 1], point, actual, q10, q90)
        )
        half_width = np.maximum((q90 - q10) / 2.0, 1e-9)
        cal_samples.append(np.abs(actual - point) / half_width)
    result = _summarize(model_name, model_version, horizon_days, step_days, folds)
    if cal_samples:
        pooled = np.concatenate(cal_samples)
        result.band_scale = float(np.quantile(pooled, 0.8))
        result.summary["calibrated_band_coverage_pct"] = float(
            np.mean(pooled <= result.band_scale) * 100
        )
    return result


def _baseline_results(history: pd.DataFrame, horizon_days, step_days, max_folds):
    """Naive + SMA on the same folds (bands are degenerate -> coverage 0)."""
    values = history["close"].astype(float).to_numpy()
    dates = pd.to_datetime(history["date"]).reset_index(drop=True)
    n = len(values)
    first_origin = max(512, min(settings.context_length, n // 2))
    origins = list(range(first_origin, n - horizon_days + 1, step_days))
    if len(origins) > max_folds:
        stride = len(origins) // max_folds or 1
        origins = origins[::stride][:max_folds]
    nv_folds, sma_folds = [], []
    for origin in origins:
        actual = values[origin : origin + horizon_days]
        if len(actual) < horizon_days:
            continue
        nv = np.full(horizon_days, values[origin - 1])
        nv_folds.append(_fold_metrics(dates, origin, horizon_days, values[origin - 1], nv, actual, nv, nv))
        sma = np.full(horizon_days, float(np.mean(values[max(0, origin - 20) : origin])))
        sma_folds.append(_fold_metrics(dates, origin, horizon_days, values[origin - 1], sma, actual, sma, sma))
    return nv_folds, sma_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="timesfm,chronos")
    args = parser.parse_args()
    setup_logging(settings.log_level)

    try:
        history = load_canonical()
    except FileNotFoundError:
        raise SystemExit("no canonical data — run ingestion first")

    models: list[tuple[str, object, int, str]] = []
    if "timesfm" in args.models:
        tm = GoldForecastService(
            model_version=settings.model_version, context_length=settings.context_length
        )
        tm.load_model()
        models.append(("timesfm-2.5", tm._predict_fn, settings.context_length, settings.model_version))
    if "chronos" in args.models:
        from forecasting.chronos_service import ChronosForecastService

        ch = ChronosForecastService()
        ch.load_model()
        models.append(("chronos-t5-tiny", ch.predict, ch.context_length, "t5-tiny"))

    entries = []
    for horizon_days, step_days, max_folds, label in HORIZONS:
        for model_name, predict_fn, context_length, model_version in models:
            result = walk_forward(
                history, predict_fn, context_length, horizon_days, step_days, max_folds,
                model_name, model_version,
            )
            entries.append(_entry(result, label))
        nv_folds, sma_folds = _baseline_results(history, horizon_days, step_days, max_folds)
        entries.append(_entry(_summarize("naive-last-value", "-", horizon_days, step_days, nv_folds), label))
        entries.append(_entry(_summarize("sma-20", "-", horizon_days, step_days, sma_folds), label))

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_range": [
            pd.Timestamp(history["date"].iloc[0]).date().isoformat(),
            pd.Timestamp(history["date"].iloc[-1]).date().isoformat(),
        ],
        "entries": entries,
    }
    out_dir = Path(settings.data_dir) / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bakeoff_{dt.date.today().isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Bakeoff report written to %s", path)

    print(f"\n{'model':>18} {'horizon':>8} {'MAPE%':>8} {'dir%':>7} {'cov%':>7} {'cal-cov%':>9}")
    for e in sorted(entries, key=lambda x: (x["horizon_days"], x["model"])):
        print(
            f"{e['model']:>18} {e['horizon_label']:>8} {e['mape_pct']:>8.2f} "
            f"{e['directional_accuracy_pct']:>7.1f} {e['band_coverage_pct']:>7.1f} "
            f"{e['calibrated_band_coverage_pct']:>9.1f}"
        )


def _entry(result, label: str) -> dict:
    summary = result.summary
    return {
        "horizon_days": result.horizon_days,
        "horizon_label": label,
        "model": result.model_name,
        "mape_pct": round(summary.get("mape_pct", float("nan")), 3),
        "directional_accuracy_pct": round(
            summary.get("directional_accuracy_pct", float("nan")), 2
        ),
        "band_coverage_pct": round(summary.get("band_coverage_pct", float("nan")), 2),
        "calibrated_band_coverage_pct": round(
            summary.get("calibrated_band_coverage_pct", float("nan")), 2
        ),
        "band_scale": result.band_scale,
        "n_folds": len(result.folds),
    }


if __name__ == "__main__":
    main()
