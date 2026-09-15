"""Forecast accountability: log every generated forecast, score it later.

Every day the scheduler (or startup bootstrap) generates forecasts for the
standard horizons and appends one row per (origin, horizon, target date) to
data/forecasts/forecast_log.parquet. When the actual close becomes known, the
log is scored: predicted vs realized error, and whether the actual fell inside
the model's own p10-p90 band. This is the public track record — it measures
the model on promises it actually made, not ones invented afterwards.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger("gold_forecast.accountability")

LOG_FILENAME = "forecast_log.parquet"
HORIZONS = (5, 21, 63, 126, 252)  # 1w 1m 3m 6m 1y trading days

LOG_COLUMNS = [
    "made_at_utc",
    "origin_date",
    "horizon_days",
    "target_date",
    "point",
    "q10",
    "q90",
    "model_version",
]


def _log_path(data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir or settings.data_dir)
    return data_dir / "forecasts" / LOG_FILENAME


def log_daily_forecasts(
    service,
    history: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    data_dir: Path | None = None,
) -> int:
    """Generate + persist today's forecasts for all standard horizons.

    Returns the number of rows now in the log. Rows are deduped on
    (origin_date, horizon_days, target_date) with the latest write winning,
    so re-running on the same day is safe.
    """
    from ingestion.fetch_gold_prices import load_canonical  # noqa: F401 (typing)

    now = dt.datetime.now(dt.timezone.utc)
    origin_date = pd.Timestamp(history["date"].iloc[-1]).date()
    rows: list[dict] = []
    for days in horizons:
        try:
            result = service.forecast(history, days, quantiles=True)
        except Exception as exc:  # noqa: BLE001 - one horizon must not block the rest
            logger.warning("accountability: forecast h=%d failed: %s", days, exc)
            continue
        for i, target in enumerate(result.dates):
            rows.append(
                {
                    "made_at_utc": now,
                    "origin_date": pd.Timestamp(origin_date),
                    "horizon_days": days,
                    "target_date": pd.Timestamp(target),
                    "point": float(result.point[i]),
                    "q10": float(result.q10[i]),
                    "q90": float(result.q90[i]),
                    "model_version": result.model_version,
                }
            )
    if not rows:
        logger.warning("accountability: nothing logged today.")
        path = _log_path(data_dir)
        return len(pd.read_parquet(path)) if path.exists() else 0
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
    out_path = _log_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = (
        frame.sort_values(["origin_date", "horizon_days", "target_date", "made_at_utc"])
        .drop_duplicates(subset=["origin_date", "horizon_days", "target_date"], keep="last")
        .reset_index(drop=True)
    )
    frame.to_parquet(out_path, index=False)
    logger.info("accountability: log updated (%d rows).", len(frame))
    return len(frame)


def load_forecast_log(data_dir: Path | None = None) -> pd.DataFrame:
    path = _log_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"forecast log not found at {path}")
    frame = pd.read_parquet(path)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.normalize()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"]).dt.normalize()
    return frame


def score_forecasts(history: pd.DataFrame, data_dir: Path | None = None) -> dict:
    """Score logged forecasts against realized closes.

    Returns {"per_horizon": [...], "recent": [...]} where `per_horizon` has one
    entry per horizon (scored count, realized MAPE, realized band coverage,
    pending count) and `recent` lists the most recent evaluated points.
    """
    log = load_forecast_log(data_dir)
    actuals = pd.DataFrame(
        {
            "target_date": pd.to_datetime(history["date"]).dt.normalize(),
            "actual": history["close"].astype(float),
        }
    ).drop_duplicates(subset="target_date", keep="last")
    merged = log.merge(actuals, on="target_date", how="left")
    merged["scored"] = merged["actual"].notna()
    merged["err_pct"] = (
        (merged["point"] - merged["actual"]).abs() / merged["actual"] * 100
    )
    merged["in_band"] = (merged["actual"] >= merged["q10"]) & (
        merged["actual"] <= merged["q90"]
    )
    scored = merged[merged["scored"]].copy()

    per_horizon = []
    for days, group in merged.groupby("horizon_days"):
        g_scored = group[group["scored"]]
        per_horizon.append(
            {
                "horizon_days": int(days),
                "n_scored": int(len(g_scored)),
                "mape_pct": float(g_scored["err_pct"].mean()) if len(g_scored) else 0.0,
                "band_coverage_pct": (
                    float(g_scored["in_band"].mean() * 100) if len(g_scored) else 0.0
                ),
                "n_pending": int((~group["scored"]).sum()),
            }
        )
    per_horizon.sort(key=lambda e: e["horizon_days"])

    pending = merged[~merged["scored"]]
    pending_counts = {int(d): int(len(g)) for d, g in pending.groupby("horizon_days")}
    recent = (
        scored.sort_values("target_date", ascending=False)
        .head(12)
    )
    recent_list = [
        {
            "origin_date": pd.Timestamp(r["origin_date"]).date().isoformat(),
            "horizon_days": int(r["horizon_days"]),
            "target_date": pd.Timestamp(r["target_date"]).date().isoformat(),
            "predicted": float(r["point"]),
            "actual": float(r["actual"]),
            "err_pct": float(r["err_pct"]),
            "in_band": bool(r["in_band"]),
        }
        for _, r in recent.iterrows()
    ]
    return {
        "per_horizon": per_horizon,
        "pending_counts": pending_counts,
        "recent": recent_list,
    }
