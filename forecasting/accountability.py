"""Forecast accountability: log every generated forecast, score it later.

Every day the scheduler (or startup bootstrap) generates forecasts for the
standard horizons and appends one row per (series, origin, horizon, target
date) to data/forecasts/forecast_log.parquet. When the actual close becomes
known, the log is scored: predicted vs realized error, direction correctness,
and whether the actual fell inside the model's own p10-p90 band. This is the
public track record — it measures the model on promises it actually made, not
ones invented afterwards.

`series` names: "gold_comex" (legacy default), "silver", "premium:<city>:<karat>".
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
ROLLING_WINDOWS = (7, 30, 90)  # calendar days, rolling view windows
DEFAULT_SERIES = "gold_comex"

LOG_COLUMNS = [
    "made_at_utc",
    "origin_date",
    "horizon_days",
    "target_date",
    "series",
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
    series: str = DEFAULT_SERIES,
) -> int:
    """Generate + persist today's forecasts for all standard horizons.

    Returns the number of rows now in the log. Rows are deduped on
    (series, origin_date, horizon_days, target_date) with the latest write
    winning, so re-running on the same day is safe.
    """
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
                    "series": series,
                    "point": float(result.point[i]),
                    "q10": float(result.q10[i]),
                    "q90": float(result.q90[i]),
                    "model_version": result.model_version,
                }
            )
    if not rows:
        logger.warning("accountability: nothing logged today (series=%s).", series)
        path = _log_path(data_dir)
        return len(pd.read_parquet(path)) if path.exists() else 0
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
    out_path = _log_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = (
        frame.sort_values(
            ["series", "origin_date", "horizon_days", "target_date", "made_at_utc"]
        )
        .drop_duplicates(
            subset=["series", "origin_date", "horizon_days", "target_date"],
            keep="last",
        )
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
    if "series" not in frame.columns:
        frame["series"] = DEFAULT_SERIES  # tolerate pre-series logs
    return frame


def _actuals_for_series(series_name: str, gold_history: pd.DataFrame) -> pd.DataFrame:
    """Realized closes (dates + actual value) for one logged series."""
    if series_name == DEFAULT_SERIES or series_name.startswith("gold"):
        actual = pd.DataFrame(
            {
                "target_date": pd.to_datetime(gold_history["date"]).dt.normalize(),
                "actual": gold_history["close"].astype(float),
            }
        )
    elif series_name.startswith("premium:"):
        _city, _, _karat = series_name.split(":")
        from ingestion.premium_series import premium_actuals

        actual = premium_actuals(_city, _karat)
    elif series_name.startswith("silver"):
        from ingestion.fetch_silver_prices import load_canonical as load_silver

        actual = pd.DataFrame(
            {
                "target_date": pd.to_datetime(load_silver()["date"]).dt.normalize(),
                "actual": load_silver()["close"].astype(float),
            }
        )
    else:
        raise ValueError(f"no actuals source wired for series {series_name!r}")
    return actual.drop_duplicates(subset="target_date", keep="last")


def score_forecasts(gold_history: pd.DataFrame, data_dir: Path | None = None) -> dict:
    """Score logged forecasts against realized closes, per series.

    Returns:
      {
        "per_series": {series: [AccountabilityHorizon-like dicts with
            horizon_days, n_scored, mape_pct, mae, directional_acc_pct,
            band_coverage_pct, coverage_pct, n_pending, windows (each window
            has mae, mape_pct, coverage_pct, n)]},
        "per_horizon": same list for the default (gold) series, kept for
            backward-compatible callers,
        "pending_counts": {series.horizon: n},
        "recent": most recent evaluated points (default series),
        "generated_at": iso now,
      }
    """
    log = load_forecast_log(data_dir)
    if log.empty:
        return _empty_score_response()

    today = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    per_series: dict[str, list[dict]] = {}
    per_series_merged: dict[str, pd.DataFrame] = {}
    for series_name, group in log.groupby("series"):
        try:
            actuals = _actuals_for_series(series_name, gold_history)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("accountability: no actuals for %s (%s); skipping", series_name, exc)
            per_series[series_name] = [
                {
                    "horizon_days": int(days),
                    "n_scored": 0,
                    "mape_pct": 0.0,
                    "mae": 0.0,
                    "directional_acc_pct": 0.0,
                    "band_coverage_pct": 0.0,
                    "coverage_pct": 0.0,
                    "n_pending": int(len(g)),
                    "windows": {
                        str(w): {
                            "mae": 0.0,
                            "mape_pct": 0.0,
                            "coverage_pct": 0.0,
                            "n": 0,
                        }
                        for w in ROLLING_WINDOWS
                    },
                }
                for days, g in group.groupby("horizon_days")
            ]
            per_series[series_name].sort(key=lambda e: e["horizon_days"])
            continue

        merged = group.merge(actuals, on="target_date", how="left")
        # Origin close = the series' realized value on the day the forecast was
        # made, used for sign-of-change direction evaluation.
        origin = actuals.rename(
            columns={"target_date": "origin_date", "actual": "origin_close"}
        )
        merged = merged.merge(origin, on="origin_date", how="left")

        merged["scored"] = merged["actual"].notna()
        merged["err_abs"] = (merged["point"] - merged["actual"]).abs().astype(float)
        merged["err_pct"] = (
            (merged["point"] - merged["actual"]).abs() / merged["actual"] * 100
        ).astype(float)
        merged["in_band"] = (merged["actual"] >= merged["q10"]) & (
            merged["actual"] <= merged["q90"]
        )
        # Explicit per-point band hit, nullable until the actual is known —
        # mirrors direction_correct. Same value as in_band once scored.
        merged["within_q10_q90"] = merged["in_band"].astype("object")
        merged.loc[~merged["scored"], "within_q10_q90"] = None
        merged["direction_correct"] = (
            (merged["point"] - merged["origin_close"]) > 0
        ) == ((merged["actual"] - merged["origin_close"]) > 0)
        merged["direction_correct"] = merged["direction_correct"].astype("object")
        merged.loc[merged["origin_close"].isna(), "direction_correct"] = None
        merged["n_direction"] = merged["direction_correct"].notna()
        per_series_merged[series_name] = merged

        stats_by_horizon: list[dict] = []
        for days, hg in merged.groupby("horizon_days"):
            s = hg[hg["scored"]]
            dir_acc: float | None = None
            dir_n = int(s["n_direction"].sum())
            if dir_n:
                dir_acc = float(s["direction_correct"].astype(float).mean() * 100)
            windows: dict[str, dict] = {}
            for win in ROLLING_WINDOWS:
                wg = s[pd.to_datetime(s["target_date"]) >= (today - pd.Timedelta(days=win))]
                windows[str(win)] = {
                    "mae": float(wg["err_abs"].mean()) if len(wg) else 0.0,
                    "mape_pct": float(wg["err_pct"].mean()) if len(wg) else 0.0,
                    "coverage_pct": (
                        float(wg["in_band"].mean() * 100) if len(wg) else 0.0
                    ),
                    "n": int(len(wg)),
                }
            stats_by_horizon.append(
                {
                    "horizon_days": int(days),
                    "n_scored": int(len(s)),
                    "mape_pct": float(s["err_pct"].mean()) if len(s) else 0.0,
                    "mae": float(s["err_abs"].mean()) if len(s) else 0.0,
                    "directional_acc_pct": float(dir_acc) if dir_acc is not None else 0.0,
                    "band_coverage_pct": (
                        float(s["in_band"].mean() * 100) if len(s) else 0.0
                    ),
                    "coverage_pct": (
                        float(s["in_band"].mean() * 100) if len(s) else 0.0
                    ),
                    "n_pending": int((~hg["scored"]).sum()),
                    "windows": windows,
                }
            )
        stats_by_horizon.sort(key=lambda e: e["horizon_days"])
        per_series[series_name] = stats_by_horizon
    # Legacy accessor: the default (gold) series list.
    per_horizon = per_series.get(DEFAULT_SERIES, [])

    pending_counts: dict[str, int] = {}
    for series_name, sgroup in log.groupby("series"):
        merged_here = per_series_merged.get(series_name)
        if merged_here is None:
            for (sname, days), g in sgroup.groupby(["series", "horizon_days"]):
                pending_counts[f"{sname}.{int(days)}"] = int(len(g))
            continue
        pend = merged_here[~merged_here["scored"]]
        for (sname, days), g in pend.groupby(["series", "horizon_days"]):
            pending_counts[f"{sname}.{int(days)}"] = int(len(g))

    recent: list[dict] = []
    scored_df = per_series_merged.get(DEFAULT_SERIES)
    if scored_df is not None and len(scored_df):
        recent_rows = scored_df[scored_df["scored"]].sort_values(
            "target_date", ascending=False
        ).head(12)
        recent = [
            {
                "origin_date": pd.Timestamp(r["origin_date"]).date().isoformat(),
                "horizon_days": int(r["horizon_days"]),
                "target_date": pd.Timestamp(r["target_date"]).date().isoformat(),
                "series": str(r["series"]),
                "predicted": float(r["point"]),
                "actual": float(r["actual"]),
                "err_pct": float(r["err_pct"]),
                "in_band": bool(r["in_band"]),
                "within_q10_q90": (
                    bool(r["within_q10_q90"])
                    if pd.notna(r["within_q10_q90"])
                    else None
                ),
                "direction_correct": (
                    bool(r["direction_correct"])
                    if pd.notna(r["direction_correct"])
                    else None
                ),
            }
            for _, r in recent_rows.iterrows()
        ]

    return {
        "per_series": per_series,
        "per_horizon": per_horizon,
        "pending_counts": pending_counts,
        "recent": recent,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _empty_score_response() -> dict:
    return {
        "per_series": {},
        "per_horizon": [],
        "pending_counts": {},
        "recent": [],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }