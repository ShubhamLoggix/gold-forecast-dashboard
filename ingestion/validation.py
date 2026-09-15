"""Data validation for ingested gold price series.

All checks log explicitly; nothing is silently dropped or filled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("gold_forecast.ingestion")

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "source"]

MAX_SINGLE_DAY_MOVE = 0.20  # >20% close-to-close move is suspicious


@dataclass
class ValidationReport:
    rows: int
    dropped_negative: int = 0
    dropped_missing_close: int = 0
    duplicates_removed: int = 0
    weekend_rows_dropped: int = 0
    weekday_gaps_logged: int = 0
    big_moves_warned: int = 0
    sorted: bool = False

    def summary(self) -> str:
        return (
            f"rows={self.rows} dropped_negative={self.dropped_negative} "
            f"dropped_missing_close={self.dropped_missing_close} "
            f"duplicates_removed={self.duplicates_removed} "
            f"weekend_rows_dropped={self.weekend_rows_dropped} "
            f"weekday_gaps={self.weekday_gaps_logged} big_move_warnings={self.big_moves_warned}"
        )


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + clean a raw OHLC frame; returns the canonical daily series.

    Guarantees: unique sorted business-day `date` index-like column, positive
    prices, no silent forward-fill of prices (weekend/holiday gaps are kept as
    missing and logged).
    """
    report = ValidationReport(rows=len(df))
    if df.empty:
        raise ValueError("No data rows returned by source; cannot build series.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in out.columns and c != "source"]
    if missing_cols:
        raise ValueError(f"Source frame missing required columns: {missing_cols}")
    if "source" not in out.columns:
        out["source"] = "unknown"

    # Duplicate dates: keep first occurrence.
    dup_mask = out.duplicated(subset=["date"], keep="first")
    if dup_mask.any():
        report.duplicates_removed = int(dup_mask.sum())
        logger.warning(
            "Removed %d duplicate dates (kept first occurrence).",
            report.duplicates_removed,
        )
        out = out[~dup_mask]

    # Sort monotonic.
    if not out["date"].is_monotonic_increasing:
        out = out.sort_values("date").reset_index(drop=True)
        report.sorted = True
        logger.info("Re-sorted %d rows to enforce monotonic dates.", len(out))

    # Negative / non-positive prices.
    price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    neg_mask = (out[price_cols] <= 0).any(axis=1)
    if neg_mask.any():
        report.dropped_negative = int(neg_mask.sum())
        logger.warning("Dropping %d rows with non-positive prices.", report.dropped_negative)
        out = out[~neg_mask]

    # Missing close: cannot build a series from rows without close.
    miss_mask = out["close"].isna()
    if miss_mask.any():
        report.dropped_missing_close = int(miss_mask.sum())
        logger.warning(
            "Dropping %d rows with missing close (kept as gaps; NOT forward-filled).",
            report.dropped_missing_close,
        )
        out = out[~miss_mask]

    # Weekend rows: sources should not supply them; drop + log if they appear.
    weekend_mask = out["date"].dt.dayofweek >= 5
    if weekend_mask.any():
        report.weekend_rows_dropped = int(weekend_mask.sum())
        logger.warning(
            "Dropping %d weekend rows (markets closed).", report.weekend_rows_dropped
        )
        out = out[~weekend_mask]

    # Weekday holiday gaps: log count of weekday slots absent from the series
    # (inferred holidays / market closures). Prices are NOT forward-filled.
    if len(out) > 1:
        all_weekdays = pd.bdate_range(out["date"].min(), out["date"].max())
        present = set(out["date"])
        gaps = [d for d in all_weekdays if d not in present]
        report.weekday_gaps_logged = len(gaps)
        if gaps:
            logger.info(
                "%d weekday slots absent (holidays/market closures) between %s and %s; "
                "kept as gaps, not forward-filled.",
                len(gaps),
                all_weekdays.min().date(),
                all_weekdays.max().date(),
            )

    # Suspicious jumps: >MAX_SINGLE_DAY_MOVE close-to-close.
    if len(out) > 1:
        pct = out["close"].pct_change().abs()
        jump_mask = pct > MAX_SINGLE_DAY_MOVE
        report.big_moves_warned = int(jump_mask.sum())
        for _, row in out.loc[jump_mask, ["date", "close"]].iterrows():
            logger.warning(
                "Possible bad tick: %0.1f%% single-day close move at %s "
                "(close=%s). Row kept, but verify against source.",
                float(pct.loc[row.name] or 0) * 100,
                row["date"].date(),
                row["close"],
            )

    out = out.reset_index(drop=True)
    logger.info("Validation complete: %s", report.summary())
    return out[REQUIRED_COLUMNS]
