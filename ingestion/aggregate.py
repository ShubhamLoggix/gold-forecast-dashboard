"""Shared historical-series aggregation used by the Streamlit app and the API."""

from __future__ import annotations

import pandas as pd

GRANULARITY_RULES = {"day": None, "week": "W-FRI", "month": "ME"}


def aggregate_history(history: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate the canonical daily frame to day/week/month granularity.

    Day returns the frame unchanged; Week ends Friday; Month ends month-end.
    open=first, high=max, low=min, close=last, volume=sum.
    """
    rule = GRANULARITY_RULES.get(granularity.lower())
    if rule is None:
        return history
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"])
    agg = (
        df.set_index("date")
        .resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source=("source", "last"),
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    return agg
