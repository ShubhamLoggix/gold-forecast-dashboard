"""Naive baselines for honest comparison against the TimesFM forecast.

Published research shows zero-shot TSFMs perform near chance level (~50%
directional accuracy) on financial series; these baselines let users judge
whether the model actually adds value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_last_value(history: pd.DataFrame, horizon_days: int) -> np.ndarray:
    """Carry the last observed close forward for `horizon_days` trading days."""
    _validate_history(history)
    last = float(history["close"].iloc[-1])
    return np.full(horizon_days, last, dtype=np.float64)


def simple_moving_average(
    history: pd.DataFrame, horizon_days: int, window: int = 20
) -> np.ndarray:
    """Mean of the last `window` closes, carried forward for `horizon_days`."""
    _validate_history(history)
    values = history["close"].astype(float).to_numpy()
    if len(values) < window:
        raise ValueError(f"Need at least {window} observations for SMA({window})")
    return np.full(horizon_days, values[-window:].mean(), dtype=np.float64)


def _validate_history(history: pd.DataFrame) -> None:
    if history is None or len(history) == 0 or "close" not in history.columns:
        raise ValueError("history must be a DataFrame with a 'close' column")
    if history["close"].isna().any():
        raise ValueError("history contains NaN close values")
