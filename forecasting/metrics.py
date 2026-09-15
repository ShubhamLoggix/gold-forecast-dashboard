"""Metrics shared by forecast + baseline evaluation."""

from __future__ import annotations

import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def directional_accuracy(
    last_observed: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> float:
    """Fraction of horizon steps where predicted direction matches actual direction.

    Direction at step i = sign(value_i - value_{i-1}), where value_{i-1} for the
    first horizon step is the last observed value before the forecast starts.
    """
    last_observed = np.asarray(last_observed)
    actual = np.concatenate(([last_observed[0]], np.asarray(actual)))
    predicted = np.concatenate(([last_observed[0]], np.asarray(predicted)))
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    if len(actual_dir) == 0:
        return float("nan")
    return float(np.mean(actual_dir == pred_dir))


def quantile_spread(q10: np.ndarray, q90: np.ndarray) -> float:
    """Mean width of the p10-p90 band (used for request logging)."""
    q10, q90 = np.asarray(q10), np.asarray(q90)
    return float(np.mean(q90 - q10))
