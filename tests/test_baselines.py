"""Baseline tests."""

import numpy as np
import pytest

from forecasting.baselines import naive_last_value, simple_moving_average
from tests.conftest import make_history


def test_naive_last_value_carry_forward():
    h = make_history(n=50)
    out = naive_last_value(h, 21)
    assert out.shape == (21,)
    assert np.allclose(out, h["close"].iloc[-1])


def test_sma20_is_mean_of_last_20():
    h = make_history(n=50)
    out = simple_moving_average(h, 10)
    expected = h["close"].to_numpy()[-20:].mean()
    assert out.shape == (10,)
    assert np.allclose(out, expected)


def test_sma_short_history_raises():
    h = make_history(n=10)
    with pytest.raises(ValueError):
        simple_moving_average(h, 5)


def test_naive_rejects_nan():
    h = make_history(n=50)
    h.loc[3, "close"] = float("nan")
    with pytest.raises(ValueError):
        naive_last_value(h, 5)
