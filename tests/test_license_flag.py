"""License gating tests for the MODEL_VERSION flag."""

import numpy as np
import pytest

import timesfm

import config
from forecasting.timesfm_service import GoldForecastService
from tests.conftest import make_history


def test_3point0_requires_non_commercial_flag(monkeypatch):
    monkeypatch.setattr(config.settings, "non_commercial", False)
    service = GoldForecastService(model_version="3.0")
    with pytest.raises(PermissionError, match="non-commercial"):
        service.load_model()


def test_3point0_with_flag_loads_and_predicts(monkeypatch):
    monkeypatch.setattr(config.settings, "non_commercial", True)

    class StubOut:
        point = np.arange(5, dtype=float)
        quantiles = np.stack([np.arange(5) - 2, np.arange(5) + 2], axis=-1)

    class StubForecaster:
        config = type("Cfg", (), {"quantiles": [0.1, 0.9]})()

        @staticmethod
        def predict(context, horizon, return_quantiles=False, make_positive=False):
            return StubOut()

    monkeypatch.setattr(
        timesfm.TimesFM3Forecaster,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: StubForecaster()),
    )
    service = GoldForecastService(model_version="3.0")
    service.load_model()
    h = make_history(n=100)
    res = service.forecast(h, 5)
    assert res.point.shape == (5,)
    assert np.all(res.q10 <= res.q90)
