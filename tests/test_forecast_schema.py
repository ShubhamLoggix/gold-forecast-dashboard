"""Forecast service tests using a stubbed model (no real weights, no network)."""

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from forecasting import timesfm_service as svc_mod
from forecasting.timesfm_service import (
    HORIZON_PRESETS,
    GoldForecastService,
    TimesFMNotLoaded,
)
from tests.conftest import make_history


class StubModel:
    """Deterministic stub: point = last close * (1 + 0.001 * step).

    Bands widen with step. Mimics the real output contract used by the
    service's 2.5 adapter (point array + quantile array with 10 columns).
    """

    def forecast(self, horizon, inputs):
        ctx = np.asarray(inputs[0], dtype=np.float64)
        last = ctx[-1]
        point = np.array([last * (1 + 0.001 * (i + 1)) for i in range(horizon)])
        quant = np.zeros((1, horizon, 10))
        for i in range(horizon):
            band = last * 0.02 * (i + 1)
            quant[0, i, svc_mod.Q10_COL] = point[i] - band
            quant[0, i, svc_mod.Q50_COL] = point[i]
            quant[0, i, svc_mod.Q90_COL] = point[i] + band
            quant[0, i, 0] = point[i]  # point channel
        return point[None, :], quant


def make_service() -> GoldForecastService:
    service = GoldForecastService()
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    return service


def test_horizon_presets_match_dashboard():
    assert HORIZON_PRESETS == {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}


def test_forecast_shapes_and_quantile_ordering():
    service = make_service()
    h = make_history(n=300)
    res = service.forecast(h, 21)
    assert res.point.shape == (21,)
    assert res.q10.shape == res.q50.shape == res.q90.shape == (21,)
    assert np.all(res.q10 <= res.q50)
    assert np.all(res.q50 <= res.q90)
    assert res.dates[0] > h["date"].iloc[-1].date()
    # dates are consecutive business days
    expected = pd.bdate_range(h["date"].iloc[-1] + pd.offsets.BDay(1), periods=21)
    assert list(res.dates) == [d.date() for d in expected]


def test_future_dates_skip_weekends():
    service = make_service()
    h = make_history(n=300, start="2024-01-01")
    res = service.forecast(h, 10)
    weekdays = {d.weekday() for d in res.dates}
    assert weekdays <= {0, 1, 2, 3, 4}


def test_forecast_requires_model_load():
    service = GoldForecastService()
    h = make_history(n=100)
    with pytest.raises(TimesFMNotLoaded):
        service.forecast(h, 5)


def test_forecast_rejects_bad_history():
    service = make_service()
    with pytest.raises(ValueError):
        service.forecast(pd.DataFrame({"date": [], "close": []}), 5)
    h = make_history(n=100)
    bad = h.copy()
    bad.loc[5, "close"] = float("nan")
    with pytest.raises(ValueError):
        service.forecast(bad, 5)
    bad2 = h.copy()
    bad2.loc[5, "close"] = -3.0
    with pytest.raises(ValueError):
        service.forecast(bad2, 5)


def test_horizon_too_long_raises():
    service = make_service()
    h = make_history(n=300)
    with pytest.raises(ValueError):
        service.forecast(h, 300)


def test_quantiles_false_collapses_band():
    service = make_service()
    h = make_history(n=300)
    res = service.forecast(h, 5, quantiles=False)
    assert np.allclose(res.q10, res.q90)


def test_backtest_runs_against_baselines_and_persists(tmp_data_dir):
    service = make_service()
    h = make_history(n=250)
    results = service.backtest(h, horizon_days=10, step_days=20, max_folds=5)
    assert set(results) == {"timesfm-2.5", "naive-last-value", "sma-20"}
    for res in results.values():
        assert len(res.folds) == 5
        assert set(res.summary) == {
            "mape_pct", "rmse", "mae", "directional_accuracy_pct", "n_folds",
        }
        assert 0.0 <= res.summary["directional_accuracy_pct"] <= 100.0
    path = service.persist_backtest(results, data_dir=tmp_data_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "2.5"
    assert set(payload["results"]) == {"timesfm-2.5", "naive-last-value", "sma-20"}


def test_backtest_insufficient_history_raises():
    service = make_service()
    h = make_history(n=40)
    with pytest.raises(ValueError):
        service.backtest(h, horizon_days=30, step_days=7)


def test_backtest_metrics_sane_vs_baselines(tmp_data_dir):
    service = make_service()
    h = make_history(n=250)
    results = service.backtest(h, horizon_days=10, step_days=20, max_folds=5)
    # MAPE must be finite and positive for all models
    for res in results.values():
        assert np.isfinite(res.summary["mape_pct"])
        assert res.summary["mape_pct"] > 0


def test_real_timesfm_output_layout_calibration_guard():
    """Guard the empirically calibrated quantile column indices.

    scripts/verify_model.py established: col 0 = point channel, cols 1..9 =
    q(0.1..0.9), point == col 5. If these indices are wrong, the dashboard
    bands will be mislabeled - fix in scripts/verify_model.py first.
    """
    assert svc_mod.Q10_COL == 1
    assert svc_mod.Q50_COL == 5
    assert svc_mod.Q90_COL == 9
    assert svc_mod.MAX_HORIZON >= HORIZON_PRESETS["1y"]
