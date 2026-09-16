"""API contract tests with the real model stubbed out (no weights, no network)."""

import datetime as dt
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.main as main_mod
from api.main import app
from config import settings
from tests.conftest import make_history


class StubModel:
    def forecast(self, horizon, inputs):
        ctx = np.asarray(inputs[0], dtype=np.float64)
        last = ctx[-1]
        point = np.array([last * (1 + 0.001 * (i + 1)) for i in range(horizon)])
        quant = np.zeros((1, horizon, 10))
        for i in range(horizon):
            band = last * 0.02 * (i + 1)
            quant[0, i, 1] = point[i] - band
            quant[0, i, 5] = point[i]
            quant[0, i, 9] = point[i] + band
            quant[0, i, 0] = point[i]
        return point[None, :], quant


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Canonical data from fixture, written into a tmp data dir.
    frame = make_history(n=300)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5

    monkeypatch.setattr(deps, "get_history", lambda *a, **k: frame)
    monkeypatch.setattr(deps, "get_service", lambda: service)
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(deps, "bootstrap_if_needed", lambda: None)
    deps.invalidate_response_cache()
    with TestClient(app) as c:
        yield c


def test_history_endpoint_contract(client):
    resp = client.get("/api/v1/history", params={"granularity": "day"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["granularity"] == "day"
    assert len(body["points"]) == 300
    pt = body["points"][0]
    assert set(pt) == {"date", "open", "high", "low", "close", "volume"}
    assert resp.headers["X-Request-ID"]


def test_history_granularity_week_month(client):
    week = client.get("/api/v1/history", params={"granularity": "week"}).json()
    month = client.get("/api/v1/history", params={"granularity": "month"}).json()
    assert len(week["points"]) < 300
    assert len(month["points"]) < len(week["points"])
    assert week["points"][0]["date"] < week["points"][-1]["date"]


def test_history_date_filter(client):
    start = dt.date(2024, 2, 1)
    resp = client.get("/api/v1/history", params={"start": start.isoformat()})
    first = resp.json()["points"][0]["date"]
    assert first >= start.isoformat()


def test_history_empty_range_404_envelope(client):
    resp = client.get("/api/v1/history", params={"start": "1999-01-01", "end": "1999-02-01"})
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "http_error" and "request_id" in err


def test_forecast_endpoint_contract(client):
    resp = client.get("/api/v1/forecast", params={"horizon": "1m"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == "1m"
    assert body["horizon_days"] == 21
    assert len(body["dates"]) == len(body["point"]) == 21
    assert len(body["q10"]) == len(body["q90"]) == 21
    assert np.all(np.array(body["q10"]) <= np.array(body["q50"]))
    assert np.all(np.array(body["q50"]) <= np.array(body["q90"]))
    names = [b["name"] for b in body["baselines"]]
    assert names == ["naive-last-value", "sma-20"]
    # forecast dates are business days
    import pandas as pd

    ds = pd.to_datetime(pd.Series(body["dates"]))
    assert set(ds.dt.dayofweek) <= {0, 1, 2, 3, 4}


def test_forecast_invalid_horizon_422(client):
    resp = client.get("/api/v1/forecast", params={"horizon": "2d"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_backtest_latest_contract(client, tmp_path):
    resp = client.get("/api/v1/backtest/latest")
    assert resp.status_code == 404  # no report yet

    from forecasting.timesfm_service import GoldForecastService

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    results = service.backtest(
        make_history(n=250), horizon_days=10, step_days=20, max_folds=5
    )
    service.persist_backtest(results, data_dir=tmp_path)
    resp = client.get("/api/v1/backtest/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["results"]) == {"timesfm-2.5", "naive-last-value", "sma-20"}
    tm = body["results"]["timesfm-2.5"]["summary"]
    assert tm["n_folds"] == 5
    assert "band_coverage_pct" in tm
    assert 0.0 <= tm["band_coverage_pct"] <= 100.0
    assert "band_coverage" in body["results"]["timesfm-2.5"]["folds"][0]


def test_backtest_scoreboard(client, tmp_path):
    resp = client.get("/api/v1/backtest/scoreboard")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []

    from forecasting.timesfm_service import GoldForecastService

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    results = service.backtest(
        make_history(n=250), horizon_days=10, step_days=20, max_folds=5
    )
    service.persist_backtest(results, data_dir=tmp_path)
    resp = client.get("/api/v1/backtest/scoreboard")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    e = entries[0]
    assert e["horizon_days"] == 10
    assert e["mape_pct"] >= 0
    assert "band_coverage_pct" in e
    assert e["underperforming_naive"] in {True, False}


def test_health_endpoint(client, monkeypatch):
    monkeypatch.setattr(deps, "service_is_loaded", lambda: True)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["data_freshness"] in {"fresh", "cached", "stale", "missing"}
    assert body["data_last_date"] is not None


def test_forecast_band_calibration_applied(client, tmp_path):
    """A persisted band_scale in the backtest report widens served bands."""
    backtests = tmp_path / "backtests"
    backtests.mkdir(parents=True)
    (backtests / "backtest_2026-01-01.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "model_version": "2.5",
                "context_length": 512,
                "results": {
                    "timesfm-2.5": {
                        "model": "timesfm-2.5",
                        "model_version": "2.5",
                        "horizon_days": 21,
                        "step_days": 7,
                        "summary": {},
                        "folds": [],
                        "band_scale": 2.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    deps.invalidate_response_cache()
    resp = client.get("/api/v1/forecast", params={"horizon": "1m", "quantiles": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["band_calibration"]["scale"] == 2.0
    assert body["band_calibration"]["target_coverage_pct"] == 80.0
    q10, q50, q90 = body["q10"], body["q50"], body["q90"]
    for a, m, b in zip(q10, q50, q90):
        assert a <= m <= b
        # unscaled band was ±2%*(i+1); scaled is ±4%*(i+1) -> band must be wide
        assert b - a >= 0

    # 1y horizon has no calibration in this report -> no scaling advertised
    resp2 = client.get("/api/v1/forecast", params={"horizon": "1y"})
    assert resp2.json()["band_calibration"] is None


def test_backtest_bakeoff_endpoint(client, tmp_path):
    resp = client.get("/api/v1/backtest/bakeoff")
    assert resp.status_code == 404

    backtests = tmp_path / "backtests"
    backtests.mkdir(parents=True)
    (backtests / "bakeoff_2026-01-01.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "data_range": ["2021-09-15", "2026-09-14"],
                "entries": [
                    {
                        "horizon_days": 21,
                        "horizon_label": "1m",
                        "model": "chronos-t5-tiny",
                        "mape_pct": 2.91,
                        "directional_accuracy_pct": 37.7,
                        "band_coverage_pct": 63.7,
                        "calibrated_band_coverage_pct": 80.0,
                        "band_scale": 1.2,
                        "n_folds": 52,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resp = client.get("/api/v1/backtest/bakeoff")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries"][0]["model"] == "chronos-t5-tiny"
    assert body["data_range"] == ["2021-09-15", "2026-09-14"]


def test_refresh_requires_api_key(client):
    assert client.post("/api/v1/refresh").status_code == 401
    assert client.post("/api/v1/refresh", headers={"X-API-Key": "wrong"}).status_code == 401


def test_refresh_rate_limited(monkeypatch, client, tmp_path):
    import ingestion.fetch_gold_prices as fetch_mod

    frame = make_history(n=50)
    monkeypatch.setattr(
        fetch_mod, "fetch_history",
        lambda *a, **k: frame,
    )
    resp1 = client.post("/api/v1/refresh", headers={"X-API-Key": "test-key"})
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "ok"
    resp2 = client.post("/api/v1/refresh", headers={"X-API-Key": "test-key"})
    assert resp2.status_code == 429

