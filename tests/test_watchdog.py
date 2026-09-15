"""Model-drift watchdog + security hardening tests."""

import json

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.main as main_mod
from api.main import app
from config import settings
from forecasting.timesfm_service import BacktestResult, evaluate_drift
from tests.conftest import make_history


class StubModel:
    def forecast(self, horizon, inputs):
        ctx = float(inputs[0][-1])
        point = [ctx * (1 + 0.001 * (i + 1)) for i in range(horizon)]
        quant = __import__("numpy").zeros((1, horizon, 10))
        for i in range(horizon):
            band = ctx * 0.02 * (i + 1)
            quant[0, i, 1] = point[i] - band
            quant[0, i, 5] = point[i]
            quant[0, i, 9] = point[i] + band
            quant[0, i, 0] = point[i]
        return [point], quant


def _stub_service():
    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    return service


def test_evaluate_drift_flags_more_than_5pp_below():
    results = {
        "timesfm-2.5": _result("timesfm-2.5", dir_acc=40.0),
        "naive-last-value": _result("naive-last-value", dir_acc=50.0),
    }
    drift = evaluate_drift(results)
    assert drift["delta_pp"] == -10.0
    assert drift["underperforming"] is True


def test_evaluate_drift_ok_when_close():
    results = {
        "timesfm-2.5": _result("timesfm-2.5", dir_acc=48.0),
        "naive-last-value": _result("naive-last-value", dir_acc=50.0),
    }
    drift = evaluate_drift(results)
    assert drift["underperforming"] is False


def test_evaluate_drift_none_without_summaries():
    results = {
        "timesfm-2.5": _result("timesfm-2.5", dir_acc=None),
        "naive-last-value": _result("naive-last-value", dir_acc=None),
    }
    assert evaluate_drift(results) is None


def _result(name: str, dir_acc: float | None) -> BacktestResult:
    summary = {} if dir_acc is None else {"directional_accuracy_pct": dir_acc}
    return BacktestResult(name, "2.5" if name.startswith("timesfm") else "-",
                          10, 5, [], summary)


def test_backtest_persists_drift_block(tmp_data_dir):
    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    results = service.backtest(make_history(n=250), horizon_days=10,
                               step_days=20, max_folds=5)
    path = service.persist_backtest(results, data_dir=tmp_data_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "model_drift" in payload
    assert set(payload["model_drift"]) == {
        "delta_pp", "timesfm_dir_acc_pct", "naive_dir_acc_pct", "underperforming",
    }


def test_health_reports_drift_flag(monkeypatch, tmp_path):
    frame = make_history(n=300)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    # Write a backtest report where TimesFM is 8 pp below the baseline.
    drift = {"delta_pp": -8.0, "timesfm_dir_acc_pct": 42.0,
             "naive_dir_acc_pct": 50.0, "underperforming": True}
    backtests = tmp_path / "backtests"
    backtests.mkdir()
    (backtests / "backtest_2026-09-15.json").write_text(
        json.dumps({"generated_at": "2026-09-15T00:00:00Z",
                    "model_version": "2.5", "context_length": 512,
                    "model_drift": drift, "results": {}}),
        encoding="utf-8",
    )
    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    monkeypatch.setattr(deps, "get_history", lambda: frame)
    monkeypatch.setattr(deps, "get_service", lambda: service)
    monkeypatch.setattr(deps, "bootstrap_error", lambda: None)
    deps.invalidate_response_cache()
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/v1/health")
    body = resp.json()
    assert body["model_underperforming_baseline"] is True
    assert body["status"] == "degraded"
    assert body["source_status"]["drift_delta_pp"] == "-8.0"


def test_health_does_not_trigger_model_load(monkeypatch, tmp_path):
    frame = make_history(n=100)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(deps, "get_history", lambda: frame)
    monkeypatch.setattr(deps, "bootstrap_error", lambda: None)
    deps.invalidate_response_cache()

    called = {"get_service": False}
    def _boom():
        called["get_service"] = True
        raise AssertionError("health must not call get_service()")
    monkeypatch.setattr(deps, "get_service", _boom)
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert called["get_service"] is False
    assert resp.json()["model_loaded"] is False


def test_public_rate_limit_429(monkeypatch, tmp_path):
    frame = make_history(n=300)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(deps, "get_history", lambda: frame)
    # Tight limiter to keep the test fast.
    monkeypatch.setattr(
        main_mod, "public_rate_limiter",
        deps.RateLimiter(max_calls=3, period_seconds=60.0),
    )
    deps.invalidate_response_cache()
    with TestClient(main_mod.app) as client:
        codes = [
            client.get("/api/v1/history", params={"granularity": "day"}).status_code
            for _ in range(6)
        ]
    assert codes[:3] == [200, 200, 200]
    assert set(codes[3:]) == {429}


def test_oversized_body_rejected():
    with TestClient(main_mod.app) as client:
        resp = client.post(
            "/api/v1/refresh",
            headers={"X-API-Key": "test-key", "Content-Length": str(2 * 1024 * 1024)},
        )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
