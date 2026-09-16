"""API currency/karat/unit endpoint tests (stubbed model + fixture FX data)."""

import datetime as dt
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.main as main_mod
from api.main import app
from config import settings
from conversion import TROY_OZ_TO_GRAM
from tests.conftest import make_history


class StubModel:
    def forecast(self, horizon, inputs):
        last = float(inputs[0][-1])
        point = [last * (1 + 0.001 * (i + 1)) for i in range(horizon)]
        quant = __import__("numpy").zeros((1, horizon, 10))
        for i in range(horizon):
            band = last * 0.02 * (i + 1)
            quant[0, i, 1] = point[i] - band
            quant[0, i, 5] = point[i]
            quant[0, i, 9] = point[i] + band
            quant[0, i, 0] = point[i]
        return [point], quant


@pytest.fixture
def inr_client(monkeypatch, tmp_path):
    frame = make_history(n=300)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    # USDINR fixture series covering the gold fixture range, with a rate
    # STEP so per-date conversion is observable: 70 until 2024-10-01, 88 after.
    fx = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=320),
            "open": 70.0, "high": 70.0, "low": 70.0,
            "close": 70.0, "volume": 0.0, "source": "fixture",
        }
    )
    fx.loc[fx["date"] >= pd.Timestamp("2024-10-01"), ["open", "high", "low", "close"]] = 88.0
    fx.to_parquet(processed / "usdinr_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    monkeypatch.setattr(deps, "get_history", lambda *a, **k: frame)
    monkeypatch.setattr(deps, "get_service", lambda: service)
    monkeypatch.setattr(deps, "service_is_loaded", lambda: True)
    monkeypatch.setattr(deps, "bootstrap_error", lambda: None)
    deps.invalidate_response_cache()
    with TestClient(app) as c:
        yield c


def test_history_usd_default_unchanged(inr_client):
    resp = inr_client.get("/api/v1/history")
    body = resp.json()
    assert body["currency"] == "usd"
    assert body["karat"] is None and body["unit"] is None and body["rate"] is None
    first_close = body["points"][0]["close"]
    assert first_close == pytest.approx(make_history(n=300)["close"].iloc[0], rel=1e-9)


def test_history_inr_22k_per10gram_math(inr_client):
    resp = inr_client.get(
        "/api/v1/history",
        params={"currency": "inr", "karat": "22k", "unit": "10gram"},
    )
    body = resp.json()
    assert body["currency"] == "inr"
    assert body["karat"] == "22k" and body["unit"] == "10gram"
    assert body["rate"]["usd_inr_rate"] == 88.0
    assert body["rate"]["usd_inr_rate_date"] is not None
    assert body["rate"]["rate_may_be_stale"] is False
    assert "theoretical" in body["rate"]["disclaimer"]

    # Manually check the conversion of the first close — using the rate in
    # effect on THAT date (70.0), not the latest rate.
    usd_close = float(make_history(n=300)["close"].iloc[0])
    expected = usd_close * 70.0 / TROY_OZ_TO_GRAM * 0.916 * 10
    assert math.isclose(body["points"][0]["close"], expected, rel_tol=1e-9)
    # Last point uses the late rate (88.0).
    usd_last = float(make_history(n=300)["close"].iloc[-1])
    expected_last = usd_last * 88.0 / TROY_OZ_TO_GRAM * 0.916 * 10
    assert math.isclose(body["points"][-1]["close"], expected_last, rel_tol=1e-9)


def test_forecast_inr_18k_per_gram_math(inr_client):
    resp = inr_client.get(
        "/api/v1/forecast",
        params={"horizon": "1w", "currency": "inr", "karat": "18k", "unit": "gram"},
    )
    body = resp.json()
    assert body["currency"] == "inr" and body["karat"] == "18k" and body["unit"] == "gram"
    # The stub forecast starts at last*(1.001); last = 300th fixture close.
    # (Model path passes through float32, so use a realistic tolerance.)
    usd_last = float(make_history(n=300)["close"].iloc[-1])
    usd_point0 = usd_last * 1.001
    expected = usd_point0 * 88.0 / TROY_OZ_TO_GRAM * 0.750 * 1.0
    assert math.isclose(body["point"][0], expected, rel_tol=1e-5)
    # Baselines converted too: naive = last close, flat.
    naive0 = body["baselines"][0]["values"][0]
    assert math.isclose(naive0, usd_last * 88.0 / TROY_OZ_TO_GRAM * 0.750, rel_tol=1e-9)
    # Quantile ordering preserved under (positive) conversion
    assert body["q10"][0] <= body["q50"][0] <= body["q90"][0]
    assert body["history_last_close"] == pytest.approx(
        usd_last * 88.0 / TROY_OZ_TO_GRAM * 0.750, rel=1e-9
    )


def test_usd_request_ignores_karat(inr_client):
    resp = inr_client.get(
        "/api/v1/forecast", params={"currency": "usd", "karat": "18k", "unit": "gram"}
    )
    body = resp.json()
    # Values are plain USD/oz even though karat/unit were passed
    usd_last = float(make_history(n=300)["close"].iloc[-1])
    assert body["baselines"][0]["values"][0] == pytest.approx(usd_last, rel=1e-9)
    assert body["karat"] is None and body["unit"] is None and body["rate"] is None


def test_invalid_currency_or_karat_422(inr_client):
    assert inr_client.get("/api/v1/history", params={"currency": "gbp"}).status_code == 422
    assert inr_client.get("/api/v1/history", params={"karat": "14k"}).status_code == 422
    assert inr_client.get("/api/v1/history", params={"unit": "tola"}).status_code == 422


def test_missing_fx_series_503(monkeypatch, inr_client, tmp_path):
    # Remove the FX parquet -> 503 with a helpful message.
    (tmp_path / "processed" / "usdinr_daily.parquet").unlink()
    deps.invalidate_response_cache()
    resp = inr_client.get("/api/v1/history", params={"currency": "inr"})
    assert resp.status_code == 503
    assert "USDINR" in resp.json()["error"]["message"]

