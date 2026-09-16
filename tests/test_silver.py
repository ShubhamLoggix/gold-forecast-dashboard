"""Silver (SI=F) ingestion, conversion, and API metal-parameter tests."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.main as main_mod
from api.main import app
from config import settings
from tests.conftest import make_history


def _silver_frame(n: int = 300) -> pd.DataFrame:
    """Synthetic silver-like daily series (prices near $30/oz). FIXTURE ONLY."""
    rng = np.random.default_rng(7)
    close = 30.0 + np.cumsum(rng.normal(0.01, 0.15, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": close + rng.normal(0, 0.05, n),
            "high": close + np.abs(rng.normal(0, 0.1, n)),
            "low": close - np.abs(rng.normal(0, 0.1, n)),
            "close": close,
            "volume": rng.integers(0, 50_000, n).astype(float),
            "source": "yfinance:SI=F",
        }
    )


def test_silver_conversion_math():
    from conversion import convert_silver_series

    # 30 USD/oz at 95 INR/USD, 925 fineness, per kg:
    # 30 * 95 / 31.1034768 * 0.925 * 1000
    expected = 30 * 95 / 31.1034768 * 0.925 * 1000
    out = convert_silver_series(np.array([30.0]), 95.0, "925", "kg")
    assert abs(out[0] - expected) < 1e-6
    with pytest.raises(ValueError):
        convert_silver_series(np.array([30.0]), 95.0, "22k", "kg")  # gold karat on silver
    with pytest.raises(ValueError):
        convert_silver_series(np.array([30.0]), 95.0, "999", "10gram")


def test_silver_ingestion_roundtrip(isolated_settings, monkeypatch):
    import ingestion.fetch_silver_prices as sp

    monkeypatch.setattr(sp, "_fetch_yfinance", lambda s, e: _silver_frame(300))
    frame = sp.update_canonical(
        pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date(),
        data_dir=isolated_settings.data_dir,
    )
    assert len(frame) > 200
    loaded = sp.load_canonical(isolated_settings.data_dir)
    assert set(loaded["source"]) == {"yfinance:SI=F"}
    assert loaded["close"].iloc[0] > 0


def test_cross_metal_units_rejected_with_422(isolated_settings, monkeypatch, tmp_path):
    """The backend must reject the other metal's unit with a clear 422.

    This pins the contract the frontend's unit-mismatch fix relies on: the UI
    must never send e.g. silver's `kg` to the gold endpoint. Regression origin:
    the Gold<->Silver unit-carryover bug on the frontend.
    """
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    make_history(n=300).to_parquet(processed / "gold_prices_daily.parquet", index=False)
    _silver_frame(300).to_parquet(processed / "silver_prices_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    service = main_mod.GoldForecastService(model_version="2.5")
    monkeypatch.setattr(deps, "get_service", lambda: service)
    deps.invalidate_response_cache()
    with TestClient(app) as client:
        assert client.get(
            "/api/v1/history", params={"metal": "gold", "unit": "kg", "currency": "inr"}
        ).status_code == 422
        assert client.get(
            "/api/v1/history", params={"metal": "silver", "unit": "10gram", "currency": "inr"}
        ).status_code == 422
        assert client.get(
            "/api/v1/forecast", params={"metal": "gold", "unit": "kg", "currency": "inr", "horizon": "1w"}
        ).status_code == 422
        assert client.get(
            "/api/v1/forecast", params={"metal": "silver", "unit": "10gram", "currency": "inr", "horizon": "1w"}
        ).status_code == 422


def test_silver_api_endpoints(isolated_settings, monkeypatch, tmp_path):
    from tests.test_api import StubModel

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _silver_frame(300).to_parquet(processed / "silver_prices_daily.parquet", index=False)
    make_history(n=300).to_parquet(processed / "gold_prices_daily.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=300),
            "open": 83.0, "high": 83.0, "low": 83.0, "close": 83.2,
            "volume": 0.0, "source": "fixture",
        }
    ).to_parquet(processed / "usdinr_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    monkeypatch.setattr(deps, "get_service", lambda: service)
    monkeypatch.setattr(settings, "api_key", "test-key")
    deps.invalidate_response_cache()
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/history",
            params={"metal": "silver", "granularity": "day", "unit": "kg"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["metal"] == "silver"
        assert body["source"] == "yfinance:SI=F"
        assert body["karat"] is None

        # invalid unit for silver -> 422
        bad = client.get(
            "/api/v1/history",
            params={"metal": "silver", "currency": "inr", "unit": "10gram"},
        )
        assert bad.status_code == 422

        # gold rejects fineness
        bad2 = client.get(
            "/api/v1/history",
            params={"metal": "gold", "fineness": "925"},
        )
        assert bad2.status_code == 422

        fc = client.get(
            "/api/v1/forecast",
            params={"metal": "silver", "currency": "inr", "fineness": "958", "unit": "kg", "horizon": "1w"},
        )
        assert fc.status_code == 200
        fbody = fc.json()
        assert fbody["metal"] == "silver"
        assert fbody["fineness"] == "958"
        assert fbody["unit"] == "kg"
        assert len(fbody["point"]) == 5
