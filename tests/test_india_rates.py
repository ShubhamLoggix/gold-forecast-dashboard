"""Tests for the India city retail rates ingestion (Groww source).

All tests run offline: the Groww HTML/JSON layer is stubbed.
"""

import datetime as dt

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.main as main_mod
from api.main import app
from config import settings
from tests.conftest import make_history
from tests.test_api import StubModel


def _gold_rate_data():
    return {
        "physicalGoldRate": {
            "pune": {
                "date": "2026-09-15",
                "percentageChange": {"TWENTY_FOUR": -0.597, "TWENTY_TWO": -0.594, "EIGHTEEN": -0.597},
                "price": {"TWENTY_FOUR": 15317, "TWENTY_TWO": 14041, "EIGHTEEN": 11488},
                "priceLocation": "Pune",
            },
            "mumbai": {
                "date": "2026-09-15",
                "percentageChange": {"TWENTY_FOUR": -0.597},
                "price": {"TWENTY_FOUR": 15317, "TWENTY_TWO": 14041, "EIGHTEEN": 11488},
                "priceLocation": "Mumbai",
            },
            "india": {
                "date": "2026-09-15",
                "percentageChange": {},
                "price": {"TWENTY_FOUR": 15090.2, "TWENTY_TWO": 13833, "EIGHTEEN": 11318},
                "priceLocation": "India",
            },
        },
        "daySummary": {
            "pune": {
                "2026-09-15": {
                    "date": "2026-09-15",
                    "price": {"TWENTY_FOUR": 15317, "TWENTY_TWO": 14041, "EIGHTEEN": 11488},
                    "priceLocation": "Pune",
                },
                "2026-09-14": {
                    "date": "2026-09-14",
                    "price": {"TWENTY_FOUR": 15409, "TWENTY_TWO": 14125, "EIGHTEEN": 11557},
                    "priceLocation": "Pune",
                },
            },
        },
        "locations": [
            {"name": "Pune", "slug": "pune", "type": "city", "stateName": "Maharashtra"},
            {"name": "Delhi", "slug": "delhi", "type": "union_territory", "stateName": "Delhi"},
        ],
    }


@pytest.fixture
def stub_groww(monkeypatch):
    import ingestion.india_rates as mod

    grd = _gold_rate_data()
    monkeypatch.setattr(mod, "_fetch_gold_rate_data", lambda url: grd)
    monkeypatch.setattr(mod, "_fetch_hub", lambda: grd)
    return mod


def test_snapshot_frame_parses_cities():
    import ingestion.india_rates as mod

    frame = mod.snapshot_frame(_gold_rate_data())
    assert len(frame) == 3
    pune = frame[frame["city_slug"] == "pune"].iloc[0]
    assert pune["price_24k_pg"] == 15317
    assert pune["city"] == "Pune"
    assert pune["date"] == pd.Timestamp("2026-09-15")


def test_city_history_frame_sorted_and_typed():
    import ingestion.india_rates as mod

    frame = mod.city_history_frame(_gold_rate_data(), "pune")
    assert len(frame) == 2
    assert list(frame["date"]) == list(sorted(frame["date"]))
    assert frame.iloc[0]["price_24k_pg"] == 15409


def test_get_city_rates_uses_cache_and_fetches_history(stub_groww, tmp_data_dir):
    data = stub_groww.get_city_rates("pune", data_dir=tmp_data_dir)
    assert data["city"] == "Pune"
    assert data["per_gram"]["24k"] == 15317
    assert data["date"] == dt.date(2026, 9, 15)
    assert [h["date"] for h in data["history"]] == ["2026-09-14", "2026-09-15"]
    cache = stub_groww.load_cache(tmp_data_dir)
    assert (cache["city_slug"] == "pune").sum() == 2


def test_get_city_rates_unknown_slug(stub_groww):
    with pytest.raises(KeyError):
        stub_groww.get_city_rates("atlantis")


def test_cities_catalog():
    import ingestion.india_rates as mod

    catalog = mod.cities_catalog(_gold_rate_data())
    assert {c["slug"] for c in catalog} == {"pune", "delhi"}
    assert catalog[0]["state_name"] == "Maharashtra"


def test_backfill_tags_sources_and_preserves_live(isolated_settings, monkeypatch):
    import ingestion.fetch_gold_prices as fgp
    import ingestion.india_rates as mod
    import ingestion.usdinr as usd_mod
    from tests.conftest import make_history

    history = make_history(n=300)  # 2024-01-01 .. ~2025-07
    monkeypatch.setattr(fgp, "load_canonical", lambda *a, **k: history)
    monkeypatch.setattr(
        usd_mod, "load_usdinr",
        lambda **k: pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=300),
                "open": 83.0, "high": 83.0, "low": 83.0, "close": 83.2,
                "volume": 0.0, "source": "fixture",
            }
        ),
    )

    processed = isolated_settings.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    live = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-15")],
            "city_slug": ["pune", "pune"],
            "city": ["Pune", "Pune"],
            "price_24k_pg": [15409.0, 15317.0],
            "price_22k_pg": [14125.0, 14041.0],
            "price_18k_pg": [11557.0, 11488.0],
            "source": ["groww_live", "groww_live"],
        }
    )
    live.to_parquet(processed / "india_gold_rates.parquet", index=False)

    result = mod.backfill_from_comex(city_slug="pune", data_dir=isolated_settings.data_dir)
    assert result["backfilled"] > 0
    cache = mod.load_cache(isolated_settings.data_dir)
    est = cache[cache["source"] == "comex_converted"]
    live_rows = cache[cache["source"] == "groww_live"]
    assert len(est) > 0
    assert est["date"].max() < pd.Timestamp("2026-09-14")  # never crosses live data
    assert len(live_rows) == 2
    assert (est["price_24k_pg"] > 0).all()

    data = mod.get_city_rates("pune", data_dir=isolated_settings.data_dir)
    sources = {p["source"] for p in data["history"]}
    assert "comex_converted" in sources
    assert "groww_live" in sources


def test_india_api_endpoints(stub_groww, monkeypatch, tmp_path):
    frame = make_history(n=300)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    service = main_mod.GoldForecastService(model_version="2.5")
    monkeypatch.setattr(deps, "get_history", lambda *a, **k: frame)
    monkeypatch.setattr(deps, "get_service", lambda: service)
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(deps, "bootstrap_if_needed", lambda: None)
    deps.invalidate_response_cache()
    with TestClient(app) as client:
        cities = client.get("/api/v1/india/cities")
        assert cities.status_code == 200
        slugs = [c["slug"] for c in cities.json()["cities"]]
        assert slugs == ["delhi", "pune"]  # sorted by name

        rates = client.get("/api/v1/india/rates", params={"city": "pune", "unit": "10gram"})
        assert rates.status_code == 200
        body = rates.json()
        assert body["city"] == "Pune"
        assert body["per_10g"]["24k"] == 153170
        assert body["per_gram"]["22k"] == 14041
        assert len(body["history"]) == 2
        assert "Groww" in body["disclaimer"] or "groww" in body["disclaimer"]

        missing = client.get("/api/v1/india/rates", params={"city": "atlantis"})
        assert missing.status_code == 404

        bad = client.get("/api/v1/india/rates", params={"city": "PUNE!"})
        assert bad.status_code == 422


def test_india_forecast_endpoint(stub_groww, monkeypatch, tmp_path):
    frame = make_history(n=300)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    frame.to_parquet(processed / "gold_prices_daily.parquet", index=False)
    fx = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=300),
            "open": 83.0,
            "high": 83.5,
            "low": 82.5,
            "close": 83.2,
            "volume": 0.0,
            "source": "fixture",
        }
    )
    fx.to_parquet(processed / "usdinr_daily.parquet", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    monkeypatch.setattr(deps, "get_history", lambda *a, **k: frame)
    monkeypatch.setattr(deps, "get_service", lambda: service)
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(deps, "bootstrap_if_needed", lambda: None)
    deps.invalidate_response_cache()
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/india/forecast",
            params={"city": "pune", "horizon": "1m", "karat": "24k", "unit": "10gram"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["point"]) == 21
        assert body["premium_ratio"] > 1.0
        assert body["bullion_history_last_close"] > 0
        # retail last close is today's per-10g quote
        assert body["history_last_close"] == 153170
        # all forecast values scaled up by the premium
        assert min(body["point"]) > 100000
        assert min(body["q10"]) <= min(body["q50"]) <= min(body["q90"])
        assert "ESTIMATED" in body["disclaimer"]

        missing = client.get("/api/v1/india/forecast", params={"city": "atlantis"})
        assert missing.status_code == 404

