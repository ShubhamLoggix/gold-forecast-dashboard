"""Offline tests for the forecast accountability log and Telegram alerts."""

import datetime as dt

import pandas as pd
import pytest

from config import settings
from tests.conftest import make_history


@pytest.fixture
def stub_service():
    from tests.test_api import StubModel
    import api.main as main_mod

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5
    return service


def test_log_daily_forecasts_writes_and_dedupes(stub_service, isolated_settings):
    from forecasting.accountability import load_forecast_log, log_daily_forecasts

    history = make_history(n=400)
    n1 = log_daily_forecasts(stub_service, history, data_dir=isolated_settings.data_dir)
    assert n1 > 0
    log = load_forecast_log(isolated_settings.data_dir)
    assert set(log["horizon_days"]) <= {5, 21, 63, 126, 252}
    n2 = log_daily_forecasts(stub_service, history, data_dir=isolated_settings.data_dir)
    assert n2 == n1  # dedupe keeps the log stable across same-day reruns


def _synthetic_log(tmp_path: "object") -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    return pd.DataFrame(
        {
            "made_at_utc": [now] * 4,
            "origin_date": [pd.Timestamp("2026-09-10")] * 4,
            "horizon_days": [5, 5, 21, 21],
            "target_date": [
                pd.Timestamp("2026-09-11"),
                pd.Timestamp("2026-09-14"),
                pd.Timestamp("2026-10-12"),
                pd.Timestamp("2026-10-13"),
            ],
            "point": [2001.0, 2002.0, 2050.0, 2051.0],
            "q10": [1980.0, 1981.0, 2000.0, 2001.0],
            "q90": [2030.0, 2031.0, 2100.0, 2101.0],
            "model_version": ["2.5"] * 4,
        }
    )


def test_score_forecasts_scores_realized(isolated_settings):
    from forecasting.accountability import score_forecasts

    history = make_history(n=400)  # dates 2024-01-01 .. ~2025-09, closes ~2000-2100
    log = _synthetic_log(isolated_settings)
    # Two targets inside history (2026-09-11/14 are beyond history's end; shift
    # onto known history dates for the scored case).
    log.loc[0, "target_date"] = history["date"].iloc[10]
    log.loc[1, "target_date"] = history["date"].iloc[10]
    out = isolated_settings.data_dir / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    log.to_parquet(out / "forecast_log.parquet", index=False)

    report = score_forecasts(history, data_dir=isolated_settings.data_dir)
    h5 = next(e for e in report["per_horizon"] if e["horizon_days"] == 5)
    assert h5["n_scored"] == 2
    assert h5["n_pending"] == 0
    assert 0.0 <= h5["mape_pct"] <= 100.0
    assert 0.0 <= h5["band_coverage_pct"] <= 100.0
    h21 = next(e for e in report["per_horizon"] if e["horizon_days"] == 21)
    assert h21["n_scored"] == 0
    assert h21["n_pending"] == 2
    assert report["recent"], "recent evaluated points expected"


def test_alert_targets_crud(isolated_settings):
    from alerts.telegram import add_target, list_targets, remove_target

    t = add_target("22k", "10gram", "inr", "<=", 120000, data_dir=isolated_settings.data_dir)
    assert t["id"]
    assert len(list_targets(isolated_settings.data_dir)) == 1
    assert remove_target(t["id"], data_dir=isolated_settings.data_dir) is True
    assert list_targets(isolated_settings.data_dir) == []
    assert remove_target("missing", data_dir=isolated_settings.data_dir) is False


def test_evaluate_targets_triggers_once(isolated_settings, monkeypatch):
    from alerts import telegram as tg

    sent: list[str] = []
    monkeypatch.setattr(tg, "send_message", lambda text: sent.append(text) or True)

    history = make_history(n=30, base=2000.0)  # closes near 2000 USD/oz
    t_low = tg.add_target("22k", "10gram", "inr", "<=", 1000, data_dir=isolated_settings.data_dir)
    t_high = tg.add_target("22k", "10gram", "inr", ">=", 50, data_dir=isolated_settings.data_dir)
    rate = 83.0
    hit = tg.evaluate_targets(history, rate, data_dir=isolated_settings.data_dir)
    ids = {h["id"] for h in hit}
    assert t_low["id"] not in ids
    assert t_high["id"] in ids
    hit2 = tg.evaluate_targets(history, rate, data_dir=isolated_settings.data_dir)
    assert t_high["id"] not in {h["id"] for h in hit2}
    assert len(sent) == 1


def test_daily_digest_disabled_without_config(monkeypatch, isolated_settings):
    from alerts import telegram as tg

    monkeypatch.setattr(settings, "telegram_bot_token", "")
    assert tg.daily_digest(make_history(n=30), 83.0, dt.date(2026, 9, 15), None) is False


def test_alert_api_endpoints(isolated_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    import api.deps as deps

    monkeypatch.setattr(deps, "bootstrap_if_needed", lambda: None)
    monkeypatch.setattr(settings, "api_key", "test-key")
    with TestClient(app) as client:
        assert client.get("/api/v1/alerts/targets").status_code == 200
        unauth = client.post(
            "/api/v1/alerts/targets",
            json={"op": "<=", "price": 1000, "currency": "inr", "karat": "22k", "unit": "10gram"},
        )
        assert unauth.status_code == 401
        created = client.post(
            "/api/v1/alerts/targets",
            headers={"X-API-Key": "test-key"},
            json={"op": ">=", "price": 5000, "currency": "inr", "karat": "24k", "unit": "gram"},
        )
        assert created.status_code == 200
        tid = created.json()["created"]["id"]
        assert client.get("/api/v1/alerts/targets").json()["targets"]
        assert client.delete(
            f"/api/v1/alerts/targets/{tid}", headers={"X-API-Key": "test-key"}
        ).json()["deleted"] is True
        assert client.delete(
            f"/api/v1/alerts/targets/{tid}", headers={"X-API-Key": "test-key"}
        ).status_code == 404


def test_accountability_endpoint_empty(isolated_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    import api.deps as deps

    monkeypatch.setattr(deps, "bootstrap_if_needed", lambda: None)
    monkeypatch.setattr(deps, "get_history", lambda: make_history(n=400))

    with TestClient(app) as client:
        resp = client.get("/api/v1/accountability")
        assert resp.status_code == 200
        assert resp.json()["per_horizon"] == []
