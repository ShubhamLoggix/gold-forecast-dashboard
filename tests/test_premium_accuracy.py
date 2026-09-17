"""Tests for the premium (retail-COMEX) series and the accuracy-tracker
extensions (direction scoring, rolling windows, per-series breakdown)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_history


def _seed_premium_inputs(data_dir, n_bullion=400, n_fx=200, first_live="2026-09-11"):
    """Write synthetic gold + FX + India cache so the premium series builds."""
    from ingestion.fetch_gold_prices import CANONICAL_FILENAME
    from ingestion.india_rates import INDIA_RATES_FILENAME
    from ingestion.usdinr import USDINR_CANONICAL_FILENAME

    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    gold = make_history(n=n_bullion, start="2024-01-01", base=2400.0)
    gold[["date", "close"]].to_parquet(
        processed / CANONICAL_FILENAME, index=False
    )

    fx_dates = pd.bdate_range("2024-01-01", periods=n_fx)
    fx = pd.DataFrame(
        {
            "date": fx_dates,
            "close": np.linspace(82.0, 84.0, n_fx),
        }
    )
    fx["close"] = fx["close"].astype(float)
    fx.to_parquet(processed / USDINR_CANONICAL_FILENAME, index=False)

    def _bullion_pg(d):
        row = gold[gold["date"] <= pd.Timestamp(d)].iloc[-1]
        fxrow = fx[fx["date"] <= pd.Timestamp(d)].iloc[-1]
        return (
            float(row["close"])
            * float(fxrow["close"])
            / 31.1034768
            * 0.916  # 22k purity
        )

    rows = []
    est_dates = pd.bdate_range("2024-01-20", periods=30)
    for d in est_dates:
        b = _bullion_pg(d)
        rows.append(
            {
                "date": pd.Timestamp(d).normalize(),
                "city_slug": "pune",
                "city": "Pune",
                "source": "comex_converted",
                "price_24k_pg": b / 0.916 * 0.999 * 1.15,
                "price_22k_pg": b * 1.15,
                "price_18k_pg": b / 0.916 * 0.75 * 1.15,
            }
        )
    live_dates = pd.bdate_range(first_live, periods=5)
    for i, d in enumerate(live_dates):
        b = _bullion_pg(d)
        rows.append(
            {
                "date": pd.Timestamp(d).normalize(),
                "city_slug": "pune",
                "city": "Pune",
                "source": "groww_live",
                "price_24k_pg": b / 0.916 * 0.999 * 1.10,
                "price_22k_pg": b * 1.10 + 50.0 * i,  # real premium drifts up
                "price_18k_pg": b / 0.916 * 0.75 * 1.10,
            }
        )
    pd.DataFrame(rows).to_parquet(processed / INDIA_RATES_FILENAME, index=False)


def test_premium_series_math_and_quality_tags(isolated_settings):
    from ingestion.premium_series import (
        build_premium_series,
        load_premium_series,
        premium_forecast_input,
        series_id,
    )

    _seed_premium_inputs(isolated_settings.data_dir)
    df = build_premium_series("pune", "22k", data_dir=isolated_settings.data_dir)
    assert len(df) == 35
    assert df["quality"].value_counts().to_dict() == {
        "estimated": 30,
        "real": 5,
    }
    assert (df["premium_pg"] == df["retail_pg"] - df["bullion_pg"]).all()
    # Estimated rows: premium == bullion*(1.15 - 1) == bullion*0.15.
    est = df[df["quality"] == "estimated"].iloc[0]
    assert np.isclose(est["premium_pg"], est["bullion_pg"] * 0.15, rtol=1e-9)
    # Real premiums are NOT a fixed fraction of bullion (that's the point).
    real = df[df["quality"] == "real"].iloc[-1]
    assert not np.isclose(real["premium_pg"], real["bullion_pg"] * 0.15)

    inp = premium_forecast_input("pune", "22k", data_dir=isolated_settings.data_dir)
    assert list(inp.columns) == ["date", "close"]
    assert pd.isna(inp["close"]).sum() == 0
    assert (inp["close"] > 0).all()
    assert series_id("pune", "22k") == "premium:pune:22k"
    load_premium_series("pune", "22k", data_dir=isolated_settings.data_dir)


def test_premium_missing_series_raises(isolated_settings):
    from ingestion.premium_series import load_premium_series

    with pytest.raises(FileNotFoundError):
        load_premium_series("pune", "18k", data_dir=isolated_settings.data_dir)


def test_score_forecasts_premium_series_direction_and_windows(isolated_settings, monkeypatch):
    import api.main as main_mod
    import ingestion.premium_series as ps
    from forecasting.accountability import (
        load_forecast_log,
        log_daily_forecasts,
        score_forecasts,
    )
    from tests.test_api import StubModel

    _seed_premium_inputs(isolated_settings.data_dir)
    ps.build_premium_series("pune", "22k", data_dir=isolated_settings.data_dir)

    service = main_mod.GoldForecastService(model_version="2.5")
    service._model = StubModel()
    service._predict_fn = service._predict_2p5

    # Log premium forecasts (series + horizon overlap so scoring occurs).
    premium_input = ps.premium_forecast_input("pune", "22k", data_dir=isolated_settings.data_dir)
    n = log_daily_forecasts(
        service,
        premium_input,
        data_dir=isolated_settings.data_dir,
        series="premium:pune:22k",
    )
    assert n > 0
    log = load_forecast_log(isolated_settings.data_dir)
    assert (log["series"] == "premium:pune:22k").all()
    assert log["point"].notna().all()

    # Give the log rows whose target_date maps onto actual premium rows so some
    # get scored, then re-score.
    scorer = score_forecasts(
        make_history(n=100), data_dir=isolated_settings.data_dir
    )
    assert "premium:pune:22k" in scorer["per_series"]
    entry = scorer["per_series"]["premium:pune:22k"][0]
    for field in (
        "horizon_days",
        "n_scored",
        "mape_pct",
        "mae",
        "directional_acc_pct",
        "band_coverage_pct",
        "windows",
    ):
        assert field in entry
    assert set(entry["windows"].keys()) == {"7", "30", "90"}
    assert isinstance(entry["windows"]["30"]["mape_pct"], float)


def test_score_forecasts_new_fields_on_gold(isolated_settings):
    from forecasting.accountability import score_forecasts

    now = dt.datetime.now(dt.timezone.utc)
    history = make_history(n=400)
    log = pd.DataFrame(
        {
            "made_at_utc": [now, now],
            "origin_date": [history["date"].iloc[0], history["date"].iloc[0]],
            "horizon_days": [5, 21],
            "target_date": [history["date"].iloc[10], history["date"].iloc[30]],
            "series": ["gold_comex", "gold_comex"],
            "point": [2005.0, 2050.0],
            "q10": [1980.0, 2000.0],
            "q90": [2030.0, 2100.0],
            "model_version": ["2.5", "2.5"],
        }
    )
    out = isolated_settings.data_dir / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    log.to_parquet(out / "forecast_log.parquet", index=False)

    report = score_forecasts(history, data_dir=isolated_settings.data_dir)
    assert report["per_series"]["gold_comex"]
    h5 = next(e for e in report["per_series"]["gold_comex"] if e["horizon_days"] == 5)
    assert 0 <= h5["directional_acc_pct"] <= 100
    assert isinstance(h5["mae"], float)
    assert "7" in h5["windows"]
    assert all(p["series"] == "gold_comex" for p in report["recent"])
    assert report["generated_at"]


def test_premium_endpoint_502_without_series(isolated_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    import api.deps as deps

    monkeypatch.setattr(deps, "bootstrap_if_needed", lambda: None)

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/premium/forecast",
            params={"city": "pune", "karat": "22k", "horizon": "1m"},
        )
        # No India cache / premium series in the isolated data dir -> 502, not 500.
        assert resp.status_code == 502
