"""Ingestion tests: mocked sources only (no network)."""

import datetime as dt
import logging

import pandas as pd
import pytest

from ingestion.fetch_gold_prices import (
    fetch_history,
    update_canonical,
)
from ingestion.validation import validate_and_clean
from tests.conftest import make_history


@pytest.fixture
def fake_source(monkeypatch):
    """Replace the yfinance adapter with a fixture-backed function."""
    calls = {"n": 0}
    frame = make_history(n=60, start="2024-01-01")
    # Fixture spans 2024-01-01..2024-03-27 (60 business days). Tests must
    # request ranges within that window or the cache coverage check fails.
    start_day = frame["date"].min().date()
    end_day = frame["date"].max().date()

    def _fake(start_date, end_date):
        calls["n"] += 1
        mask = (frame["date"] >= pd.Timestamp(start_date)) & (
            frame["date"] <= pd.Timestamp(end_date)
        )
        return frame.loc[mask].reset_index(drop=True)

    import ingestion.fetch_gold_prices as mod

    monkeypatch.setitem(mod._fetchers, "yfinance", _fake)
    return calls, start_day, end_day


def test_fetch_history_schema(fake_source, tmp_data_dir):
    calls, start, end = fake_source
    df = fetch_history(start, end, source="yfinance", data_dir=tmp_data_dir)
    assert list(df.columns) == [
        "date", "open", "high", "low", "close", "volume", "source",
    ]
    assert len(df) == 60
    assert df["date"].is_monotonic_increasing
    assert (df["source"] == "fixture").all()


def test_cache_hit_avoids_refetch(fake_source, tmp_data_dir):
    calls, start, end = fake_source
    fetch_history(start, end, source="yfinance", data_dir=tmp_data_dir)
    n_first = calls["n"]
    df2 = fetch_history(start, end, source="yfinance", data_dir=tmp_data_dir)
    assert calls["n"] == n_first  # served from cache
    assert len(df2) == 60


def test_force_refresh_refetches(fake_source, tmp_data_dir):
    calls, start, end = fake_source
    fetch_history(start, end, source="yfinance", data_dir=tmp_data_dir)
    fetch_history(
        start, end, source="yfinance", force_refresh=True, data_dir=tmp_data_dir
    )
    assert calls["n"] == 2


def test_cache_partial_range_refetches_only_missing(fake_source, tmp_data_dir):
    calls, start, end = fake_source
    fetch_history(
        start, dt.date(2024, 1, 31), source="yfinance", data_dir=tmp_data_dir
    )
    df2 = fetch_history(start, end, source="yfinance", data_dir=tmp_data_dir)
    # cache covers [start, jan31]; [feb1, end] fetched; merged = full fixture range
    assert calls["n"] == 2


def test_auto_falls_back_when_primary_fails(monkeypatch, tmp_data_dir):
    import config
    import ingestion.fetch_gold_prices as mod

    # metals_api is only part of the "auto" chain when an API key is configured
    monkeypatch.setattr(config.settings, "metals_api_key", "test-key")

    def _broken(start, end):
        raise RuntimeError("network down")

    monkeypatch.setitem(mod._fetchers, "yfinance", _broken)
    monkeypatch.setitem(mod._fetchers, "lbma", _broken)
    frame = make_history(n=20)
    monkeypatch.setitem(mod._fetchers, "metals_api", lambda s, e: frame)
    df = fetch_history(
        dt.date(2024, 1, 1), dt.date(2024, 1, 31), source="auto", data_dir=tmp_data_dir
    )
    assert len(df) == 20


def test_auto_without_api_key_skips_metals_api(monkeypatch, tmp_data_dir):
    import config
    import ingestion.fetch_gold_prices as mod

    assert config.settings.metals_api_key == ""
    def _broken(start, end):
        raise RuntimeError("network down")
    monkeypatch.setitem(mod._fetchers, "yfinance", _broken)
    monkeypatch.setitem(mod._fetchers, "lbma", _broken)
    with pytest.raises(RuntimeError, match="All sources failed"):
        fetch_history(
            dt.date(2024, 1, 1), dt.date(2024, 1, 31), source="auto", data_dir=tmp_data_dir
        )


def test_validate_removes_duplicates_and_negatives():
    frame = make_history(n=30)
    dup = pd.concat([frame, frame.iloc[[-5]]], ignore_index=True)
    dup.loc[0, "close"] = -1.0
    clean = validate_and_clean(dup)
    assert clean["close"].is_unique
    assert (clean["close"] > 0).all()
    # 30 rows + 1 duplicate (dropped) - 1 negative row (dropped)
    assert len(clean) == 29


def test_validate_drops_weekend_rows():
    frame = make_history(n=10)
    weekend = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-13"), "close": 2000.0, "open": 2000.0,
          "high": 2001.0, "low": 1999.0, "volume": 0.0, "source": "fixture"}]
    )
    cleaned = validate_and_clean(pd.concat([frame, weekend], ignore_index=True))
    assert (cleaned["date"].dt.dayofweek < 5).all()
    assert len(cleaned) == 10


def test_validate_warns_on_big_jump(caplog):
    frame = make_history(n=30).copy()
    frame.loc[10, "close"] = frame.loc[9, "close"] * 1.5  # +50% single-day
    with caplog.at_level(logging.WARNING, logger="gold_forecast.ingestion"):
        validate_and_clean(frame)
    assert any("bad tick" in r.message for r in caplog.records)


def test_validate_does_not_forward_fill_gaps(caplog):
    frame = make_history(n=40)
    drop_idx = frame.index[[15, 16, 17, 18]]
    gapped = frame.drop(drop_idx).reset_index(drop=True)
    with caplog.at_level(logging.INFO, logger="gold_forecast.ingestion"):
        clean = validate_and_clean(gapped)
    assert len(clean) == len(gapped)  # rows not synthesized
    assert any("weekday slots absent" in r.message for r in caplog.records)


def test_update_canonical_writes_processed_file(fake_source, tmp_data_dir):
    calls, start, end = fake_source
    df = update_canonical(
        start, end, source="yfinance", data_dir=tmp_data_dir
    )
    path = tmp_data_dir / "processed" / "gold_prices_daily.parquet"
    assert path.exists()
    reloaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, reloaded)
