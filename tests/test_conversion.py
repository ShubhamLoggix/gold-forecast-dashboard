"""Conversion + USDINR rate resolution tests (fixtures only, no network)."""

import datetime as dt
import math

import pandas as pd
import pytest

from conversion import (
    PURITY_FACTORS,
    TROY_OZ_TO_GRAM,
    UNIT_FACTORS,
    convert_series,
    usd_per_oz_to_inr,
    usd_per_troy_oz_to_inr_per_gram,
)
from ingestion.usdinr import latest_rate, resolve_rate_for_date


def test_purity_factors_match_spec():
    assert PURITY_FACTORS == {"24k": 0.999, "22k": 0.916, "18k": 0.750}
    assert math.isclose(TROY_OZ_TO_GRAM, 31.1034768)


def test_per_gram_conversion_math():
    # 24K: $4330.10/oz * 88 / 31.1034768 * 0.999 = INR/gram
    expected = 4330.10 * 88.0 / TROY_OZ_TO_GRAM * 0.999
    got = usd_per_troy_oz_to_inr_per_gram(4330.10, 88.0, "24k")
    assert math.isclose(got, expected, rel_tol=1e-12)
    assert got == pytest.approx(12238.75, abs=1.0)


def test_22k_per_10gram_conversion_math():
    # $4330.10/oz * 88 / 31.1034768 * 0.916 * 10
    expected = 4330.10 * 88.0 / TROY_OZ_TO_GRAM * 0.916 * 10
    got = usd_per_oz_to_inr(4330.10, 88.0, "22k", unit="10gram")
    assert math.isclose(got, expected, rel_tol=1e-12)
    # sanity: 22K must be cheaper than 24K for the same unit
    assert got < usd_per_oz_to_inr(4330.10, 88.0, "24k", unit="10gram")


def test_18k_is_0750_of_bullion():
    per_gram_24 = usd_per_troy_oz_to_inr_per_gram(4000.0, 90.0, "24k")
    per_gram_18 = usd_per_troy_oz_to_inr_per_gram(4000.0, 90.0, "18k")
    assert math.isclose(per_gram_18 / per_gram_24, 0.750 / 0.999)


def test_bad_karat_and_unit_raise():
    with pytest.raises(ValueError):
        usd_per_oz_to_inr(100.0, 88.0, "14k")
    with pytest.raises(ValueError):
        usd_per_oz_to_inr(100.0, 88.0, "24k", unit="tola")
    with pytest.raises(ValueError):
        usd_per_oz_to_inr(100.0, 0.0, "24k")


def test_convert_series_matches_scalar():
    vals = [4330.10, 4200.0, 4400.5]
    got = convert_series(vals, 88.0, "22k", "10gram")
    for v, g in zip(vals, got):
        assert math.isclose(g, usd_per_oz_to_inr(v, 88.0, "22k", "10gram"))


def _fx_frame(dates_and_rates: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d, _ in dates_and_rates],
            "open": [r for _, r in dates_and_rates],
            "high": [r for _, r in dates_and_rates],
            "low": [r for _, r in dates_and_rates],
            "close": [r for _, r in dates_and_rates],
            "volume": [0.0] * len(dates_and_rates),
            "source": "fixture",
        }
    )


def test_resolve_rate_uses_exact_date_when_available():
    rates = _fx_frame([("2026-09-10", 87.9), ("2026-09-14", 88.3)])
    out = resolve_rate_for_date(dt.date(2026, 9, 14), rates=rates)
    assert out == {"rate": 88.3, "rate_date": dt.date(2026, 9, 14),
                   "rate_may_be_stale": False}


def test_resolve_rate_falls_back_to_most_recent_prior():
    rates = _fx_frame([("2026-09-10", 87.9), ("2026-09-14", 88.3)])
    # Sep 15 (Mon) has no rate -> uses Sep 14's 88.3, 1 day gap: not stale
    out = resolve_rate_for_date(dt.date(2026, 9, 15), rates=rates)
    assert out["rate"] == 88.3
    assert out["rate_date"] == dt.date(2026, 9, 14)
    assert out["rate_may_be_stale"] is False


def test_resolve_rate_flags_stale_after_gap():
    rates = _fx_frame([("2026-09-04", 87.5)])
    # Target 8+ days after the last rate -> stale flag
    out = resolve_rate_for_date(dt.date(2026, 9, 15), rates=rates)
    assert out["rate"] == 87.5
    assert out["rate_may_be_stale"] is True


def test_resolve_rate_never_interpolates_forward():
    # Rate series ends BEFORE the target; must not use future rates either.
    rates = _fx_frame([("2026-09-04", 87.5), ("2026-09-18", 89.0)])
    out = resolve_rate_for_date(dt.date(2026, 9, 15), rates=rates)
    assert out["rate"] == 87.5  # last prior, not the future one


def test_rate_frame_pre_series_dates_fall_back_to_earliest_rate():
    from ingestion.usdinr import rate_frame_for_dates

    rates = _fx_frame([("2026-09-01", 96.0)])  # FX starts long after gold
    dates = pd.Series(pd.to_datetime(["2016-06-01", "2026-09-15"]))
    out = rate_frame_for_dates(dates, rates=rates)
    # No rate before 2026-09-01: falls back to the earliest available rate,
    # never 0/NaN.
    assert out["usd_inr_rate"].tolist() == [96.0, 96.0]
    assert out["usd_inr_rate_date"].iloc[0] == pd.Timestamp("2026-09-01")


def test_latest_rate_reference_date_staleness():
    rates = _fx_frame([("2026-09-08", 88.0)])
    out = latest_rate(rates=rates, reference_date=dt.date(2026, 9, 15))
    assert out["rate"] == 88.0
    assert out["rate_may_be_stale"] is True  # 7 days before the newest gold date
    out2 = latest_rate(rates=rates, reference_date=dt.date(2026, 9, 9))
    assert out2["rate_may_be_stale"] is False


def test_rate_frame_per_date_uses_each_dates_own_rate():
    from ingestion.usdinr import rate_frame_for_dates

    rates = _fx_frame([("2026-01-05", 66.0), ("2026-09-14", 96.0)])
    dates = pd.Series(
        pd.to_datetime(["2026-01-06", "2026-01-07", "2026-06-01", "2026-09-15"])
    )
    out = rate_frame_for_dates(dates, rates=rates)
    # Early dates use the early (66) rate; the later date falls back to 96.
    assert out["usd_inr_rate"].tolist() == [66.0, 66.0, 66.0, 96.0]
    assert out["usd_inr_rate_date"].iloc[1] == pd.Timestamp("2026-01-05")
    assert out["usd_inr_rate_date"].iloc[3] == pd.Timestamp("2026-09-14")
