"""Currency/karat/unit conversion for gold prices.

COMEX prices are USD per troy ounce. Indian gold is quoted in INR per gram or
per 10 grams at a purity (24K/22K/18K). This module converts between them.

HONESTY NOTE (do not remove): the result is a *theoretical bullion-equivalent*
price. Real Indian retail/jeweler prices also include import duty, GST, and
dealer making charges and are typically noticeably higher than this figure.
"""

from __future__ import annotations

TROY_OZ_TO_GRAM = 31.1034768

PURITY_FACTORS: dict[str, float] = {
    "24k": 0.999,
    "22k": 0.916,
    "18k": 0.750,
}

UNIT_FACTORS: dict[str, float] = {
    "gram": 1.0,
    "10gram": 10.0,
}

SUPPORTED_KARATS = tuple(PURITY_FACTORS)
SUPPORTED_UNITS = tuple(UNIT_FACTORS)


def usd_per_troy_oz_to_inr_per_gram(
    usd_price: float, usd_inr_rate: float, karat: str
) -> float:
    """Converts COMEX USD/troy-oz to a theoretical INR/gram price at the given purity.

    Formula: INR/gram = USD/oz * USDINR / (troy oz in grams) * purity.

    Note this is a bullion-equivalent value: actual Indian retail prices add
    import duty, GST, and making charges on top.
    """
    return usd_per_oz_to_inr(usd_price, usd_inr_rate, karat, unit="gram")


def usd_per_oz_to_inr(
    usd_price: float, usd_inr_rate: float, karat: str, unit: str = "10gram"
) -> float:
    """USD/troy-oz -> INR per `unit` (gram or 10gram) at the given purity."""
    try:
        purity = PURITY_FACTORS[karat.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported karat: {karat!r}") from exc
    try:
        unit_factor = UNIT_FACTORS[unit.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported unit: {unit!r}") from exc
    if usd_inr_rate <= 0:
        raise ValueError("usd_inr_rate must be positive")
    return (
        float(usd_price)
        * float(usd_inr_rate)
        / TROY_OZ_TO_GRAM
        * purity
        * unit_factor
    )


def convert_series(
    values: "list[float] | object",  # numpy array or list
    usd_inr_rate: float,
    karat: str,
    unit: str = "10gram",
):
    """Vectorised conversion of a whole price series (numpy array in/out)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return arr * usd_inr_rate / TROY_OZ_TO_GRAM * PURITY_FACTORS[karat.lower()] * UNIT_FACTORS[unit.lower()]
