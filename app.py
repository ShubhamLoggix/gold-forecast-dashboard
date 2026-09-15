"""Stage A dashboard: Streamlit gold price forecasting app.

Run:  streamlit run app.py
Wired directly to the ingestion + forecasting modules (no API layer yet).

DISCLAIMER (must remain visible, per project constraints): forecasts are
experimental; gold series behave close to random walks; never use as the sole
basis for financial decisions.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from conversion import PURITY_FACTORS, TROY_OZ_TO_GRAM, UNIT_FACTORS, convert_series
from forecasting.baselines import naive_last_value, simple_moving_average
from forecasting.timesfm_service import HORIZON_PRESETS, GoldForecastService
from ingestion.fetch_gold_prices import update_canonical
from ingestion.usdinr import latest_rate
from logging_setup import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger("gold_forecast.dashboard")

RANGE_PRESETS = {"1M": 21, "6M": 126, "1Y": 252, "5Y": 1260, "Max": None}

DISCLAIMER = (
    "**DISCLAIMER — NOT INVESTMENT ADVICE.** Price forecasts shown here are experimental. "
    "Gold and other financial time series behave close to random walks; published benchmarks "
    "put zero-shot model directional accuracy near chance (~50%). Do **not** use this "
    "dashboard as the sole basis for any financial decision."
)

st.set_page_config(
    page_title="Gold Forecast (TimesFM)",
    page_icon=":gold:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading TimesFM (first run downloads ~1 GB)...")
def get_service() -> GoldForecastService:
    service = GoldForecastService(
        model_version=settings.model_version, context_length=settings.context_length
    )
    service.load_model()
    return service


@st.cache_data(show_spinner=False)
def get_history() -> pd.DataFrame:
    try:
        return pd.read_parquet(settings.processed_dir / "gold_prices_daily.parquet")
    except FileNotFoundError:
        from ingestion.fetch_gold_prices import update_canonical

        return update_canonical(
            dt.date.fromisoformat(settings.history_start), dt.date.today()
        )


@st.cache_data(show_spinner=False)
def cached_forecast(history: pd.DataFrame, horizon: int) -> dict:
    service = get_service()
    result = service.forecast(history, horizon, quantiles=True)
    return {
        "dates": result.dates,
        "point": result.point.tolist(),
        "q10": result.q10.tolist(),
        "q50": result.q50.tolist(),
        "q90": result.q90.tolist(),
        "latency_ms": result.latency_ms,
        "model_version": result.model_version,
    }


def load_latest_backtest() -> dict | None:
    out_dir = settings.backtest_dir
    files = sorted(out_dir.glob("backtest_*.json"))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Data refresh / backtest actions
# --------------------------------------------------------------------------- #
def refresh_data() -> None:
    import datetime as dt_

    from ingestion.fetch_gold_prices import update_canonical
    from ingestion.usdinr import fetch_usdinr_rate

    update_canonical(dt_.date.fromisoformat(settings.history_start), dt_.date.today())
    # Same schedule/side-effect as the API's daily job: refresh FX alongside gold.
    try:
        fetch_usdinr_rate(dt_.date.today() - dt_.timedelta(days=14), dt_.date.today())
    except Exception as exc:  # noqa: BLE001 - FX failure must not kill gold refresh
        logger.warning("USDINR refresh failed: %s", exc)
    get_history.clear()
    cached_forecast.clear()


def run_backtest_action() -> None:
    history = get_history()
    service = get_service()
    results = service.backtest(history, horizon_days=30, step_days=7, max_folds=26)
    service.persist_backtest(results)


# --------------------------------------------------------------------------- #
# Chart builder
# --------------------------------------------------------------------------- #
def build_figure(
    history_view: pd.DataFrame,
    forecast: dict,
    naive: pd.Series,
    sma: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    dates_hist = history_view["date"]
    closes = history_view["close"].astype(float)

    # Confidence band p10-p90 (drawn first so lines render on top).
    fdates = pd.to_datetime(forecast["dates"])
    band_x = fdates.tolist() + fdates.tolist()[::-1]
    band_y = forecast["q90"] + forecast["q10"][::-1]

    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=band_y,
            fill="toself",
            fillcolor="rgba(217,164,6,0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="p10-p90 band",
            showlegend=True,
            legendgroup="band",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates_hist,
            y=closes,
            mode="lines",
            name="Historical close",
            line=dict(color="#1266a2", width=2),
        )
    )
    # Connect history end -> forecast start so it reads as a continuation.
    bridge_x = [dates_hist.iloc[-1], fdates[0]]
    bridge_y = [closes.iloc[-1], forecast["q50"][0]]

    fig.add_trace(
        go.Scatter(
            x=bridge_x + fdates.tolist(),
            y=bridge_y + forecast["q50"],
            mode="lines",
            name="TimesFM forecast (p50)",
            line=dict(color="#d9a406", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bridge_x + fdates.tolist(),
            y=bridge_y + naive.tolist(),
            mode="lines",
            name="Naive last-value",
            line=dict(color="#888", width=1.2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bridge_x + fdates.tolist(),
            y=bridge_y + sma.tolist(),
            mode="lines",
            name="SMA(20) baseline",
            line=dict(color="#bb5588", width=1.2, dash="dot"),
        )
    )

    last_date = dates_hist.iloc[-1]
    fig.add_vline(
        x=last_date, line=dict(color="#d9a406", width=1, dash="dash"),
        annotation_text="today", annotation_position="top",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="USD/oz (COMEX GC=F)",
        xaxis_title=None,
        hovermode="x unified",
    )
    return fig


def aggregate_history(history: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "Day":
        return history
    rule = "W-FRI" if granularity == "Week" else "ME"
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"])
    agg = (
        df.set_index("date")
        .resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source=("source", "last"),
        )
        .reset_index()
    )
    return agg


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("Gold Price Forecast — TimesFM")

# Non-dismissable disclaimer near the forecast chart (also in the footer).
st.error(DISCLAIMER, icon="⚠️")

history_full = get_history()

with st.sidebar:
    st.header("View")
    granularity = st.radio(
        "Historical granularity", ["Day", "Week", "Month"], horizontal=True
    )
    range_label = st.radio(
        "Historical range", list(RANGE_PRESETS.keys()), horizontal=False
    )
    horizon_label = st.radio("Forecast horizon", list(HORIZON_PRESETS.keys()))
    st.divider()
    currency = st.radio("Currency", ["USD", "INR"], horizontal=True)
    karat = "24k"
    unit = "10gram"
    if currency == "INR":
        karat = st.radio(
            "Karat (INR view)", ["24k", "22k", "18k"], index=1, horizontal=True
        )
        unit = st.radio(
            "Quoting unit (INR view)", ["10gram", "gram"], horizontal=True
        )
    st.divider()
    if st.button("Refresh data from yfinance", width="stretch"):
        with st.spinner("Fetching latest prices..."):
            refresh_data()
        st.rerun()
    if st.button("Re-run backtest (~15s)", width="stretch"):
        with st.spinner("Running walk-forward backtest..."):
            run_backtest_action()
        st.rerun()
    st.divider()
    st.caption(
        "Model: TimesFM 2.5 (google/timesfm-2.5-200m-pytorch), Apache-2.0. "
        "See **Model info** below for details."
    )

n_show = RANGE_PRESETS[range_label]
history_view = history_full.tail(n_show) if n_show else history_full
horizon_days = HORIZON_PRESETS[horizon_label]

is_inr = currency == "INR"

try:
    forecast = cached_forecast(history_full, horizon_days)
except Exception as exc:  # noqa: BLE001
    st.error(f"Forecast failed: {exc}", icon="🚫")
    st.stop()

naive = naive_last_value(history_full, horizon_days)
sma = simple_moving_average(history_full, horizon_days)

unit_label = (
    f"INR per {'10g' if unit == '10gram' else 'g'} ({karat.upper()}, bullion-equiv.)"
    if is_inr
    else "USD/oz (COMEX GC=F)"
)
value_prefix = "₹" if is_inr else "$"
rate_info = None

if is_inr:
    try:
        rate_info = latest_rate(reference_date=history_full["date"].iloc[-1].date())
    except FileNotFoundError:
        st.error(
            "INR view needs the USD/INR rate series — click **Refresh data** to fetch it.",
            icon="🚫",
        )
        st.stop()
    rate = rate_info["rate"]
    # Per-date conversion for history (each date uses its own USD/INR rate,
    # backward as-of, never interpolated); forecast segment uses the latest rate.
    from ingestion.usdinr import rate_frame_for_dates

    history_disp = history_view.copy()
    fx = rate_frame_for_dates(history_disp["date"])
    factor = (
        fx["usd_inr_rate"].astype(float) / TROY_OZ_TO_GRAM
        * PURITY_FACTORS[karat] * UNIT_FACTORS[unit]
    ).to_numpy()
    for col in ("open", "high", "low", "close"):
        history_disp[col] = history_disp[col].astype(float) * factor
    forecast_conv = {
        **forecast,
        "point": convert_series(forecast["point"], rate, karat, unit).tolist(),
        "q10": convert_series(forecast["q10"], rate, karat, unit).tolist(),
        "q50": convert_series(forecast["q50"], rate, karat, unit).tolist(),
        "q90": convert_series(forecast["q90"], rate, karat, unit).tolist(),
    }
    naive_conv = convert_series(naive, rate, karat, unit)
    sma_conv = convert_series(sma, rate, karat, unit)
    last_close_conv = float(
        history_full["close"].iloc[-1] * rate / TROY_OZ_TO_GRAM
        * PURITY_FACTORS[karat] * UNIT_FACTORS[unit]
    )
else:
    history_disp = history_view
    forecast_conv = forecast
    naive_conv = naive
    sma_conv = sma
    last_close_conv = float(history_full["close"].iloc[-1])

# ----------------------------- metrics panel ------------------------------ #
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Last close" if not is_inr else f"Last close ({karat.upper()}, per {'10g' if unit == '10gram' else 'g'})",
    f"{value_prefix}{last_close_conv:,.2f}",
)
median_end = forecast_conv["q50"][-1]
implied = (median_end / history_full["close"].iloc[-1] - 1) * 100
col2.metric(
    f"Forecast {horizon_label} (median)",
    f"{value_prefix}{median_end:,.2f}",
    f"{implied:+.1f}%",
    delta_color="normal",
)
backtest = load_latest_backtest()
bt = backtest["results"]["timesfm-2.5"]["summary"] if backtest else None
col3.metric(
    "Backtest MAPE (30d)",
    f"{bt['mape_pct']:.2f}%" if bt else "n/a",
    help="Walk-forward MAPE of TimesFM, 30-trading-day horizon, averaged over folds.",
)
if bt:
    da = bt["directional_accuracy_pct"]
    col4.metric("Backtest dir. accuracy", f"{da:.1f}%", delta=f"{da - 50:+.1f}pp vs chance")
else:
    col4.metric("Backtest dir. accuracy", "n/a")

# ----------------------------- main chart --------------------------------- #
agg = aggregate_history(history_disp, granularity)
fig = build_figure(agg, forecast_conv, naive_conv, sma_conv)
fig.update_yaxes(title=unit_label)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
if is_inr:
    st.caption(
        "Converted from COMEX USD futures at the live USD/INR rate — this is a theoretical "
        "bullion-equivalent price, not an Indian retail/jeweler quote, which also includes "
        "import duty, GST, and making charges. Historical points use each date's USD/INR "
        "rate; the forecast uses the latest rate."
    )
    if rate_info is not None:
        stale_note = (
            " (rate may be stale — last available FX date used)"
            if rate_info["rate_may_be_stale"]
            else ""
        )
        st.caption(
            f"Converted at ₹{rate_info['rate']:.4f}/USD as of {rate_info['rate_date']}{stale_note}."
        )
st.caption(
    "Shaded area = p10-p90 quantile band. Dashed/dotted lines are naive baselines "
    "(last-value carry-forward, 20-day moving average). If the gold line is not "
    "clearly better than them, the model is not adding value — judge accordingly."
)

# ----------------------------- model info --------------------------------- #
with st.expander("Model info"):
    drift = (backtest or {}).get("model_drift")
    if drift and drift.get("underperforming"):
        st.warning(
            f"**MODEL UNDERPERFORMING BASELINE** — TimesFM directional accuracy is "
            f"{abs(drift['delta_pp']):.1f} pp *below* the naive carry-forward baseline "
            f"({drift['timesfm_dir_acc_pct']:.1f}% vs {drift['naive_dir_acc_pct']:.1f}%). "
            f"Treat all forecasts with extra skepticism.",
            icon="🚨",
        )
    last_refresh = dt.datetime.fromtimestamp(
        (settings.processed_dir / "gold_prices_daily.parquet").stat().st_mtime
    )
    license_ = (
        "Apache-2.0 (commercial-safe)"
        if settings.model_version == "2.5"
        else "timesfm-non-commercial-license-v1.0 (NON-COMMERCIAL)"
    )
    drift_note = (
        (
            f"- **Drift watchdog:** TimesFM vs naive baseline delta = "
            f"{drift['delta_pp']:+.1f} pp (warning threshold: -5 pp). Flag active: "
            f"{'YES — see warning above' if drift.get('underperforming') else 'no'}\n"
        )
        if drift
        else "- **Drift watchdog:** no backtest report yet — run one from the sidebar.\n"
    )
    inr_note = (
        (
            f"- **INR conversion:** ₹{rate_info['rate']:.4f}/USD as of {rate_info['rate_date']}"
            f"{(' — rate may be stale' if rate_info['rate_may_be_stale'] else '')}. "
            f"Theoretical bullion-equivalent at {karat.upper()} purity — *not* an Indian "
            f"retail/jeweler quote (import duty, GST, making charges excluded).\n"
        )
        if is_inr and rate_info
        else ""
    )
    st.markdown(
        f"""
- **Model:** TimesFM {settings.model_version} (`google/timesfm-2.5-200m-pytorch` /
  `google/timesfm-3.0-pytorch` depending on `MODEL_VERSION`)
- **License:** {license_}
- **Training data:** large public + synthetic time-series corpora (see the
  [TimesFM model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) for details;
  the exact training cutoff is not documented precisely — the model was *not* trained on
  up-to-the-minute gold data)
- **Last data refresh:** {last_refresh:%Y-%m-%d %H:%M} ({len(history_full)} trading days, source: COMEX GC=F via yfinance)
{drift_note}{inr_note}- **Known limitations:** zero-shot foundation models on financial series perform close to
  chance level (~50% directional accuracy); quantile bands reflect model-internal
  uncertainty, not calibrated risk. Improvement requires financial-domain fine-tuning,
  which this project does not include by default.
        """
    )

st.markdown("---")
st.markdown(DISCLAIMER)
st.caption(
    "Built with TimesFM (Google Research). Data: COMEX Gold Futures (GC=F) via Yahoo Finance. "
    "This is a forecasting demo, not investment advice."
)
