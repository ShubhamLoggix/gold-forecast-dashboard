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
from forecasting.baselines import naive_last_value, simple_moving_average
from forecasting.timesfm_service import HORIZON_PRESETS, GoldForecastService
from ingestion.fetch_gold_prices import update_canonical
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
    from ingestion.fetch_gold_prices import update_canonical

    update_canonical(dt.date.fromisoformat(settings.history_start), dt.date.today())
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

try:
    forecast = cached_forecast(history_full, horizon_days)
except Exception as exc:  # noqa: BLE001
    st.error(f"Forecast failed: {exc}", icon="🚫")
    st.stop()

naive = naive_last_value(history_full, horizon_days)
sma = simple_moving_average(history_full, horizon_days)

# ----------------------------- metrics panel ------------------------------ #
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last close", f"${history_full['close'].iloc[-1]:,.2f}")
median_end = forecast["q50"][-1]
implied = (median_end / history_full["close"].iloc[-1] - 1) * 100
col2.metric(
    f"Forecast {horizon_label} (median)",
    f"${median_end:,.2f}",
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
agg = aggregate_history(history_view, granularity)
fig = build_figure(agg, forecast, naive, sma)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
st.caption(
    "Shaded area = p10-p90 quantile band. Dashed/dotted lines are naive baselines "
    "(last-value carry-forward, 20-day moving average). If the gold line is not "
    "clearly better than them, the model is not adding value — judge accordingly."
)

# ----------------------------- model info --------------------------------- #
with st.expander("Model info"):
    last_refresh = dt.datetime.fromtimestamp(
        (settings.processed_dir / "gold_prices_daily.parquet").stat().st_mtime
    )
    license_ = (
        "Apache-2.0 (commercial-safe)"
        if settings.model_version == "2.5"
        else "timesfm-non-commercial-license-v1.0 (NON-COMMERCIAL)"
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
- **Known limitations:** zero-shot foundation models on financial series perform close to
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
