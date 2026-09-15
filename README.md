> ⚠️ **Disclaimer:** This project is a *forecasting demo*, not investment advice. Price forecasts are
> experimental; gold/financial time series are close to random walks. Do **not** use this as the sole
> basis for any financial decision. See [LIMITATIONS.md](LIMITATIONS.md).

# Gold Price Forecasting Dashboard (TimesFM)

Production-ready dashboard that ingests real gold price data (COMEX `GC=F` via `yfinance`), forecasts
future prices with Google Research's **TimesFM** foundation model, compares against naive baselines,
and validates everything with a walk-forward backtest — all displayed in an interactive dashboard.

## Architecture

```
[Data Source(s)]  ->  [Ingestion Service]  ->  [Data Store (Parquet/SQLite)]
        yfinance GC=F        validation +          canonical clean series
        LBMA (backup)        caching               data/processed/
                                   |
                                   v
                    [Forecast Service (TimesFM)]
                          point + quantile bands
                          naive baselines + backtest
                                   |
                     +-------------+-------------+
                     v                           v
              [REST API (FastAPI)]      [Stage A Streamlit app]
                     v
              [React Frontend (Recharts)]
                     |
              [Scheduler (APScheduler)]  -> daily data refresh, weekly backtest
```

## Two stages

| Stage | What it is | How to run |
|---|---|---|
| **A (MVP)** | Single Streamlit app, local Parquet storage, on-demand inference | `streamlit run app.py` |
| **B (hardened)** | FastAPI backend + React frontend + scheduler + API-key refresh endpoint + Docker | `docker compose up` (or run `uvicorn` + Vite dev server) |

## Quickstart (Stage A)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
streamlit run app.py
```

First run downloads 5 years of `GC=F` history and the `google/timesfm-2.5-200m` weights (~1 GB),
then runs on CPU or GPU.

## Quickstart (Stage B)

```bash
docker compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:5173
```

## Model & license

- **Default model:** `google/timesfm-2.5-200m` — Apache-2.0, safe for commercial use.
- `MODEL_VERSION=3.0` swaps to TimesFM 3.0, whose weights are **non-commercial**
  (`timesfm-non-commercial-license-v1.0`). The app refuses (or warns, with `NON_COMMERCIAL=true`)
  unless you explicitly accept the non-commercial terms.

## Honest accuracy note

Published benchmarks show zero-shot time-series foundation models — including TimesFM — perform
near chance level (~50% directional accuracy) on raw financial return series. The dashboard therefore
always shows TimesFM **next to** naive baselines (last-value carry-forward, 20-day moving average)
and a walk-forward backtest (MAPE / RMSE / MAE / directional accuracy) so you can judge whether it is
actually adding value. Meaningful improvement typically requires financial-domain fine-tuning, which
this project does not include by default.
