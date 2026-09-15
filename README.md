> ⚠️ **Disclaimer:** This project is a *forecasting demo*, not investment advice. Price forecasts are
> experimental; gold/financial time series are close to random walks. Do **not** use this as the sole
> basis for any financial decision. See [LIMITATIONS.md](LIMITATIONS.md).

# Gold Price Forecasting Dashboard (TimesFM)

Production-ready dashboard that ingests real gold price data (COMEX `GC=F` via `yfinance`), forecasts
future prices with Google Research's **TimesFM** foundation model, compares against naive baselines,
and validates everything with a walk-forward backtest — all displayed in an interactive dashboard.

## Architecture

```
[Data Source(s)]  ->  [Ingestion Service]  ->  [Data Store (Parquet)]
   yfinance GC=F          validation +         data/raw/   (per-source cache)
   LBMA (backup)          caching              data/processed/  (canonical daily series)
   metals-api (opt.)
                                |
                                v
                 [Forecast Service (TimesFM)]
                       point + p10/p50/p90 bands
                       naive baselines + walk-forward backtest
                                |
                  +-------------+-------------+
                  v                           v
           [REST API (FastAPI)]      [Stage A Streamlit app]
           /api/v1/history | forecast      app.py
           /backtest | /health | /refresh
                  v
           [React Frontend (Recharts)]  [Scheduler (APScheduler)]
           frontend/ (Vite)             daily refresh 22:10 UTC, weekly backtest
```

## Two stages

| Stage | What it is | How to run |
|---|---|---|
| **A (MVP)** | Single Streamlit app, local Parquet storage, on-demand inference | `streamlit run app.py` |
| **B (hardened)** | FastAPI backend + React frontend + scheduler + API-key refresh endpoint + Docker | `docker compose up` from `docker/`, or backend + Vite dev servers |

## Quickstart (Stage A)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

First run downloads ~5 years of `GC=F` history and the `google/timesfm-2.5-200m-pytorch`
weights (~1 GB), then runs on CPU or GPU. Inference takes ~0.3 s per forecast on CPU.

## Quickstart (Stage B)

Backend only:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000   # docs at http://localhost:8000/docs
```

Frontend (dev):

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, /api proxied to :8000
```

Full stack with Docker:

```bash
cd docker
docker compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:5173
```

The compose file mounts a `gold-data` volume for `data/` (canonical series + backtest reports
+ HF model cache).

## Operations

- `GET /api/v1/health` — model loaded, data freshness (`fresh|cached|stale|missing`), scheduler flag.
- `POST /api/v1/refresh` — protected by `X-API-Key` (from `.env`), rate-limited to 1/min, force
  refresh via `?force=true`.
- Scheduler (if `ENABLE_SCHEDULER=1`): daily data refresh at 22:10 UTC Mon–Fri (after COMEX close),
  weekly backtest Monday 06:00 UTC. Failures are logged and flip `data_may_be_stale` in `/health`
  instead of crashing the API; the API keeps serving last-known-good cached data.
- Backtest report: `scripts/run_backtest.py --horizon 30 --step 7 [--max-folds 52]` → writes
  `data/backtests/backtest_<date>.json`, served by `GET /api/v1/backtest/latest`.

## Model & license

- **Default model:** `google/timesfm-2.5-200m-pytorch` — Apache-2.0, commercial-safe.
- `MODEL_VERSION=3.0` swaps to TimesFM 3.0, whose weights are **non-commercial**
  (`timesfm-non-commercial-license-v1.0`). Loading 3.0 raises a clear license error unless
  `NON_COMMERCIAL=true` is set; when set, a prominent warning is logged at startup and on load.
- Inference settings: context length `CONTEXT_LENGTH` (default 512), device (`DEVICE=auto|cpu|cuda`),
  `TIMESFM_TORCH_COMPILE=1` to enable `torch.compile` (off by default; Windows-safe default).

## Tests

```bash
pytest tests -q
```

Unit tests cover ingestion validation/caching (mocked sources, no network), baseline math,
forecast service output contracts, API endpoint contracts, and the license gate. A stubbed model
deterministically emulates the real TimesFM output contract (verified against real weights by
`scripts/verify_model.py`); CI never downloads weights or hits the network.

## Honest accuracy note

Published benchmarks show zero-shot time-series foundation models — including TimesFM — perform
near chance level (~50% directional accuracy) on raw financial return series. In our own
walk-forward backtest on real COMEX data, TimesFM 2.5 achieved ~48.7% directional accuracy and
~3.4% MAPE at a 30-trading-day horizon — no better than the naive carry-forward baseline (~3.1%
MAPE). The dashboard therefore always shows TimesFM **next to** naive baselines (last-value
carry-forward, 20-day moving average) plus the measured backtest metrics, so you can judge whether
it is actually adding value. Meaningful improvement typically requires financial-domain
fine-tuning, which this project does not include by default. See [LIMITATIONS.md](LIMITATIONS.md).
