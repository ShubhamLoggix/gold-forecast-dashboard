# Gold Price Forecasting Dashboard (TimesFM)

Production-ready dashboard that ingests real gold price data (COMEX `GC=F` via `yfinance`), forecasts
future prices with Google Research's **TimesFM** foundation model, compares against naive baselines,
and validates everything with a walk-forward backtest — all displayed in an interactive dashboard.

> ⚠️ **Disclaimer:** This project is a *forecasting demo*, not investment advice. Price forecasts are
> experimental; gold/financial time series are close to random walks. Do **not** use this as the sole
> basis for any financial decision. See [LIMITATIONS.md](LIMITATIONS.md).

> Project path on this machine: `D:\projects\gold-forecast-dashboard` (migrated from C: for disk
> space; model + pip caches also live on D:).

## INR / karat view (24K, 22K, 18K)

Both the API and the dashboards can display the same historical, forecast, and baseline series
converted to **INR per gram or per 10 grams** (the conventional Indian quote) at 24K/22K/18K
purity, using the daily USD/INR rate (yfinance `INR=X`, cached alongside the gold data).

**Important honesty note:** the converted figure is a *theoretical bullion-equivalent price* —
COMEX is the international wholesale/futures price. Real Indian retail/jeweler prices are
noticeably higher because they also include **import duty, GST, and making charges**. The
dashboard labels this explicitly (persistent caption under the chart in INR mode and in the
model-info panel); don't compare the converted value against a jeweler's quote and expect them
to match. The conversion metadata (`₹X/USD as of DATE`, staleness flag) is always shown for
transparency.

## India retail rates (city-wise, real quotes)

INR mode has a **Source** toggle: `Bullion (COMEX)` (the converted view above) vs
`Retail (Groww)` — actual published city-wise retail rates (24K/22K/18K, per gram or per 10g)
scraped from [Groww's gold-rates pages](https://groww.in/gold-rates). Covers ~350 Indian
cities with a city selector, prev-day % change, and the last ~10 days of history per city
(more accumulates daily in `data/processed/india_gold_rates.parquet`).

- API: `GET /api/v1/india/cities`, `GET /api/v1/india/rates?city=pune&unit=10gram`, and
  `GET /api/v1/india/forecast?city=pune&horizon=1m&karat=24k&unit=10gram` — the forecast is
  an **estimate**: TimesFM's COMEX bullion forecast scaled by the city's current retail
  premium (today's retail quote ÷ today's bullion-equivalent), since only the COMEX series
  has enough history to forecast.
- Refreshed by the daily scheduler alongside the FX top-up; failures never degrade /health
- Retail rates include duty/GST/premium and vary by city — they intentionally differ from
  bullion prices; each response carries an attribution + disclaimer block.

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
cd D:\projects\gold-forecast-dashboard
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

First run downloads ~10 years of `GC=F` history and the `google/timesfm-2.5-200m-pytorch`
weights (~1 GB), then runs on CPU or GPU. Inference takes ~0.3 s per forecast on CPU.

Cache locations: `HF_HOME`, `TRANSFORMERS_CACHE`, and `PIP_CACHE_DIR` should be set as
persistent User environment variables pointing at a drive with headroom
(e.g. `D:\hf_cache`, `D:\pip_cache` — see `.env.example`). On low-disk system drives this
project's `config.py` also auto-falls-back to `D:\hf_cache` when unset, but the env vars are
the supported way.

## Quickstart (Stage B)

Backend only:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000   # docs at http://localhost:8000/docs
```

Frontend (dev):

```bash
cd frontend
npm install
npm run dev        # Vite dev server; /api proxied to :8000
```

Full stack with Docker (verified end-to-end — see [DEPLOYMENT_VERIFICATION.md](DEPLOYMENT_VERIFICATION.md)):

```bash
cd docker
docker compose up --build
# API:      http://localhost:8000/docs
# Frontend: http://localhost:5174  (override with FRONTEND_PORT; 5173 is often taken)
```

Notes:
- The backend image installs **CPU-only torch** (`download.pytorch.org/whl/cpu`) — the default
  Linux wheel pulls the full CUDA stack (~3.5 GB) that CPU inference never uses.
- On first start with an empty volume the backend seeds data from yfinance and downloads the
  TimesFM weights in the background; the HTTP server stays responsive and `/api/v1/health`
  reports progress until `model_loaded: true`.
- The nginx frontend container includes Docker's embedded-DNS resolver so proxied `/api`
  calls work as soon as the backend is healthy (fixed a 502 regression; details in
  DEPLOYMENT_VERIFICATION.md).

The compose file mounts a `gold-data` volume for `data/` (canonical series + backtest reports
+ HF model cache).

## Operations

- `GET /api/v1/health` — model loaded, data freshness (`fresh|cached|stale|missing`), scheduler flag,
  and the **model-drift watchdog** (`model_underperforming_baseline`) comparing the latest backtest's
  TimesFM directional accuracy against the naive baseline (warning threshold: -5 pp).
- `POST /api/v1/refresh` — protected by `X-API-Key` (constant-time compared), rate-limited to 1/min,
  force refresh via `?force=true`. Public read endpoints are rate-limited (60/min per client) and
  capped at 1 MiB body size.
- Scheduler (if `ENABLE_SCHEDULER=1`): daily data refresh at 22:10 UTC Mon–Fri (after COMEX close),
  weekly backtest Monday 06:00 UTC. Failures are logged and flip `data_may_be_stale` in `/health`
  instead of crashing the API; the API keeps serving last-known-good cached data.
- Backtest report: `scripts/run_backtest.py --horizon 30 --step 7 [--max-folds 52]` → writes
  `data/backtests/backtest_<date>.json` (including the drift block), served by
  `GET /api/v1/backtest/latest`.
- CI: `.github/workflows/ci.yml` runs unit tests, a fixture-window backtest smoke
  (`scripts/backtest_smoke.py`, no model download), and builds both Docker images with
  GitHub Actions layer caching on every PR.

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
