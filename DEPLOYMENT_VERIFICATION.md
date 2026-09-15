# Deployment Verification (Stage C)

Date: 2026-09-15 · Project root: `D:\projects\gold-forecast-dashboard`
Docker Desktop 4.82.0 · Engine 29.6.1 · Backend image `gold-backend:latest` (2.09 GB, CPU-only torch)

## 1. What was verified, through the containers

Full stack: `docker compose up -d --build` from `docker/`. Both containers
(`docker-backend-1`, `docker-frontend-1`) up; backend marked **Healthy** by the
compose healthcheck, frontend started only after `condition: service_healthy`.

| Check | From where | Result |
|---|---|---|
| `GET /api/v1/health` | Host → `localhost:8000` (backend directly) | **200**, `status: ok`, `model_loaded: true`, `data_freshness: fresh`, data through 2026-09-15 |
| `GET /api/v1/health` | Host → `localhost:5174/api/v1/health` (via nginx proxy) | **200**, same fields (proxy works) |
| `GET /api/v1/history?granularity=month` | Host → nginx proxy | **200** — 10+ years of monthly OHLC from yfinance, seeded inside the container |
| `GET /api/v1/forecast?horizon=1w` | Host → nginx proxy | **200** — real in-container TimesFM inference: point ≈ 4322.78, latency 436 ms, 2 baselines attached |
| Backend reachability from frontend container | `docker exec docker-frontend-1 wget http://backend:8000/api/v1/health` | **200** — compose DNS resolves, model loaded |
| Dashboard SPA | Host → `localhost:5174/` | **200** — same-origin nginx-proxied API calls |

First-start behavior verified: with an **empty volume**, the backend seeded
canonical data from yfinance and downloaded TimesFM weights into the mounted
`gold-data` volume during background bootstrap; the HTTP server stayed
responsive the whole time (healthcheck stayed green; `/health` surfaced
`model_loaded` flipping false → true).

## 2. The `/api → 502` root cause and fix

**Symptom:** every `GET localhost:5174/api/v1/*` through nginx returned 502 even
though the backend container was up, healthy, and reachable from the frontend
container (`wget http://backend:8000` succeeded).

**Root cause:** the nginx config uses a *variable* `proxy_pass`
(`set $backend_upstream http://backend:8000; proxy_pass $backend_upstream;`) so
that nginx can start even before the backend exists. Variable proxy passes are
resolved at **request time** — and nginx only does that through a configured
`resolver`. With no `resolver` directive, nginx logged
`no resolver defined to resolve "backend"` and returned 502 regardless of the
backend's state. (The original 502 noted in the earlier migration report had two
stacked causes: the frontend container tested standalone with no backend on its
network, *and* this missing resolver.)

**Fix:** added `resolver 127.0.0.11 valid=30s ipv6=off;` (Docker's embedded DNS)
to `docker/nginx.conf.template`. Verified: proxied `/health`, `/history`,
`/forecast` all 200.

**Regression prevention:** `docker-compose.yml` now defines an explicit backend
`healthcheck` (HTTP 200 probe, `start_period: 900s` to cover one-time weight
download) and `depends_on: condition: service_healthy` on the frontend, so the
frontend never routes to a backend that isn't up.

## 3. Host port note

Host port **5174** (not 5173) is used for the frontend container because 5173
was already bound by an unrelated local `node server.js` process. Override with
`FRONTEND_PORT=<port> docker compose up`.

## 4. Environment

Verified on Windows 11 (win32), PowerShell 5.1, Python 3.14 venv at
`D:\projects\gold-forecast-dashboard\.venv`, Docker Desktop WSL2 backend.
`HF_HOME`, `TRANSFORMERS_CACHE`, `PIP_CACHE_DIR` are persistent User
environment variables pointing at D: paths.
