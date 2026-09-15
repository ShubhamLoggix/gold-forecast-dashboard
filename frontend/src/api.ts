// API client for the gold forecast backend.
const BASE = import.meta.env.VITE_API_URL ?? ""; // "" => same-origin (vite proxy)

export type Granularity = "day" | "week" | "month";
export type Horizon = "1w" | "1m" | "3m" | "6m" | "1y";
export type Currency = "usd" | "inr";
export type Karat = "24k" | "22k" | "18k";
export type Unit = "gram" | "10gram";

export interface RateInfo {
  usd_inr_rate: number;
  usd_inr_rate_date: string;
  rate_may_be_stale: boolean;
  disclaimer: string;
}

export interface OhlcPoint {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number;
  volume: number | null;
}

export interface HistoryResponse {
  start: string;
  end: string;
  granularity: Granularity;
  source: string;
  points: OhlcPoint[];
  currency: Currency;
  karat: Karat | null;
  unit: Unit | null;
  rate: RateInfo | null;
}

export interface BaselineSeries {
  name: string;
  values: number[];
}

export interface ForecastResponse {
  horizon: Horizon;
  horizon_days: number;
  model_version: string;
  generated_at: string;
  latency_ms: number;
  history_last_date: string;
  history_last_close: number;
  dates: string[];
  point: number[];
  q10: number[];
  q50: number[];
  q90: number[];
  quantiles: boolean;
  baselines: BaselineSeries[];
  currency: Currency;
  karat: Karat | null;
  unit: Unit | null;
  rate: RateInfo | null;
}

export interface BacktestResponse {
  generated_at: string;
  model_version: string;
  context_length: number;
  model_drift?: {
    delta_pp: number;
    timesfm_dir_acc_pct: number;
    naive_dir_acc_pct: number;
    underperforming: boolean;
  };
  results: Record<
    string,
    {
      model: string;
      summary: Record<string, number>;
    }
  >;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  data_last_date: string | null;
  data_freshness: string;
  data_may_be_stale: boolean;
}

export class ApiError extends Error {
  code: string;
  requestId?: string;
  constructor(code: string, message: string, requestId?: string) {
    super(message);
    this.code = code;
    this.requestId = requestId;
  }
}

async function request<T>(path: string): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError("network_error", "API unreachable — is the backend running on :8000?");
  }
  if (!resp.ok) {
    let code = "http_error";
    let message = `Request failed with status ${resp.status}`;
    try {
      const body = await resp.json();
      if (body?.error) {
        code = body.error.code ?? code;
        message = body.error.message ?? message;
      }
    } catch {
      /* keep defaults */
    }
    throw new ApiError(code, message, resp.headers.get("X-Request-ID") ?? undefined);
  }
  return resp.json() as Promise<T>;
}

export const fetchHistory = (params: {
  start?: string;
  end?: string;
  granularity: Granularity;
  currency?: Currency;
  karat?: Karat;
  unit?: Unit;
}) => {
  const q = new URLSearchParams();
  if (params.start) q.set("start", params.start);
  if (params.end) q.set("end", params.end);
  q.set("granularity", params.granularity);
  if (params.currency) q.set("currency", params.currency);
  if (params.karat) q.set("karat", params.karat);
  if (params.unit) q.set("unit", params.unit);
  return request<HistoryResponse>(`/api/v1/history?${q.toString()}`);
};

export const fetchForecast = (
  horizon: Horizon,
  quantiles = true,
  currency: Currency = "usd",
  karat: Karat = "24k",
  unit: Unit = "10gram",
) =>
  request<ForecastResponse>(
    `/api/v1/forecast?horizon=${horizon}&quantiles=${quantiles}` +
      `&currency=${currency}&karat=${karat}&unit=${unit}`,
  );

export const fetchBacktest = () =>
  request<BacktestResponse>(`/api/v1/backtest/latest`);

export const fetchHealth = () => request<HealthResponse>(`/api/v1/health`);

export type IndiaSource = "bullion" | "retail";

export interface IndiaCity {
  slug: string;
  name: string;
  type: string;
  state_name: string | null;
}

export interface KaratPrices {
  "24k": number;
  "22k": number;
  "18k": number;
}

export interface IndiaRatePoint {
  date: string;
  price_24k_pg: number;
  price_22k_pg: number;
  price_18k_pg: number;
}

export interface IndiaCitiesResponse {
  cities: IndiaCity[];
}

export interface IndiaRatesResponse {
  city_slug: string;
  city: string;
  date: string;
  per_gram: KaratPrices;
  per_10g: KaratPrices;
  pct_change: KaratPrices | null;
  history: IndiaRatePoint[];
  source: string;
  disclaimer: string;
}

export interface IndiaForecastResponse extends ForecastResponse {
  premium_ratio: number;
  bullion_history_last_close: number;
  disclaimer: string;
}

export const fetchIndiaCities = () =>
  request<IndiaCitiesResponse>(`/api/v1/india/cities`);

export const fetchIndiaRates = (city: string, unit: Unit) =>
  request<IndiaRatesResponse>(
    `/api/v1/india/rates?city=${encodeURIComponent(city)}&unit=${unit}`,
  );

export const fetchIndiaForecast = (
  city: string,
  horizon: Horizon,
  karat: Karat,
  unit: Unit,
) =>
  request<IndiaForecastResponse>(
    `/api/v1/india/forecast?city=${encodeURIComponent(city)}` +
      `&horizon=${horizon}&karat=${karat}&unit=${unit}`,
  );
