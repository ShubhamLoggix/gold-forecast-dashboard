// API client for the gold forecast backend.
const BASE = import.meta.env.VITE_API_URL ?? ""; // "" => same-origin (vite proxy)

export type Granularity = "day" | "week" | "month";
export type Horizon = "1w" | "1m" | "3m" | "6m" | "1y";
export type Currency = "usd" | "inr";
export type Karat = "24k" | "22k" | "18k";
export type Unit = "gram" | "10gram" | "kg";
export type Metal = "gold" | "silver";
export type Fineness = "999" | "958" | "925";

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
  metal: Metal;
  karat: Karat | null;
  fineness: Fineness | null;
  unit: Unit | null;
  rate: RateInfo | null;
}

export interface BaselineSeries {
  name: string;
  values: number[];
}

export interface BandCalibration {
  scale: number;
  target_coverage_pct: number;
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
  metal: Metal;
  karat: Karat | null;
  fineness: Fineness | null;
  unit: Unit | null;
  rate: RateInfo | null;
  band_calibration: BandCalibration | null;
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
  metal?: Metal;
  fineness?: Fineness;
}) => {
  const q = new URLSearchParams();
  if (params.start) q.set("start", params.start);
  if (params.end) q.set("end", params.end);
  q.set("granularity", params.granularity);
  if (params.currency) q.set("currency", params.currency);
  if (params.karat) q.set("karat", params.karat);
  if (params.unit) q.set("unit", params.unit);
  if (params.metal) q.set("metal", params.metal);
  if (params.fineness) q.set("fineness", params.fineness);
  return request<HistoryResponse>(`/api/v1/history?${q.toString()}`);
};

export const fetchForecast = (
  horizon: Horizon,
  quantiles = true,
  currency: Currency = "usd",
  karat: Karat = "24k",
  unit: Unit = "10gram",
  metal: Metal = "gold",
  fineness: Fineness = "999",
) =>
  request<ForecastResponse>(
    `/api/v1/forecast?horizon=${horizon}&quantiles=${quantiles}` +
      `&currency=${currency}&karat=${karat}&unit=${unit}` +
      `&metal=${metal}&fineness=${fineness}`,
  );

export const fetchBacktest = () =>
  request<BacktestResponse>(`/api/v1/backtest/latest`);

export interface BacktestScoreEntry {
  horizon_days: number;
  generated_at: string;
  model_version: string;
  mape_pct: number;
  directional_accuracy_pct: number;
  naive_directional_accuracy_pct: number | null;
  skill_vs_naive_pp: number | null;
  band_coverage_pct: number | null;
  n_folds: number;
  underperforming_naive: boolean;
}

export interface BacktestScoreboard {
  entries: BacktestScoreEntry[];
}

export const fetchBacktestScoreboard = () =>
  request<BacktestScoreboard>(`/api/v1/backtest/scoreboard`);

export interface AccountabilityPoint {
  origin_date: string;
  horizon_days: number;
  target_date: string;
  series: string;
  predicted: number;
  actual: number;
  err_pct: number;
  in_band: boolean;
  direction_correct: boolean | null;
}

export interface RollingWindowStat {
  mae: number;
  mape_pct: number;
  n: number;
}

export interface AccountabilityHorizon {
  horizon_days: number;
  n_scored: number;
  mape_pct: number;
  mae: number;
  directional_acc_pct: number;
  band_coverage_pct: number;
  n_pending: number;
  windows: Record<string, RollingWindowStat>;
}

export interface AccountabilityResponse {
  per_horizon: AccountabilityHorizon[];
  per_series: Record<string, AccountabilityHorizon[]>;
  pending_counts: Record<string, number>;
  recent: AccountabilityPoint[];
  generated_at: string | null;
  disclaimer: string;
}

export const fetchAccountability = () =>
  request<AccountabilityResponse>(`/api/v1/accountability`);

export interface BakeoffEntry {
  horizon_days: number;
  horizon_label: string;
  model: string;
  mape_pct: number;
  directional_accuracy_pct: number;
  band_coverage_pct: number;
  calibrated_band_coverage_pct: number | null;
  band_scale: number | null;
  n_folds: number;
}

export interface BakeoffResponse {
  generated_at: string;
  data_range: string[];
  entries: BakeoffEntry[];
}

export const fetchBakeoff = () =>
  request<BakeoffResponse>(`/api/v1/backtest/bakeoff`);

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
  source: string;
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

export interface PremiumHistoryPoint {
  date: string;
  retail_pg: number;
  bullion_pg: number;
  premium_pg: number;
  quality: string;
}

export interface PremiumQualityCounts {
  real: number;
  estimated: number;
}

export interface PremiumForecastResponse {
  city: string;
  karat: Karat;
  series_id: string;
  generated_at: string;
  horizon: Horizon;
  horizon_days: number;
  model_version: string;
  latency_ms: number;
  last_date: string;
  last_retail_pg: number;
  last_bullion_pg: number;
  last_premium_pg: number;
  last_quality: string;
  quality_counts: PremiumQualityCounts;
  dates: string[];
  point: number[];
  q10: number[];
  q50: number[];
  q90: number[];
  history: PremiumHistoryPoint[];
  best_time_to_buy: boolean;
  best_time_to_buy_reason: string;
}

export const fetchPremiumForecast = (
  city: string,
  karat: Karat,
  horizon: Horizon,
) =>
  request<PremiumForecastResponse>(
    `/api/v1/premium/forecast?city=${encodeURIComponent(city)}` +
      `&karat=${karat}&horizon=${horizon}`,
  );
