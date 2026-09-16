import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiError,
  fetchBacktest,
  fetchAccountability,
  fetchBacktestScoreboard,
  fetchForecast,
  fetchHealth,
  fetchHistory,
  fetchIndiaCities,
  fetchIndiaForecast,
  fetchIndiaRates,
  type BacktestResponse,
  type AccountabilityResponse,
  type BacktestScoreboard,
  type Currency,
  type ForecastResponse,
  type Granularity,
  type HealthResponse,
  type HistoryResponse,
  type Horizon,
  type IndiaCity,
  type IndiaForecastResponse,
  type IndiaRatesResponse,
  type Karat,
  type Unit,
} from "./api";

const GRANULARITIES: Granularity[] = ["day", "week", "month"];
const RANGES: { label: string; days: number | null }[] = [
  { label: "1M", days: 21 },
  { label: "6M", days: 126 },
  { label: "1Y", days: 252 },
  { label: "5Y", days: 1260 },
  { label: "Max", days: null },
];
const HORIZONS: Horizon[] = ["1w", "1m", "3m", "6m", "1y"];
const KARATS: Karat[] = ["24k", "22k", "18k"];
const UNITS: { id: Unit; label: string }[] = [
  { id: "10gram", label: "per 10g" },
  { id: "gram", label: "per g" },
];

const INR_CAPTION =
  "Converted from COMEX USD futures at the live USD/INR rate — this is a theoretical " +
  "bullion-equivalent price, not an Indian retail/jeweler quote, which also includes " +
  "import duty, GST, and making charges. Historical points use each date's own " +
  "USD/INR rate; the forecast uses the latest rate.";

const RETAIL_FALLBACK_CITIES: IndiaCity[] = [
  { slug: "pune", name: "Pune", type: "city", state_name: "Maharashtra" },
];

const HORIZON_DAYS: Record<Horizon, number> = {
  "1w": 5,
  "1m": 21,
  "3m": 63,
  "6m": 126,
  "1y": 252,
};
const HORIZON_LABEL: Record<number, string> = {
  5: "1w",
  21: "1m",
  63: "3m",
  126: "6m",
  252: "1y",
};

interface Row {
  date: string;
  hist: number | null;
  point: number | null;
  bandBase: number | null;
  bandWidth: number | null;
  naive: number | null;
  sma: number | null;
}

function DisclaimerBanner() {
  return (
    <div className="disclaimer" role="alert">
      <strong>Disclaimer — not investment advice.</strong> Price forecasts are
      experimental; gold/financial time series are close to random walks. Never use this
      as the sole basis for a financial decision.
    </div>
  );
}

function Metric({
  label,
  value,
  delta,
  deltaColor,
}: {
  label: string;
  value: string;
  delta?: string;
  deltaColor?: "up" | "down";
}) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {delta && (
        <div className={deltaColor === "up" ? "delta-up" : "delta-down"}>{delta}</div>
      )}
    </div>
  );
}

function buildRows(hist: HistoryResponse, fc: ForecastResponse): Row[] {
  const rows: Row[] = hist.points.map((p) => ({
    date: p.date,
    hist: p.close,
    point: null,
    bandBase: null,
    bandWidth: null,
    naive: null,
    sma: null,
  }));
  // Bridge row so the forecast lines visually connect to the history end.
  const lastHist = hist.points[hist.points.length - 1];
  rows.push({
    date: fc.history_last_date,
    hist: lastHist.close,
    point: lastHist.close,
    bandBase: lastHist.close,
    bandWidth: 0,
    naive: lastHist.close,
    sma: lastHist.close,
  });
  fc.dates.forEach((d, i) => {
    rows.push({
      date: d,
      hist: null,
      point: fc.point[i],
      bandBase: fc.q10[i],
      bandWidth: fc.q90[i] - fc.q10[i],
      naive: fc.baselines[0]?.values[i] ?? null,
      sma: fc.baselines[1]?.values[i] ?? fc.baselines[0]?.values[i] ?? null,
    });
  });
  return rows;
}

function toCsv(hist: HistoryResponse, fc: ForecastResponse): string {
  const currency = fc.currency ?? "usd";
  const karat = fc.karat ?? "";
  const unit = fc.unit ?? "";
  const lines = [
    `date,close,forecast_point,q10,q50,q90,naive_last_value,sma_20,currency,karat,unit`,
  ];
  const histIdx = new Map(hist.points.map((p) => [p.date, p.close]));
  const allDates = [
    ...hist.points.map((p) => p.date),
    ...fc.dates,
  ];
  for (const d of allDates) {
    const i = fc.dates.indexOf(d);
    const cols = [
      d,
      histIdx.get(d)?.toString() ?? "",
      i >= 0 ? fc.point[i].toFixed(4) : "",
      i >= 0 ? fc.q10[i].toFixed(4) : "",
      i >= 0 ? fc.q50[i].toFixed(4) : "",
      i >= 0 ? fc.q90[i].toFixed(4) : "",
      i >= 0 ? fc.baselines[0]?.values[i].toFixed(4) ?? "" : "",
      i >= 0 ? fc.baselines[1]?.values[i].toFixed(4) ?? "" : "",
      currency,
      karat,
      unit,
    ];
    lines.push(cols.join(","));
  }
  return lines.join("\n");
}

export default function App() {
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [rangeLabel, setRangeLabel] = useState("1Y");
  const [horizon, setHorizon] = useState<Horizon>("1m");
  const [currency, setCurrency] = useState<Currency>("usd");
  const [karat, setKarat] = useState<Karat>("22k");
  const [unit, setUnit] = useState<Unit>("10gram");
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [inrSource, setInrSource] = useState<"bullion" | "retail">("bullion");
  const [city, setCity] = useState("pune");
  const [cities, setCities] = useState<IndiaCity[] | null>(null);
  const [india, setIndia] = useState<IndiaRatesResponse | null>(null);
  const [indiaForecast, setIndiaForecast] = useState<IndiaForecastResponse | null>(null);
  const [indiaError, setIndiaError] = useState<ApiError | null>(null);
  const [indiaLoading, setIndiaLoading] = useState(false);
  const [scoreboard, setScoreboard] = useState<BacktestScoreboard | null>(null);
  const [accountability, setAccountability] = useState<AccountabilityResponse | null>(null);

  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  const isInr = currency === "inr";
  const isRetail = isInr && inrSource === "retail";
  const unitLabel = isRetail
    ? `INR retail (Groww) — ${india?.city ?? city}, per ${unit === "10gram" ? "10g" : "g"} (${karat.toUpperCase()})`
    : isInr
    ? `INR per ${unit === "10gram" ? "10g" : "g"} (${karat.toUpperCase()}, bullion-equiv.)`
    : "USD/oz (COMEX GC=F)";

  const formatValue = useCallback(
    (v: number) => isInr ? `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
    [isInr],
  );

  const load = useCallback(async () => {
    if (isRetail) return; // retail mode fetches its own data
    setLoading(true);
    setError(null);
    try {
      const range = RANGES.find((r) => r.label === rangeLabel)!;
      const params: Parameters<typeof fetchHistory>[0] = {
        granularity,
        currency,
        karat,
        unit,
      };
      if (useCustom && customStart) params.start = customStart;
      if (useCustom && customEnd) params.end = customEnd;
      if (!useCustom && range.days) {
        const end = new Date();
        const start = new Date(end.getTime() - range.days * 24 * 3600 * 1000);
        params.start = start.toISOString().slice(0, 10);
        params.end = end.toISOString().slice(0, 10);
      }
      const [h, f, b, hl] = await Promise.all([
        fetchHistory(params),
        fetchForecast(horizon, true, currency, karat, unit),
        fetchBacktest().catch(() => null),
        fetchHealth().catch(() => null),
      ]);
      setHistory(h);
      setForecast(f);
      setBacktest(b);
      setHealth(hl);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setLoading(false);
    }
  }, [granularity, rangeLabel, horizon, currency, karat, unit, useCustom, customStart, customEnd, isRetail]);

  const loadRetail = useCallback(async () => {
    setIndiaLoading(true);
    setIndiaError(null);
    try {
      const [rates, fc] = await Promise.all([
        fetchIndiaRates(city, unit),
        fetchIndiaForecast(city, horizon, karat, unit).catch(() => null),
      ]);
      setIndia(rates);
      setIndiaForecast(fc);
    } catch (e) {
      setIndiaError(e as ApiError);
      setIndiaForecast(null);
    } finally {
      setIndiaLoading(false);
    }
  }, [city, unit, horizon, karat]);

  useEffect(() => {
    if (isRetail) void loadRetail();
  }, [isRetail, loadRetail]);

  useEffect(() => {
    if (!isRetail || cities) return;
    fetchIndiaCities()
      .then((r) => setCities(r.cities.length ? r.cities : RETAIL_FALLBACK_CITIES))
      .catch(() => setCities(RETAIL_FALLBACK_CITIES));
  }, [isRetail, cities]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    fetchBacktestScoreboard()
      .then(setScoreboard)
      .catch(() => setScoreboard(null));
    fetchAccountability()
      .then(setAccountability)
      .catch(() => setAccountability(null));
  }, []);

  const rows = useMemo(
    () => (history && forecast ? buildRows(history, forecast) : []),
    [history, forecast],
  );

  const tmSummary = backtest?.results["timesfm-2.5"]?.summary;
  const lastClose = forecast?.history_last_close ?? history?.points.slice(-1)[0]?.close;
  const medianEnd = forecast?.q50.slice(-1)[0];
  const impliedPct =
    lastClose && medianEnd ? (medianEnd / lastClose - 1) * 100 : null;

  const downloadCsv = () => {
    if (!history || !forecast) return;
    const blob = new Blob([toCsv(history, forecast)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const unitTag =
      forecast.currency === "inr"
        ? forecast.unit === "gram" ? "per1g" : "per10g"
        : "peroz";
    a.download =
      `gold_forecast_${forecast.horizon}_${forecast.currency}` +
      `${forecast.karat ? `_${forecast.karat}` : ""}` +
      `_${unitTag}` +
      `_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadRetailCsv = () => {
    if (!india) return;
    const lines = [
      `date,price_24k_pg,price_22k_pg,price_18k_pg,price_24k_p10g,price_22k_p10g,price_18k_p10g,city,source`,
    ];
    for (const p of india.history) {
      lines.push(
        [
          p.date,
          p.price_24k_pg,
          p.price_22k_pg,
          p.price_18k_pg,
          p.price_24k_pg * 10,
          p.price_22k_pg * 10,
          p.price_18k_pg * 10,
          india.city,
          "groww",
        ].join(","),
      );
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.download = `india_retail_gold_${city}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const retailRows = useMemo(() => {
    if (!india) return [];
    const pgKey =
      karat === "24k" ? "price_24k_pg" : karat === "22k" ? "price_22k_pg" : "price_18k_pg";
    const factor = unit === "10gram" ? 10 : 1;
    const histLike = {
      points: india.history.map((p) => ({ date: p.date, close: p[pgKey] * factor })),
    } as unknown as HistoryResponse;
    if (!indiaForecast) {
      return histLike.points.map((p) => ({
        date: p.date,
        hist: p.close,
        point: null,
        bandBase: null,
        bandWidth: null,
        naive: null,
        sma: null,
      }));
    }
    return buildRows(histLike, indiaForecast as unknown as ForecastResponse);
  }, [india, indiaForecast, karat, unit]);

  return (
    <div className="container">
      <h1>Gold Price Forecast — TimesFM</h1>
      <DisclaimerBanner />
      {health?.data_may_be_stale && (
        <div className="stale-flag">
          Data may be stale (last refresh failed or is older than 4 days). Last data
          date: {health.data_last_date ?? "unknown"}.
        </div>
      )}

      <div className="controls">
        <div className="control-group">
          <label>Currency</label>
          <div className="seg">
            {(["usd", "inr"] as Currency[]).map((c) => (
              <button
                key={c}
                className={currency === c ? "active" : ""}
                onClick={() => setCurrency(c)}
              >
                {c.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        {isInr && (
          <>
            <div className="control-group">
              <label>Source</label>
              <div className="seg">
                {(["bullion", "retail"] as const).map((s) => (
                  <button
                    key={s}
                    className={inrSource === s ? "active" : ""}
                    onClick={() => setInrSource(s)}
                  >
                    {s === "bullion" ? "Bullion (COMEX)" : "Retail (Groww)"}
                  </button>
                ))}
              </div>
            </div>
            {isRetail && (
              <div className="control-group">
                <label>City</label>
                <select
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid #d7dbe0" }}
                >
                  {(cities ?? RETAIL_FALLBACK_CITIES).map((c) => (
                    <option key={c.slug} value={c.slug}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <div className="control-group">
              <label>Karat</label>
              <div className="seg">
                {KARATS.map((k) => (
                  <button
                    key={k}
                    className={karat === k ? "active" : ""}
                    onClick={() => setKarat(k)}
                  >
                    {k.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="control-group">
              <label>Unit</label>
              <div className="seg">
                {UNITS.map((u) => (
                  <button
                    key={u.id}
                    className={unit === u.id ? "active" : ""}
                    onClick={() => setUnit(u.id)}
                  >
                    {u.label}
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
        {!isRetail && (
        <>
        <div className="control-group">
          <label>Granularity</label>
          <div className="seg">
            {GRANULARITIES.map((g) => (
              <button
                key={g}
                className={granularity === g ? "active" : ""}
                onClick={() => setGranularity(g)}
              >
                {g[0].toUpperCase() + g.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="control-group">
          <label>Range</label>
          <div className="seg">
            {RANGES.map((r) => (
              <button
                key={r.label}
                className={rangeLabel === r.label ? "active" : ""}
                onClick={() => {
                  setRangeLabel(r.label);
                  setUseCustom(false);
                }}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>
        </>)}
        <div className="control-group">
          <label>Forecast horizon</label>
          <div className="seg">
            {HORIZONS.map((h) => (
              <button
                key={h}
                className={horizon === h ? "active" : ""}
                onClick={() => setHorizon(h)}
              >
                {h}
              </button>
            ))}
          </div>
        </div>
        {!isRetail && (
        <>
        <div className="control-group">
          <label>Custom dates</label>
          <div className="date-inputs">
            <input
              type="date"
              value={customStart}
              onChange={(e) => {
                setCustomStart(e.target.value);
                setUseCustom(true);
              }}
            />
            <input
              type="date"
              value={customEnd}
              onChange={(e) => {
                setCustomEnd(e.target.value);
                setUseCustom(true);
              }}
            />
          </div>
        </div>
        </>)}
        <div className="control-group">
          <label>Export</label>
          <button
            className="seg"
            onClick={() => (isRetail ? downloadRetailCsv() : downloadCsv())}
            style={{ padding: "8px 14px" }}
          >
            CSV
          </button>
        </div>
      </div>

      {error ? (
        <div className="error-state">
          <strong>Failed to load data.</strong>
          <div>{error.message}</div>
          {error.requestId && <div>request id: {error.requestId}</div>}
          <button onClick={() => void load()}>Retry</button>
        </div>
      ) : loading ? (
        <div className="skeleton" />
      ) : (
        <>
          {isRetail && india && (
            <div className="metrics">
              {KARATS.map((k) => {
                const val = unit === "10gram" ? india.per_10g[k] : india.per_gram[k];
                const pct = india.pct_change?.[k];
                return (
                  <Metric
                    key={k}
                    label={`${k.toUpperCase()} — ${india.city} (per ${unit === "10gram" ? "10g" : "g"})`}
                    value={formatValue(val)}
                    delta={
                      pct != null
                        ? `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}% vs prev day`
                        : undefined
                    }
                    deltaColor={pct != null && pct >= 0 ? "up" : "down"}
                  />
                );
              })}
              {indiaForecast && (
                <Metric
                  label={`Forecast ${horizon} (est. retail, ${karat.toUpperCase()})`}
                  value={formatValue(indiaForecast.q50[indiaForecast.q50.length - 1])}
                  delta={
                    (() => {
                      const end = indiaForecast.q50[indiaForecast.q50.length - 1];
                      const pct = (end / indiaForecast.history_last_close - 1) * 100;
                      return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}% est.`;
                    })()
                  }
                  deltaColor={
                    indiaForecast.q50[indiaForecast.q50.length - 1] >=
                    indiaForecast.history_last_close
                      ? "up"
                      : "down"
                  }
                />
              )}
            </div>
          )}
          {!isRetail && (
          <div className="metrics">
            <Metric
              label={isInr ? `Last close (${karat.toUpperCase()}, per ${unit === "10gram" ? "10g" : "g"})` : "Last close (GC=F)"}
              value={lastClose != null ? formatValue(lastClose) : "—"}
            />
            <Metric
              label={`Forecast ${horizon} (median)`}
              value={medianEnd != null ? formatValue(medianEnd) : "—"}
              delta={impliedPct != null ? `${impliedPct >= 0 ? "+" : ""}${impliedPct.toFixed(1)}%` : undefined}
              deltaColor={impliedPct != null && impliedPct >= 0 ? "up" : "down"}
            />
            <Metric
              label="Backtest MAPE (30d)"
              value={tmSummary ? `${tmSummary.mape_pct.toFixed(2)}%` : "n/a"}
            />
            <Metric
              label="Backtest dir. accuracy"
              value={tmSummary ? `${tmSummary.directional_accuracy_pct.toFixed(1)}%` : "n/a"}
              delta={tmSummary ? `${(tmSummary.directional_accuracy_pct - 50).toFixed(1)}pp vs chance` : undefined}
              deltaColor={tmSummary && tmSummary.directional_accuracy_pct >= 50 ? "up" : "down"}
            />
          </div>
          )}

          {isRetail && (
            <div className="chart-card">
              {indiaLoading && <div className="skeleton" />}
              {indiaError && (
                <div className="error-state">
                  <strong>Failed to load India retail rates.</strong>
                  <div>{indiaError.message}</div>
                  <button onClick={() => void loadRetail()}>Retry</button>
                </div>
              )}
              {!indiaLoading && !indiaError && india && (
                <>
                  <ResponsiveContainer width="100%" height={420}>
                    <ComposedChart
                      data={retailRows}
                      margin={{ top: 10, right: 10, bottom: 0, left: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#eceef0" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={24} />
                      <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} width={70} />
                      <Tooltip
                        formatter={(value: number | string) =>
                          typeof value === "number" ? formatValue(value) : value
                        }
                      />
                      <Legend />
                      <Area
                        dataKey="bandBase"
                        stackId="band"
                        stroke="none"
                        fill="none"
                        connectNulls
                        isAnimationActive={false}
                      />
                      <Area
                        dataKey="bandWidth"
                        stackId="band"
                        stroke="none"
                        fill="rgba(217,164,6,0.22)"
                        name="Est. retail band p10–p90"
                        connectNulls
                        isAnimationActive={false}
                      />
                      <Line
                        dataKey="hist"
                        stroke="#1266a2"
                        strokeWidth={2.2}
                        dot={false}
                        name={`Retail close (${karat.toUpperCase()})`}
                        connectNulls={false}
                      />
                      <Line
                        dataKey="point"
                        stroke="#d9a406"
                        strokeWidth={2.4}
                        dot={false}
                        name="Est. retail forecast (p50)"
                        connectNulls={false}
                      />
                      <Line
                        dataKey="naive"
                        stroke="#888"
                        strokeWidth={1.2}
                        strokeDasharray="6 4"
                        dot={false}
                        name="Naive (scaled)"
                        connectNulls={false}
                      />
                      <Line
                        dataKey="sma"
                        stroke="#bb5588"
                        strokeWidth={1.2}
                        strokeDasharray="2 3"
                        dot={false}
                        name="SMA(20) (scaled)"
                        connectNulls={false}
                      />
                      <ReferenceLine
                        x={indiaForecast?.history_last_date}
                        stroke="#d9a406"
                        strokeDasharray="4 4"
                        label="today"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                  <div className="chart-caption" data-testid="unit-label" style={{ fontWeight: 600 }}>
                    {unitLabel}
                  </div>
                  <div className="chart-caption" style={{ marginTop: 6 }}>
                    Source: Groww gold rates — published retail quotes for {india.city} as of{" "}
                    {india.date} (last {india.history.length} available days; Groww only
                    publishes ~10 days — history accumulates here daily). Retail rates
                    include import duty, GST, and local dealer premium — they intentionally
                    differ from bullion prices.
                  </div>
                  {indiaForecast && (
                    <div className="chart-caption" style={{ marginTop: 6 }}>
                      Forecast = TimesFM COMEX bullion forecast ×{" "}
                      {indiaForecast.premium_ratio.toFixed(3)} (current retail premium vs
                      bullion-equivalent). {indiaForecast.disclaimer}
                    </div>
                  )}
                  <div className="chart-caption" style={{ marginTop: 6 }}>
                    {india.disclaimer}
                  </div>
                </>
              )}
            </div>
          )}
          {!isRetail && (
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={420}>
              <ComposedChart data={rows} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eceef0" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={40} />
                <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} width={70} />
                <Tooltip
                  formatter={(value: number | string) =>
                    typeof value === "number" ? formatValue(value) : value
                  }
                />
                <Legend />
                <Area
                  dataKey="bandBase"
                  stackId="band"
                  stroke="none"
                  fill="none"
                  connectNulls
                  isAnimationActive={false}
                />
                <Area
                  dataKey="bandWidth"
                  stackId="band"
                  stroke="none"
                  fill="rgba(217,164,6,0.22)"
                  name="p10–p90 band"
                  connectNulls
                  isAnimationActive={false}
                />
                <Line dataKey="hist" stroke="#1266a2" strokeWidth={2} dot={false} name="Historical close" connectNulls={false} />
                <Line dataKey="point" stroke="#d9a406" strokeWidth={2.4} dot={false} name="TimesFM forecast (p50)" connectNulls={false} />
                <Line dataKey="naive" stroke="#888" strokeWidth={1.2} strokeDasharray="6 4" dot={false} name="Naive last-value" connectNulls={false} />
                <Line dataKey="sma" stroke="#bb5588" strokeWidth={1.2} strokeDasharray="2 3" dot={false} name="SMA(20) baseline" connectNulls={false} />
                <ReferenceLine x={forecast?.history_last_date} stroke="#d9a406" strokeDasharray="4 4" label="today" />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="chart-caption" data-testid="unit-label" style={{ fontWeight: 600 }}>
              {unitLabel}
            </div>
            {isInr && (
              <div className="chart-caption" data-testid="inr-disclaimer" style={{ marginTop: 6 }}>
                {INR_CAPTION}
                {forecast?.rate && (
                  <>
                    {" "}Converted at ₹{forecast.rate.usd_inr_rate.toFixed(4)}/USD as of{" "}
                    {forecast.rate.usd_inr_rate_date}
                    {forecast.rate.rate_may_be_stale && (
                      <strong> (rate may be stale — last available FX date used)</strong>
                    )}
                    .
                  </>
                )}
              </div>
            )}
            <div className="chart-caption" style={{ marginTop: 6 }}>
              Shaded area = p10–p90 quantile band; dashed/dotted lines are naive
              baselines. If TimesFM does not clearly beat them, it is not adding value.
              {forecast?.band_calibration && (
                <>
                  {" "}Band scaled ×{forecast.band_calibration.scale} so that ~
                  {forecast.band_calibration.target_coverage_pct}% of historical
                  out-of-sample outcomes fell inside it.
                </>
              )}
            </div>
          </div>
          )}

          {scoreboard && scoreboard.entries.length > 0 && (
            <div className="chart-card">
              <h3 style={{ marginTop: 0, fontSize: 16 }}>
                Can you trust this forecast? — measured accuracy, not promises
              </h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ textAlign: "left", color: "#6b7280" }}>
                      <th style={{ padding: "6px 8px" }}>Horizon</th>
                      <th style={{ padding: "6px 8px" }}>Avg error (MAPE)</th>
                      <th style={{ padding: "6px 8px" }}>Direction accuracy</th>
                      <th style={{ padding: "6px 8px" }}>vs naive carry-forward</th>
                      <th style={{ padding: "6px 8px" }}>Band honesty (p10–p90 hit rate)</th>
                      <th style={{ padding: "6px 8px" }}>Folds</th>
                      <th style={{ padding: "6px 8px" }}>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scoreboard.entries.map((e) => (
                      <tr
                        key={e.horizon_days}
                        style={{
                          borderTop: "1px solid #eceef0",
                          background:
                            HORIZON_DAYS[horizon] === e.horizon_days
                              ? "rgba(217,164,6,0.10)"
                              : undefined,
                        }}
                      >
                        <td style={{ padding: "6px 8px", fontWeight: 700 }}>
                          {HORIZON_LABEL[e.horizon_days] ?? `${e.horizon_days}d`}
                        </td>
                        <td style={{ padding: "6px 8px" }}>{e.mape_pct.toFixed(2)}%</td>
                        <td style={{ padding: "6px 8px" }}>
                          {e.directional_accuracy_pct.toFixed(1)}%
                        </td>
                        <td style={{ padding: "6px 8px" }}>
                          {e.naive_directional_accuracy_pct != null
                            ? `${e.naive_directional_accuracy_pct.toFixed(1)}% (${
                                (e.skill_vs_naive_pp ?? 0) >= 0 ? "+" : ""
                              }${e.skill_vs_naive_pp}pp)`
                            : "—"}
                        </td>
                        <td style={{ padding: "6px 8px" }}>
                          {e.band_coverage_pct != null
                            ? `${e.band_coverage_pct.toFixed(1)}% (honest ≈ 80%)`
                            : "—"}
                        </td>
                        <td style={{ padding: "6px 8px" }}>{e.n_folds}</td>
                        <td style={{ padding: "6px 8px", color: e.underperforming_naive ? "#b91c1c" : "#166534" }}>
                          {e.underperforming_naive
                            ? "worse than naive — do not rely on it"
                            : "beats naive carry-forward"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="chart-caption" style={{ marginTop: 8 }}>
                Measured on real out-of-sample data (walk-forward backtest; each fold
                forecasts a period the model never saw). MAPE = average % error.
                Band honesty = how often the actual price stayed inside the shaded
                p10–p90 band — ≈80% means the band is trustworthy.{" "}
                <strong>
                  No forecast can be 99% accurate — treat every point value as a
                  scenario, not a promise.
                </strong>
              </div>
              {accountability &&
                (accountability.per_horizon.length > 0 ||
                  accountability.recent.length > 0) && (
                <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid #eceef0" }}>
                  <div style={{ fontWeight: 700, fontSize: 14 }}>
                    Live track record — forecasts actually logged, scored as reality
                    arrives
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, marginTop: 6 }}>
                    <thead>
                      <tr style={{ textAlign: "left", color: "#6b7280" }}>
                        <th style={{ padding: "4px 8px" }}>Horizon</th>
                        <th style={{ padding: "4px 8px" }}>Scored</th>
                        <th style={{ padding: "4px 8px" }}>Realized MAPE</th>
                        <th style={{ padding: "4px 8px" }}>Realized band coverage</th>
                        <th style={{ padding: "4px 8px" }}>Pending</th>
                      </tr>
                    </thead>
                    <tbody>
                      {accountability.per_horizon.map((h) => (
                        <tr key={h.horizon_days} style={{ borderTop: "1px solid #eceef0" }}>
                          <td style={{ padding: "4px 8px", fontWeight: 700 }}>
                            {HORIZON_LABEL[h.horizon_days] ?? `${h.horizon_days}d`}
                          </td>
                          <td style={{ padding: "4px 8px" }}>{h.n_scored}</td>
                          <td style={{ padding: "4px 8px" }}>
                            {h.n_scored > 0 ? `${h.mape_pct.toFixed(2)}%` : "—"}
                          </td>
                          <td style={{ padding: "4px 8px" }}>
                            {h.n_scored > 0 ? `${h.band_coverage_pct.toFixed(1)}%` : "—"}
                          </td>
                          <td style={{ padding: "4px 8px" }}>{h.n_pending}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {accountability.recent.length > 0 && (
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, marginTop: 8 }}>
                      <thead>
                        <tr style={{ textAlign: "left", color: "#6b7280" }}>
                          <th style={{ padding: "4px 8px" }}>Made on</th>
                          <th style={{ padding: "4px 8px" }}>For date</th>
                          <th style={{ padding: "4px 8px" }}>Predicted</th>
                          <th style={{ padding: "4px 8px" }}>Actual</th>
                          <th style={{ padding: "4px 8px" }}>Error</th>
                          <th style={{ padding: "4px 8px" }}>In band?</th>
                        </tr>
                      </thead>
                      <tbody>
                        {accountability.recent.slice(0, 6).map((p, i) => (
                          <tr key={i} style={{ borderTop: "1px solid #f3f4f6" }}>
                            <td style={{ padding: "4px 8px" }}>{p.origin_date}</td>
                            <td style={{ padding: "4px 8px" }}>{p.target_date}</td>
                            <td style={{ padding: "4px 8px" }}>{formatValue(p.predicted)}</td>
                            <td style={{ padding: "4px 8px" }}>{formatValue(p.actual)}</td>
                            <td style={{ padding: "4px 8px" }}>{p.err_pct.toFixed(2)}%</td>
                            <td style={{ padding: "4px 8px", color: p.in_band ? "#166534" : "#b91c1c" }}>
                              {p.in_band ? "yes" : "no"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </div>
          )}

          <details className="model-info">
            <summary>Model info</summary>
            {backtest?.model_drift?.underperforming && (
              <div className="disclaimer" style={{ marginTop: 10 }}>
                <strong>MODEL UNDERPERFORMING BASELINE</strong> — TimesFM directional
                accuracy is {Math.abs(backtest.model_drift.delta_pp).toFixed(1)} pp{" "}
                <em>below</em> the naive carry-forward baseline (
                {backtest.model_drift.timesfm_dir_acc_pct.toFixed(1)}% vs{" "}
                {backtest.model_drift.naive_dir_acc_pct.toFixed(1)}%). Treat all
                forecasts with extra skepticism.
              </div>
            )}
            <ul>
              <li>
                <strong>Model:</strong> TimesFM {forecast?.model_version} (
                <code>google/timesfm-2.5-200m-pytorch</code> for v2.5, Apache-2.0)
              </li>
              <li>
                <strong>License:</strong>{" "}
                {forecast?.model_version === "2.5"
                  ? "Apache-2.0 (commercial-safe)"
                  : "timesfm-non-commercial-license-v1.0 (NON-COMMERCIAL)"}
              </li>
              <li>
                <strong>Training data:</strong> large public + synthetic time-series
                corpora — see the TimesFM model card; not trained on up-to-the-minute
                gold data
              </li>
              <li>
                <strong>Data source:</strong>{" "}
                {isRetail
                  ? `Indian city retail rates (Groww gold rates) — ${india?.city ?? city}, as of ${india?.date ?? "—"}`
                  : `COMEX Gold Futures (GC=F) via Yahoo Finance; last data date ${history?.end}`}
              </li>
              {isInr && forecast?.rate && (
                <li>
                  <strong>INR conversion:</strong> ₹{forecast.rate.usd_inr_rate.toFixed(4)}/USD as
                  of {forecast.rate.usd_inr_rate_date}
                  {forecast.rate.rate_may_be_stale && " (rate may be stale)"}. Theoretical
                  bullion-equivalent price at {karat.toUpperCase()} purity — <em>not</em> an
                  Indian retail/jeweler quote (import duty, GST, and making charges excluded).
                </li>
              )}
              <li>
                <strong>Known limitations:</strong> zero-shot TSFMs perform near chance
                level (~50% directional accuracy) on financial series; the quantile band
                is model-internal uncertainty, not calibrated risk. Improvement typically
                requires financial-domain fine-tuning, which this project does not
                include.
              </li>
            </ul>
          </details>
        </>
      )}

      <div className="footer-disclaimer">
        <strong>Disclaimer — not investment advice.</strong> This is a forecasting demo.
        Gold and financial time series behave close to random walks; published benchmarks
        show zero-shot time-series foundation models (including TimesFM) reach only ~50%
        directional accuracy on such series. Never rely on this dashboard as the sole
        basis for a financial decision. Data: COMEX Gold Futures (GC=F) via Yahoo Finance.
      </div>
    </div>
  );
}
