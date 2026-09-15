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
  fetchForecast,
  fetchHealth,
  fetchHistory,
  type BacktestResponse,
  type ForecastResponse,
  type Granularity,
  type HealthResponse,
  type HistoryResponse,
  type Horizon,
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
  const lines = ["date,close,forecast_point,q10,q50,q90,naive_last_value,sma_20"];
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
    ];
    lines.push(cols.join(","));
  }
  return lines.join("\n");
}

export default function App() {
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [rangeLabel, setRangeLabel] = useState("1Y");
  const [horizon, setHorizon] = useState<Horizon>("1m");
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [useCustom, setUseCustom] = useState(false);

  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const range = RANGES.find((r) => r.label === rangeLabel)!;
      const params: Parameters<typeof fetchHistory>[0] = { granularity };
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
        fetchForecast(horizon),
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
  }, [granularity, rangeLabel, horizon, useCustom, customStart, customEnd]);

  useEffect(() => {
    void load();
  }, [load]);

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
    a.href = url;
    a.download = `gold_forecast_${forecast.horizon}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

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
        <div className="control-group">
          <label>Export</label>
          <button className="seg" onClick={downloadCsv} style={{ padding: "8px 14px" }}>
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
          <div className="metrics">
            <Metric label="Last close (GC=F)" value={`$${lastClose?.toLocaleString(undefined, { maximumFractionDigits: 2 })}`} />
            <Metric
              label={`Forecast ${horizon} (median)`}
              value={medianEnd ? `$${medianEnd.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "—"}
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

          <div className="chart-card">
            <ResponsiveContainer width="100%" height={420}>
              <ComposedChart data={rows} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eceef0" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={40} />
                <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} width={70} />
                <Tooltip
                  formatter={(value: number | string) =>
                    typeof value === "number" ? `$${value.toFixed(2)}` : value
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
            <div className="chart-caption">
              Shaded area = p10–p90 quantile band; dashed/dotted lines are naive
              baselines. If TimesFM does not clearly beat them, it is not adding value.
            </div>
          </div>

          <details className="model-info">
            <summary>Model info</summary>
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
                <strong>Data source:</strong> COMEX Gold Futures (GC=F) via Yahoo Finance;
                last data date {history?.end}
              </li>
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
