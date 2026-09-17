import { useEffect, useState } from "react";
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
  fetchPremiumForecast,
  type Horizon,
  type Karat,
  type PremiumForecastResponse,
} from "./api";

interface PremiumPanelProps {
  city: string;
  karat: Karat;
  horizon: Horizon;
}

interface Row {
  date: string;
  real: number | null;
  estimated: number | null;
  fc: number | null;
  lo: number | null;
  hi: number | null;
}

function buildRows(r: PremiumForecastResponse): Row[] {
  const rows: Row[] = [];
  for (const h of r.history) {
    rows.push({
      date: h.date,
      real: h.quality === "real" ? h.premium_pg : null,
      estimated: h.quality === "estimated" ? h.premium_pg : null,
      fc: null,
      lo: null,
      hi: null,
    });
  }
  for (let i = 0; i < r.dates.length; i++) {
    rows.push({
      date: r.dates[i],
      real: null,
      estimated: null,
      fc: i === 0 ? r.history.slice(-1)[0]?.premium_pg ?? r.point[i] : r.point[i],
      lo: r.q10[i],
      hi: r.q90[i],
    });
  }
  const lastDone = rows.length - r.dates.length;
  if (rows[lastDone]) {
    const hist = r.history.slice(-1)[0];
    rows[lastDone].lo = hist ? hist.premium_pg : rows[lastDone].lo;
    rows[lastDone].hi = hist ? hist.premium_pg : rows[lastDone].hi;
  }
  return rows;
}

const PREMIUM_DISCLAIMER =
  "Retail–COMEX premium = Groww retail rate − bullion-equivalent COMEX price (same date's " +
  "USD/INR), in ₹/g. 'real' points are live Groww quotes; 'estimated' points come from the " +
  "comex_converted backfill (context only, not evidence). The band is raw model p10–p90, " +
  "not calibrated like the gold-price chart.";

export default function PremiumPanel({ city, karat, horizon }: PremiumPanelProps) {
  const [data, setData] = useState<PremiumForecastResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchPremiumForecast(city, karat, horizon)
      .then((r) => {
        if (!cancelled) setData(r);
      })
      .catch((e: ApiError) => {
        if (!cancelled) setError(e);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [city, karat, horizon]);

  if (loading && !data) {
    return (
      <div className="chart-card">
        <div className="skeleton" />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="chart-card">
        <h3 style={{ marginTop: 0, fontSize: 16 }}>Retail premium forecast</h3>
        <div className="error-state" role="alert">
          {error?.message ?? "Premium series unavailable (needs Groww retail + COMEX data)."}
        </div>
      </div>
    );
  }

  const rows = buildRows(data);
  const reachableRows = rows.slice(-170);
  const lastHist = data.history.slice(-1)[0];
  const realCount = data.quality_counts.real;
  const estCount = data.quality_counts.estimated;

  return (
    <div className="chart-card">
      <h3 style={{ marginTop: 0, fontSize: 16 }}>
        Retail premium forecast — {karat.toUpperCase()} gold, {data.city}
      </h3>

      {data.best_time_to_buy && (
        <div
          role="alert"
          className="disclaimer"
          style={{ border: "1px solid #15803d", background: "rgba(21,128,61,0.06)" }}
        >
          <strong>Best time to buy?</strong> {data.best_time_to_buy_reason}
        </div>
      )}
      {!data.best_time_to_buy && (
        <div className="disclaimer" role="alert">
          {data.best_time_to_buy_reason}
        </div>
      )}

      <div style={{ height: 260, marginTop: 12 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={reachableRows} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eceef0" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} tickMargin={6} minTickGap={40} />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => `₹${Math.round(v).toLocaleString()}`}
              width={70}
            />
            <Tooltip
              formatter={(value: number | number[] | string, name: string) => {
                const lineName =
                  name === "real"
                    ? "Premium (real)"
                    : name === "estimated"
                    ? "Premium (est.)"
                    : name === "fc"
                    ? "Forecast"
                    : name === "lo"
                    ? "Band bottom"
                    : "Band top";
                const v = Array.isArray(value) ? value[0] : value;
                return [typeof v === "number" ? `₹${v.toLocaleString(undefined, { maximumFractionDigits: 1 })}` : String(v), lineName];
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Area
              dataKey="lo"
              name="band"
              stroke="none"
              fill="rgba(59,130,246,0.10)"
              stackId="band"
              connectNulls
            />
            <Area
              dataKey="hi"
              name="band-top"
              stroke="none"
              fill="transparent"
              stackId="band"
              connectNulls
            />
            <Line dataKey="estimated" name="estimated" stroke="#9ca3af" strokeWidth={1} dot={false} connectNulls />
            <Line dataKey="real" name="real" stroke="#0f766e" strokeWidth={1.6} dot={{ r: 2.4 }} connectNulls />
            <Line dataKey="fc" name="fc" stroke="#d4a404" strokeWidth={2.2} dot={false} connectNulls />
            <ReferenceLine
              y={lastHist ? lastHist.premium_pg : undefined}
              stroke="#6b7280"
              strokeDasharray="4 4"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="metric" style={{ marginTop: 10 }}>
        <div className="label">
          Latest premium (₹/g) — {data.last_quality === "real" ? "live quote" : "estimated"}
        </div>
        <div className="value">
          ₹{data.last_premium_pg.toLocaleString("en-IN", { maximumFractionDigits: 1 })}
        </div>
      </div>
      <div style={{ display: "flex", gap: 24, marginTop: 4 }}>
        <span style={{ fontSize: 12, color: "#6b7280" }}>
          {realCount} real · {estCount} estimated rows
        </span>
        <span style={{ fontSize: 12, color: "#6b7280" }}>
          Forecast horizon {horizon} · {data.model_version}
        </span>
      </div>
      <div className="chart-caption" style={{ marginTop: 8 }}>
        {PREMIUM_DISCLAIMER}
      </div>
    </div>
  );
}