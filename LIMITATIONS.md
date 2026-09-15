# LIMITATIONS — read this before trusting anything on the dashboard

## The one-line version

Zero-shot time-series foundation models — including TimesFM — perform close to
chance level (~50% directional accuracy) on raw financial return series. Gold is
one of the hardest cases. The numbers on this dashboard's backtest panel exist
precisely so you can see that for yourself.

## What the backtest actually showed (Sept 2026 run, COMEX GC=F, 30-trading-day horizon)

| Model | MAPE | Directional accuracy |
|---|---|---|
| TimesFM 2.5 (zero-shot) | ~3.4% | ~48.7% |
| Naive last-value carry-forward | ~3.1% | ~0% (predicts "flat" always) |
| SMA(20) | ~3.8% | ~1% |

TimesFM's ~48.7% directional accuracy is statistically indistinguishable from
coin-flipping. Its point-forecast MAPE is not better than the naive
carry-forward either. This matches published benchmark results for foundation
models on financial series (e.g., evaluations in the TimesFM and Moirai papers
and independent benchmarks), where raw financial returns behave close to a
random walk and any model that does not use domain-specific features or
fine-tuning lands near 50% directional accuracy.

## Why this is expected, not a bug

1. **Near-random-walk data.** Daily gold returns have tiny signal relative to
   noise; the theoretically optimal point forecast for a random walk is
   "yesterday's price", which is exactly what the naive baseline does.
2. **Zero-shot, no domain fine-tuning.** TimesFM was trained on broad,
   general-purpose time-series corpora — not to extract signal from financial
   microstructure. Meaningful improvement typically requires financial-domain
   fine-tuning (or feature/covariate engineering), which this project's default
   configuration deliberately does not include.
3. **The quantile band is model-internal uncertainty**, not calibrated risk.
   Treat p10–p90 as a rough spread, not a confidence interval with known
   coverage.

## What the dashboard is for

- Learning how TimesFM works and how to productionize a forecasting stack.
- Comparing a foundation model against honest naive baselines.
- A demonstration of the engineering pipeline (ingestion → validation →
  forecasting → backtesting → dashboard), **not** a trading signal.

## Not investment advice

Nothing here is investment advice. Do not make financial decisions based on
this dashboard alone. Past backtest performance says almost nothing about
future accuracy, especially for models near chance level.
