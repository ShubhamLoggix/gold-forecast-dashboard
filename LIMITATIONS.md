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

## COMEX vs Indian retail prices (INR/karat view)

The INR view converts COMEX USD/futures prices to INR/gram (or per 10 g) at 24K/22K/18K purity
using the USD/INR exchange rate. **This is a theoretical bullion-equivalent price, not an Indian
retail/jeweler quote.** Real Indian retail prices sit noticeably higher because they stack on top
of the international price:

1. **Import duty** (set by the Government of India, changes over time),
2. **GST** (3% on gold; making-charges GST applies as well),
3. **Dealer/jeweler making charges** (highly variable, often 5–20%+ of value on jewelry).

So a converted figure of, say, ₹1,22,500 per 10 g (22K) can coexist with an actual jeweler price
well above ₹1,30,000 for the same weight. The dashboard labels the converted series as
bullion-equivalent everywhere (persistent caption in INR mode, API `RateInfo.disclaimer`), and
shows the exact rate and its date used for the conversion. Do not use the converted figure as an
arbitrage or "fair price" reference against retail quotes.

**Real retail quotes** are available separately: the INR-mode `Retail (Groww)` source serves
published city-wise retail rates (24K/22K/18K) scraped from Groww's gold-rates pages. Limitations
of that source: it is a third-party website (format/layout changes can break ingestion), it
provides only ~10 days of look-back per city (longer history accumulates locally day by day),
rates are updated by Groww on their own schedule (not real-time tick data), and TimesFM
forecasts are still computed on the COMEX bullion series — the retail-tab forecast is an
**estimate** (bullion forecast × the city's current retail premium ratio), since only the
COMEX series has enough history to forecast directly.

## Fine-tuning experiment (negative result — documented deliberately)

Following Google's documented TimesFM 2.5 fine-tuning path (HuggingFace
Transformers + PEFT LoRA, `timesfm-forecasting/examples/finetuning/`), the model
was LoRA-fine-tuned (r=4, all-linear, ~0.6% params) on gold data with strict
split discipline: training used ONLY the first 512 daily points (2021-09-15 →
2023-09-27), strictly before every walk-forward test window, then evaluated on
the identical folds as the zero-shot bake-off.

**Result: fine-tuning made out-of-sample performance WORSE at every horizon.**

| Horizon | zero-shot (same HF code path) | fine-tuned LoRA |
|---|---|---|
| 1w MAPE | 1.59% | 1.95% |
| 1m MAPE | 3.09% | 3.88% |
| 3m MAPE | 5.77% | 7.22% |
| 6m MAPE | 10.17% | 12.73% |
| band coverage (1w) | 40.0% | 13.5% |

In-sample training loss fell 1.36 → 0.13 while out-of-sample degraded — the
classic overfitting signature on ~500 training points, exactly the failure mode
expected for daily financial data. The fine-tuned bands are also badly
under-covering (13–19% vs 40% zero-shot). No leakage: the training window ends
months before the first evaluation fold.

Scope note: the HF inference path caps forecasts at 128 steps, so the
fine-tuned comparison covers 1w–6m only; the 1y row is served by the zero-shot
package path (which supports 256).

Compute cost (CPU only, no GPU available): training 4 epochs / 400 windows /
batch 4 took **107 min**; evaluation (both models × 4 horizons × folds) took
37 min; ~2.5 h total. Any serious fine-tuning attempt needs GPU time.

**Conclusion: keep the zero-shot models.** The negative result is valuable:
it demonstrates that "fine-tune it" is not an automatic accuracy win for small
financial datasets, contrary to common hype.

## Model-drift watchdog

The project includes an automatic honesty check: after every backtest run
(daily/weekly scheduled runs and `scripts/run_backtest.py`), TimesFM's
directional accuracy is compared against the naive carry-forward baseline. If
TimesFM falls **more than 5 percentage points below** the naive baseline, the
run logs a `WARNING`, `/api/v1/health` reports
`model_underperforming_baseline: true` (status becomes `degraded`), and the
dashboard's model-info panel shows a prominent warning. Given that TimesFM's
directional accuracy hovers near 50% (chance), this flag will appear and
disappear between backtest windows — that is expected behavior and is precisely
the transparency the dashboard is designed to provide. It is not a bug fix
signal; it is a reminder that the model has no demonstrated edge over "do
nothing".

## Not investment advice

Nothing here is investment advice. Do not make financial decisions based on
this dashboard alone. Past backtest performance says almost nothing about
future accuracy, especially for models near chance level.
