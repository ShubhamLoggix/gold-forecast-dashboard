"""TimesFM-backed gold price forecasting service.

Loads TimesFM once (model object cached per (version, context, device) key),
exposes quantile forecasts, and implements a walk-forward backtest against
naive baselines so the dashboard can show *measured* historical accuracy
instead of claiming the model works.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from forecasting.baselines import naive_last_value, simple_moving_average
from forecasting.metrics import (
    directional_accuracy,
    mae,
    mape,
    quantile_spread,
    rmse,
)

logger = logging.getLogger("gold_forecast.forecasting")

# Trading-day horizons exposed by the dashboard (1w/1m/3m/6m/1y).
HORIZON_PRESETS: dict[str, int] = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

MAX_HORIZON = 256  # compile ceiling (multiple of TimesFM output patch size 128)

# Empirically calibrated TimesFM 2.5 output layout (scripts/verify_model.py):
# col 0 = point/median channel, cols 1..9 = quantiles 0.1..0.9, and the
# model's returned point forecast equals col 5 (the 0.5 quantile).
Q10_COL, Q50_COL, Q90_COL = 1, 5, 9

CONTEXT_MIN = 32  # one input patch


@dataclasses.dataclass
class ForecastResult:
    """Point forecast + quantile bands for `horizon_days` trading days ahead."""

    dates: list[dt.date]
    point: np.ndarray  # (horizon,) median forecast
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    horizon_days: int
    model_version: str
    context_used: int
    latency_ms: float
    history_last_date: dt.date
    last_close: float

    def band_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(self.dates),
                "point": self.point,
                "q10": self.q10,
                "q50": self.q50,
                "q90": self.q90,
            }
        )


@dataclasses.dataclass
class BacktestFold:
    origin_date: str
    horizon_days: int
    mape: float
    rmse: float
    mae: float
    directional_accuracy: float
    band_coverage: float = 0.0


@dataclasses.dataclass
class BacktestResult:
    model_name: str
    model_version: str
    horizon_days: int
    step_days: int
    folds: list[BacktestFold]
    summary: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "horizon_days": self.horizon_days,
            "step_days": self.step_days,
            "summary": self.summary,
            "folds": [dataclasses.asdict(f) for f in self.folds],
        }


_MODEL_CACHE: dict[tuple, object] = {}
_MODEL_LOCK = threading.Lock()


class TimesFMNotLoaded(RuntimeError):
    pass


class GoldForecastService:
    """TimesFM wrapper for gold price forecasting (one model load per config)."""

    def __init__(
        self,
        model_version: str = "2.5",
        context_length: int = 512,
        device: str = "auto",
    ):
        if model_version not in {"2.5", "3.0"}:
            raise ValueError("model_version must be '2.5' or '3.0'")
        if context_length < CONTEXT_MIN:
            raise ValueError(f"context_length must be >= {CONTEXT_MIN}")
        self.model_version = model_version
        self.context_length = context_length
        self.device = device
        self._model: object | None = None
        self._predict_fn = None

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Load + compile TimesFM once; subsequent calls are no-ops (thread-safe)."""
        if self._model is not None:
            return
        with _MODEL_LOCK:
            if self._model is not None:
                return
            key = (self.model_version, self.context_length, self.device)
            model = _MODEL_CACHE.get(key)
            if model is None:
                model = self._load_model_uncached()
                _MODEL_CACHE[key] = model
            self._model = model

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover - torch missing
            pass
        return "cpu"

    def _load_model_uncached(self):
        if self.model_version == "3.0" and not settings.non_commercial:
            raise PermissionError(
                "TimesFM 3.0 pretrained weights are licensed under the non-commercial "
                "license (timesfm-non-commercial-license-v1.0). Set NON_COMMERCIAL=true "
                "if you accept those terms, or keep MODEL_VERSION=2.5 (Apache-2.0)."
            )
        if self.model_version == "3.0":
            logger.warning(
                "MODEL_VERSION=3.0 selected with NON_COMMERCIAL=true: TimesFM 3.0 weights "
                "may NOT be used for commercial purposes (non-commercial license v1.0)."
            )
        t0 = time.perf_counter()
        resolved = self._resolve_device()
        logger.info("Loading TimesFM %s (%s)...", self.model_version, resolved)
        import timesfm

        if self.model_version == "2.5":
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch",
                torch_compile=settings.torch_compile,
            )
            model.compile(
                timesfm.configs.ForecastConfig(
                    max_context=self.context_length,
                    max_horizon=MAX_HORIZON,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                    return_backcast=False,
                )
            )
            self._predict_fn = self._predict_2p5
        else:
            forecaster = timesfm.TimesFM3Forecaster.from_pretrained(
                "google/timesfm-3.0-pytorch",
                device=None if resolved == "auto" else resolved,
            )
            model = forecaster
            self._predict_fn = self._predict_3p0
        elapsed = time.perf_counter() - t0
        logger.info(
            "TimesFM %s loaded on %s in %.1fs (torch_compile=%s).",
            self.model_version,
            resolved,
            elapsed,
            settings.torch_compile,
        )
        return model

    # ------------------------------------------------------------------ #
    # Forecast
    # ------------------------------------------------------------------ #
    def forecast(
        self,
        history: pd.DataFrame,
        horizon_days: int,
        quantiles: bool = True,
    ) -> ForecastResult:
        """Point forecast + quantile bands for `horizon_days` trading days ahead."""
        values = self._validated_values(history)
        ctx = values[-self.context_length :]
        if self._predict_fn is None:
            raise TimesFMNotLoaded("Call load_model() before forecast().")
        t0 = time.perf_counter()
        point, q10, q90 = self._predict_fn(ctx.astype(np.float32), horizon_days)
        latency = (time.perf_counter() - t0) * 1000
        q50 = point
        if not quantiles:
            q10, q90 = q50, q50
        dates = self._future_dates(history, horizon_days)
        result = ForecastResult(
            dates=dates,
            point=point,
            q10=q10,
            q50=q50,
            q90=q90,
            horizon_days=horizon_days,
            model_version=self.model_version,
            context_used=len(ctx),
            latency_ms=latency,
            history_last_date=pd.Timestamp(history["date"].iloc[-1]).date(),
            last_close=float(values[-1]),
        )
        logger.info(
            "forecast request: range=[%s, %s] n=%d horizon=%d model=%s "
            "latency=%.1fms spread(p90-p10)=%.2f",
            pd.Timestamp(history["date"].iloc[0]).date(),
            pd.Timestamp(history["date"].iloc[-1]).date(),
            len(history),
            horizon_days,
            self.model_version,
            latency,
            quantile_spread(q10, q90),
        )
        return result

    def _validated_values(self, history: pd.DataFrame) -> np.ndarray:
        if history is None or len(history) == 0:
            raise ValueError("history is empty")
        if not {"date", "close"}.issubset(history.columns):
            raise ValueError("history must have 'date' and 'close' columns")
        if not history["date"].is_monotonic_increasing:
            raise ValueError("history dates must be sorted ascending")
        if history["close"].isna().any():
            raise ValueError("history contains NaN close values")
        if (history["close"] <= 0).any():
            raise ValueError("history contains non-positive close values")
        values = history["close"].astype(np.float64).to_numpy()
        if len(values) < CONTEXT_MIN:
            raise ValueError(f"history must contain at least {CONTEXT_MIN} observations")
        return values

    @staticmethod
    def _future_dates(history: pd.DataFrame, horizon_days: int) -> list[dt.date]:
        last_date = pd.Timestamp(history["date"].iloc[-1])
        future = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon_days)
        return [d.date() for d in future]

    # ----- per-version inference adapters -----
    def _predict_2p5(
        self, values: np.ndarray, horizon_days: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if horizon_days > MAX_HORIZON:
            raise ValueError(f"horizon_days must be <= {MAX_HORIZON}")
        point, quant = self._model.forecast(horizon_days, [values])
        return (
            np.asarray(point[0], dtype=np.float64),
            np.asarray(quant[0, :, Q10_COL], dtype=np.float64),
            np.asarray(quant[0, :, Q90_COL], dtype=np.float64),
        )

    def _predict_3p0(
        self, values: np.ndarray, horizon_days: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self._model.predict(
            values, horizon_days, return_quantiles=True, make_positive=False
        )
        point = np.asarray(out.point, dtype=np.float64).ravel()[:horizon_days]
        quantiles = np.asarray(out.quantiles, dtype=np.float64)
        levels = getattr(self._model.config, "quantiles", [])
        if 0.1 in levels and 0.9 in levels:
            i10, i90 = levels.index(0.1), levels.index(0.9)
        else:  # fall back to symmetric outer bands
            i10, i90 = 0, quantiles.shape[-1] - 1
        q10 = quantiles[..., i10].ravel()[:horizon_days]
        q90 = quantiles[..., i90].ravel()[:horizon_days]
        return point, q10, q90

    # ------------------------------------------------------------------ #
    # Backtest
    # ------------------------------------------------------------------ #
    def backtest(
        self,
        history: pd.DataFrame,
        window: Literal["walk-forward"] = "walk-forward",
        horizon_days: int = 30,
        step_days: int = 7,
        max_folds: int = 52,
    ) -> dict[str, BacktestResult]:
        """Rolling-origin backtest: TimesFM vs naive last-value vs SMA(20).

        Returns {model_name: BacktestResult}. Persist with `persist_backtest`.
        """
        if window != "walk-forward":
            raise ValueError("Only 'walk-forward' backtesting is implemented")
        if self._predict_fn is None:
            raise TimesFMNotLoaded("Call load_model() before backtest().")
        values = self._validated_values(history)
        dates = pd.to_datetime(history["date"]).reset_index(drop=True)
        close = values.astype(np.float64)
        n = len(close)
        if horizon_days > MAX_HORIZON:
            raise ValueError(f"horizon_days must be <= {MAX_HORIZON}")

        first_origin = max(CONTEXT_MIN, min(self.context_length, n // 2))
        origins = list(range(first_origin, n - horizon_days + 1, step_days))
        if len(origins) > max_folds:
            stride = len(origins) // max_folds or 1
            origins = origins[::stride][:max_folds]
        if not origins:
            raise ValueError(
                "Not enough history for a backtest: need context + horizon + 1 fold."
            )

        model_name = f"timesfm-{self.model_version}"
        tm_folds: list[BacktestFold] = []
        nv_folds: list[BacktestFold] = []
        sma_folds: list[BacktestFold] = []
        t0 = time.perf_counter()
        for origin in origins:
            ctx = close[max(0, origin - self.context_length) : origin]
            actual = close[origin : origin + horizon_days]
            if len(actual) < horizon_days:
                continue
            point, q10, q90 = self._predict_fn(ctx.astype(np.float32), horizon_days)
            tm_folds.append(
                _fold_metrics(dates, origin, horizon_days, close[origin - 1], point, actual, q10, q90)
            )
            nv = naive_last_value(history.iloc[:origin], horizon_days)
            nv_folds.append(
                _fold_metrics(dates, origin, horizon_days, close[origin - 1], nv, actual, nv, nv)
            )
            sma = simple_moving_average(history.iloc[:origin], horizon_days)
            sma_folds.append(
                _fold_metrics(dates, origin, horizon_days, close[origin - 1], sma, actual, sma, sma)
            )
        elapsed = time.perf_counter() - t0

        result = {
            model_name: _summarize(
                model_name, self.model_version, horizon_days, step_days, tm_folds
            ),
            "naive-last-value": _summarize(
                "naive-last-value", "-", horizon_days, step_days, nv_folds
            ),
            "sma-20": _summarize("sma-20", "-", horizon_days, step_days, sma_folds),
        }
        drift = evaluate_drift(result)
        if drift and drift["underperforming"]:
            logger.warning(
                "MODEL UNDERPERFORMING BASELINE: TimesFM directional accuracy is "
                "%.2f pp below the naive carry-forward baseline (%s vs %s).",
                -drift["delta_pp"],
                drift["timesfm_dir_acc_pct"],
                drift["naive_dir_acc_pct"],
            )
        logger.info(
            "backtest done: folds=%d horizon=%d elapsed=%.1fs | %s",
            len(tm_folds),
            horizon_days,
            elapsed,
            {k: v.summary for k, v in result.items()},
        )
        return result

    def persist_backtest(
        self,
        results: dict[str, BacktestResult],
        data_dir: Path | None = None,
        suffix: str = "",
    ) -> Path:
        """Save backtest results to data/backtests/backtest_<date>[<suffix>].json."""
        data_dir = Path(data_dir or settings.data_dir)
        out_dir = data_dir / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"backtest_{dt.date.today().isoformat()}{suffix}.json"
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_version": self.model_version,
            "context_length": self.context_length,
            "model_drift": evaluate_drift(results),
            "results": {name: res.to_dict() for name, res in results.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Backtest report written to %s", path)
        return path


MODEL_DRIFT_THRESHOLD_PP = -5.0  # TimesFM < naive by more than 5 percentage points


def evaluate_drift(results: dict[str, "BacktestResult"]) -> dict | None:
    """Model-drift watchdog: TimesFM directional accuracy vs naive baseline.

    Returns None when either summary is missing; otherwise
    {"delta_pp", "timesfm_dir_acc_pct", "naive_dir_acc_pct", "underperforming"}
    where underperforming is True when TimesFM falls more than 5 pp below the
    naive carry-forward baseline.
    """
    tm = next(
        (r for name, r in results.items() if name.startswith("timesfm-")), None
    )
    nv = results.get("naive-last-value")
    if (
        tm is None
        or nv is None
        or not tm.summary
        or not nv.summary
        or tm.summary.get("directional_accuracy_pct") is None
        or nv.summary.get("directional_accuracy_pct") is None
    ):
        return None
    tm_acc = float(tm.summary["directional_accuracy_pct"])
    nv_acc = float(nv.summary["directional_accuracy_pct"])
    delta = tm_acc - nv_acc
    return {
        "delta_pp": round(delta, 2),
        "timesfm_dir_acc_pct": round(tm_acc, 2),
        "naive_dir_acc_pct": round(nv_acc, 2),
        "underperforming": bool(delta < MODEL_DRIFT_THRESHOLD_PP),
    }


def _fold_metrics(
    dates: pd.Series,
    origin: int,
    horizon_days: int,
    prev_close: float,
    predicted: np.ndarray,
    actual: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> BacktestFold:
    # Band coverage: fraction of actual outcomes that landed inside the
    # model's own p10–p90 band (≈80% when the band is well calibrated).
    coverage = float(np.mean((actual >= q10) & (actual <= q90))) if len(actual) else 0.0
    return BacktestFold(
        origin_date=str(pd.Timestamp(dates.iloc[origin - 1]).date()),
        horizon_days=horizon_days,
        mape=mape(actual, predicted),
        rmse=rmse(actual, predicted),
        mae=mae(actual, predicted),
        directional_accuracy=directional_accuracy(
            np.array([prev_close]), actual, predicted
        ),
        band_coverage=coverage,
    )


def _summarize(
    name: str, version: str, horizon_days: int, step_days: int, folds: list[BacktestFold]
) -> BacktestResult:
    if not folds:
        return BacktestResult(name, version, horizon_days, step_days, folds, {})
    summary = {
        "mape_pct": float(np.mean([f.mape for f in folds])),
        "rmse": float(np.mean([f.rmse for f in folds])),
        "mae": float(np.mean([f.mae for f in folds])),
        "directional_accuracy_pct": float(
            np.mean([f.directional_accuracy for f in folds]) * 100
        ),
        "band_coverage_pct": float(
            np.mean([f.band_coverage for f in folds]) * 100
        ),
        "n_folds": len(folds),
    }
    return BacktestResult(name, version, horizon_days, step_days, folds, summary)
