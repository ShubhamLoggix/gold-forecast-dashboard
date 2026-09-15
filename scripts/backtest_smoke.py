"""CI smoke: run the walk-forward backtest on a small synthetic fixture window.

Exercises the full pipeline (history prep -> fold loop -> TimesFM stub +
baselines -> metrics -> persistence) without downloading model weights or
hitting the network. Exits non-zero if anything throws or the results are not
finite. The fixture data is SYNTHETIC (clearly labeled), not real prices.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting.timesfm_service import GoldForecastService  # noqa: E402


def fixture_history(n: int = 220) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    rng = np.random.default_rng(7)
    close = 2000.0 + np.cumsum(rng.normal(0.02, 6.0, n))  # UNIT-TEST FIXTURE ONLY
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": close + rng.normal(0, 0.5, n),
            "high": close + np.abs(rng.normal(0, 1, n)),
            "low": close - np.abs(rng.normal(0, 1, n)),
            "close": close,
            "volume": np.zeros(n),
            "source": "fixture",
        }
    )


class StubTimesFM:
    """Deterministic stub matching the TimesFM 2.5 output contract."""

    def forecast(self, horizon, inputs):
        last = float(inputs[0][-1])
        point = np.array([last * (1 + 0.001 * (i + 1)) for i in range(horizon)])
        quant = np.zeros((1, horizon, 10))
        for i in range(horizon):
            band = last * 0.02 * (i + 1)
            quant[0, i, 1] = point[i] - band
            quant[0, i, 5] = point[i]
            quant[0, i, 9] = point[i] + band
            quant[0, i, 0] = point[i]
        return point[None, :], quant


def main() -> int:
    service = GoldForecastService(model_version="2.5")
    service._model = StubTimesFM()
    service._predict_fn = service._predict_2p5
    results = service.backtest(
        fixture_history(), horizon_days=5, step_days=25, max_folds=4
    )
    assert set(results) == {"timesfm-2.5", "naive-last-value", "sma-20"}, results.keys()
    for name, res in results.items():
        s = res.summary
        assert s["n_folds"] == 4, (name, s)
        for key in ("mape_pct", "rmse", "mae", "directional_accuracy_pct"):
            assert np.isfinite(s[key]) and s[key] >= 0, (name, key, s[key])
    with tempfile.TemporaryDirectory() as tmp:
        path = service.persist_backtest(results, data_dir=Path(tmp))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "model_drift" in payload
        assert set(payload["model_drift"]) == {
            "delta_pp", "timesfm_dir_acc_pct", "naive_dir_acc_pct", "underperforming",
        }
    print("backtest smoke OK:", {k: v.summary for k, v in results.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
