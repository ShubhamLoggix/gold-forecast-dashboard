"""Smoke-verify the TimesFM model and empirically calibrate the quantile layout.

The torch module exposes q=10 output columns while the checkpoint config lists
9 quantiles [0.1..0.9]. For iid Gaussian input the model's quantile head
converges to the theoretical quantiles of N(0,1), which lets us pin each
column to its nominal quantile level.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import timesfm
from timesfm import configs

# Theoretical N(0,1) quantile values (for calibration reporting only).
THEORY = {
    0.05: -1.6449,
    0.1: -1.2816,
    0.2: -0.8416,
    0.3: -0.5244,
    0.4: -0.2533,
    0.45: -0.1257,
    0.5: 0.0,
    0.55: 0.1257,
    0.6: 0.2533,
    0.7: 0.5244,
    0.8: 0.8416,
    0.9: 1.2816,
    0.95: 1.6449,
}


def main() -> None:
    t0 = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    print(f"model loaded in {time.time() - t0:.1f}s on {model.model.device}")

    fc = configs.ForecastConfig(
        max_context=512,
        max_horizon=16,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
        return_backcast=False,
    )
    model.compile(fc)

    rng = np.random.default_rng(0)
    inputs = [rng.normal(0.0, 1.0, size=256).astype(np.float32) for _ in range(64)]
    point, quant = model.forecast(16, inputs)
    print("point", point.shape, "quant", quant.shape)
    print("finite:", np.isfinite(quant).all())

    cols = quant[:, 0, :].mean(axis=0)
    print("\ncol means at step 0:", np.round(cols, 3))
    for c in range(quant.shape[-1]):
        q_nom, q_theory = min(THEORY.items(), key=lambda kv: abs(kv[1] - cols[c]))
        print(f"col {c}: nominal q={q_nom} theory={q_theory:+.4f} observed={cols[c]:+.4f}")

    diff = np.abs(point[:, 0] - quant[:, 0, 5]).max()
    print("\npoint vs col5 max diff:", diff)

    # Real-ish trend series sanity check.
    trend = (np.linspace(2000, 2100, 256) + rng.normal(0, 3, 256)).astype(np.float32)
    p, qd = model.forecast(8, [trend])
    print("\ntrend series point forecast:", np.round(p[0], 2))
    print("trend p10/p90:", np.round(qd[0, -1, [0, 9]], 2))
    sys.exit(0)


if __name__ == "__main__":
    main()
