"""Amazon Chronos zero-shot forecaster (challenger for the model bake-off).

Same call shape as the TimesFM wrapper's `_predict_fn`: context closes in,
(point, q10, q90) out. Uses amazon/chronos-t5-tiny by default — small enough
for CPU inference while remaining a genuine zero-shot TSFM baseline.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger("gold_forecast.chronos")

CHRONOS_MODEL_ID = "amazon/chronos-t5-tiny"


class ChronosForecastService:
    def __init__(
        self,
        model_id: str = CHRONOS_MODEL_ID,
        context_length: int = 512,
        num_samples: int = 20,
    ):
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self._pipeline = None
        self.is_loaded = False

    def load_model(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from chronos import ChronosPipeline

        logger.info("Loading Chronos %s on cpu...", self.model_id)
        t0 = time.perf_counter()
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.is_loaded = True
        logger.info(
            "Chronos %s loaded in %.1fs", self.model_id, time.perf_counter() - t0
        )

    def forecast_points(
        self, values: np.ndarray, horizon_days: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(point=median, q10, q90) plus latency in ms, from raw close values."""
        if self._pipeline is None:
            raise RuntimeError("Call load_model() before forecast_points().")
        import torch

        context = torch.from_numpy(
            np.asarray(values, dtype=np.float32)[-self.context_length :]
        ).reshape(1, -1)
        t0 = time.perf_counter()
        out = self._pipeline.predict(
            context,
            prediction_length=horizon_days,
            num_samples=self.num_samples,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        arr = out.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[1] == self.num_samples:
            # Sample layout [batch, samples, horizon]
            draw = arr[0]
        elif arr.ndim == 3:
            # Quantile layout [batch, horizon, quantiles]
            draw = np.transpose(arr[0], (1, 0))
        else:
            raise RuntimeError(f"Unexpected Chronos output shape {arr.shape}")
        point = np.quantile(draw, 0.5, axis=0).astype(np.float64)
        q10 = np.quantile(draw, 0.1, axis=0).astype(np.float64)
        q90 = np.quantile(draw, 0.9, axis=0).astype(np.float64)
        return point, q10, q90, latency_ms

    def predict(
        self, ctx: np.ndarray, horizon_days: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TimesFM-compatible signature: context values -> (point, q10, q90)."""
        point, q10, q90, _ = self.forecast_points(ctx, horizon_days)
        return point, q10, q90
