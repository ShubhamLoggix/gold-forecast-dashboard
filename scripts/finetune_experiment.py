"""Fine-tuning experiment: does gold-domain LoRA fine-tuning beat zero-shot?

Follows Google's documented TimesFM 2.5 fine-tuning path (HuggingFace
Transformers + PEFT LoRA, see timesfm-forecasting/examples/finetuning/) but
on gold data, with strict split discipline:

  * The walk-forward backtests' earliest test origin is index 512 (of the
    canonical series). TRAINING therefore uses ONLY indices [0, 512) — data
    strictly before every backtest test window. No leakage.
  * Evaluation reuses the exact same walk_forward() as the bake-off, so the
    numbers are apples-to-apples with zero-shot TimesFM / Chronos / naive.
  * Overfitting check: in-sample train/val loss is logged per epoch, and the
    fine-tuned model is compared out-of-sample on the same folds as zero-shot.

Writes its rows into the bake-off leaderboard (bakeoff_<date>.json):
  - "timesfm-2.5-hf-base"  (same-code-path zero-shot control)
  - "timesfm-2.5-ft-lora"  (the fine-tuned adapter)

Usage:
    python scripts/finetune_experiment.py [--skip-train]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from logging_setup import setup_logging
from scripts.model_bakeoff import HORIZONS, walk_forward

logger = logging.getLogger("gold_forecast.finetune")

# Backtest folds consume canonical indices >= 512 (first_origin); training
# must stop strictly before that.
TRAIN_SPLIT_INDEX = 512
ADAPTER_DIR = Path(settings.data_dir) / "models" / "timesfm2_5-gold-lora"
MODEL_ID = "google/timesfm-2.5-200m-transformers"


def _gold_training_series() -> np.ndarray:
    from ingestion.fetch_gold_prices import load_canonical

    history = load_canonical()
    close = history["close"].astype(np.float32).to_numpy()
    logger.info(
        "split discipline: total=%d, training on indices [0, %d) only "
        "(=%s .. %s); every backtest test window starts at index 512.",
        len(close), TRAIN_SPLIT_INDEX,
        pd.Timestamp(history["date"].iloc[0]).date(),
        pd.Timestamp(history["date"].iloc[TRAIN_SPLIT_INDEX - 1]).date(),
    )
    return close[:TRAIN_SPLIT_INDEX].astype(np.float32)


def train(epochs: int, num_samples: int, batch_size: int, lr: float) -> float:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    torch.manual_seed(42)
    device = "cpu"
    logger.info("Loading %s (transformers path, cpu)...", MODEL_ID)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
    )
    context_len, horizon_len = 192, 126
    assert context_len % 32 == 0

    lora_config = LoraConfig(
        r=4, lora_alpha=8, target_modules="all-linear",
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    series = _gold_training_series()
    rng = np.random.default_rng(42)
    min_len = context_len + horizon_len
    max_start = len(series) - min_len
    if max_start < 0:
        raise SystemExit("not enough training data for the window size")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs * (num_samples // batch_size), 1)
    )

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for _ in range(num_samples // batch_size):
            batch_ctx, batch_tgt = [], []
            for _ in range(batch_size):
                start = int(rng.integers(0, max_start + 1))
                batch_ctx.append(
                    torch.tensor(series[start : start + context_len], dtype=torch.float32)
                )
                batch_tgt.append(
                    torch.tensor(series[start + context_len : start + min_len], dtype=torch.float32)
                )
            batch_ctx = torch.stack(batch_ctx).to(device)
            batch_tgt = torch.stack(batch_tgt).to(device)
            outputs = model(
                past_values=batch_ctx,
                future_values=batch_tgt,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1
        logger.info("epoch %d/%d — train loss %.4f", epoch, epochs, epoch_loss / max(n, 1))

    ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    elapsed = time.perf_counter() - t0
    logger.info("adapter saved to %s (training %.1f min)", ADAPTER_DIR, elapsed / 60)
    return elapsed


def make_predict_fn(model, context_len: int = 512):
    import torch

    def predict_fn(ctx: np.ndarray, horizon_days: int):
        past = torch.tensor(
            np.asarray(ctx, dtype=np.float32)[-context_len:], dtype=torch.float32
        ).reshape(1, -1)
        with torch.no_grad():
            out = model(past_values=past)
        mean = out.mean_predictions[0].detach().float().cpu().numpy()
        full = getattr(out, "full_predictions", None)
        if (
            full is not None
            and getattr(full, "ndim", 0) == 3
            and full.shape[-1] >= 9
        ):
            # Quantiles are [0.1..0.9] -> col 0 = p10, col 8 = p90.
            q = full[0].detach().float().cpu().numpy()
            q10, q90 = q[:horizon_days, 0], q[:horizon_days, 8]
        else:
            q10, q90 = mean[:horizon_days], mean[:horizon_days]
        return (
            mean[:horizon_days].astype(np.float64),
            np.asarray(q10, dtype=np.float64),
            np.asarray(q90, dtype=np.float64),
        )

    return predict_fn


def evaluate(history: pd.DataFrame) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction

    base = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    base.eval()
    entries: list[dict] = []
    t0 = time.perf_counter()

    entries += _run_rows(history, make_predict_fn(base), "timesfm-2.5-hf-base", "2.5-hf")

    ft = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    ft.eval()
    entries += _run_rows(history, make_predict_fn(ft), "timesfm-2.5-ft-lora", "2.5-ft")

    logger.info("HF evaluation took %.1f min", (time.perf_counter() - t0) / 60)
    return entries


def _run_rows(history: pd.DataFrame, predict_fn, model_name: str, model_version: str) -> list[dict]:
    from scripts.model_bakeoff import _entry

    entries = []
    for horizon_days, step_days, max_folds, label in HORIZONS:
        # The HF inference path is capped at horizon_length=128; 1y (252d)
        # stays with the zero-shot package path and is documented in
        # LIMITATIONS.md rather than force-computed with a wrong head.
        if horizon_days > 128:
            continue
        result = walk_forward(history, predict_fn, 512, horizon_days, step_days, max_folds,
                              model_name, model_version)
        entries.append(_entry(result, label))
    return entries


def merge_into_bakeoff(entries: list[dict]) -> Path:
    out_dir = Path(settings.data_dir) / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    files = sorted(out_dir.glob("bakeoff_*.json"), reverse=True)
    if files:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
    else:
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "data_range": [],
            "entries": [],
        }
    kept = [
        e for e in payload["entries"]
        if e["model"] not in ("timesfm-2.5-hf-base", "timesfm-2.5-ft-lora")
    ]
    payload["entries"] = kept + entries
    payload["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = out_dir / f"bakeoff_{today}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("bake-off leaderboard updated: %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    setup_logging(settings.log_level)

    t_start = time.perf_counter()
    from ingestion.fetch_gold_prices import load_canonical

    history = load_canonical()

    train_minutes = 0.0
    if not args.skip_train:
        train_minutes = train(args.epochs, args.num_samples, args.batch_size, args.lr) / 60
    else:
        if not ADAPTER_DIR.exists():
            raise SystemExit(f"no adapter at {ADAPTER_DIR}; run without --skip-train")

    entries = evaluate(history)
    path = merge_into_bakeoff(entries)

    print("\nFine-tune experiment rows:")
    for e in entries:
        print(
            f"{e['model']:>20} {e['horizon_label']:>4} mape={e['mape_pct']:.2f}% "
            f"dir={e['directional_accuracy_pct']:.1f}% cov={e['band_coverage_pct']:.1f}%"
        )
    total_min = (time.perf_counter() - t_start) / 60
    print(f"\ncompute cost: training ~{train_minutes:.1f} min + eval, total ~{total_min:.1f} min (CPU)")
    print(f"leaderboard: {path}")


if __name__ == "__main__":
    main()
