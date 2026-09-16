"""Telegram alerts: daily digest + one-shot price-target notifications.

Config (see .env.example):
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_IDS   — comma-separated chat ids (family groups)
  DASHBOARD_URL       — optional link appended to messages

Targets are stored in data/alerts/targets.json; a triggered target is marked
and never fires twice.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import uuid
from pathlib import Path

import pandas as pd

from config import settings

logger = logging.getLogger("gold_forecast.alerts")

_TARGETS_FILENAME = "targets.json"
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #
def send_message(text: str) -> bool:
    """Post to every configured chat. Returns True if at least one send OK."""
    if not settings.alerts_enabled:
        logger.info("alerts disabled (TELEGRAM_BOT_TOKEN/CHAT_IDS not set).")
        return False
    import requests

    ok = False
    for chat_id in settings.telegram_chat_ids:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                ok = True
            else:
                logger.warning("telegram send to %s failed: HTTP %s %s",
                               chat_id, resp.status_code, resp.text[:200])
        except Exception as exc:  # noqa: BLE001 - alerts must never crash jobs
            logger.warning("telegram send to %s failed: %s", chat_id, exc)
    return ok


# --------------------------------------------------------------------------- #
# Target storage
# --------------------------------------------------------------------------- #
def _targets_path(data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir or settings.data_dir)
    return data_dir / "alerts" / _TARGETS_FILENAME


def list_targets(data_dir: Path | None = None) -> list[dict]:
    path = _targets_path(data_dir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupt file -> start clean
        return []


def save_targets(targets: list[dict], data_dir: Path | None = None) -> None:
    path = _targets_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(json.dumps(targets, indent=2), encoding="utf-8")


def add_target(
    karat: str, unit: str, currency: str, op: str, price: float,
    data_dir: Path | None = None,
) -> dict:
    targets = list_targets(data_dir)
    target = {
        "id": uuid.uuid4().hex[:10],
        "karat": karat,
        "unit": unit,
        "currency": currency,
        "op": op,
        "price": float(price),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "triggered_at": None,
    }
    targets.append(target)
    save_targets(targets, data_dir)
    return target


def remove_target(target_id: str, data_dir: Path | None = None) -> bool:
    targets = list_targets(data_dir)
    remaining = [t for t in targets if t.get("id") != target_id]
    if len(remaining) == len(targets):
        return False
    save_targets(remaining, data_dir)
    return True


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _current_value(target: dict, history: pd.DataFrame, usd_inr_rate: float) -> float:
    """The price an alert watches. INR targets watch the Groww RETAIL quote for
    the digest city (what you'd actually pay in a shop); falls back to the
    bullion-equivalent conversion if the retail source is unavailable."""
    close = float(history["close"].iloc[-1])
    if target["currency"] == "usd":
        return close  # native series is USD per troy oz
    unit_factor = 10 if target["unit"] == "10gram" else 1
    try:
        from ingestion.india_rates import get_city_rates

        retail = get_city_rates(settings.digest_city)
        return float(retail["per_gram"][target["karat"]]) * unit_factor
    except Exception:  # noqa: BLE001 - retail unavailable -> bullion fallback
        from conversion import convert_series

        return float(convert_series(close, usd_inr_rate, target["karat"], target["unit"]))


def evaluate_targets(history: pd.DataFrame, usd_inr_rate: float, data_dir: Path | None = None) -> list[dict]:
    """Check active targets against the latest close; notify on crossings.

    Returns the list of newly triggered targets. Triggered ones are marked
    with triggered_at and never fire again.
    """
    targets = list_targets(data_dir)
    if not targets:
        return []
    triggered_now: list[dict] = []
    for target in targets:
        if target.get("triggered_at"):
            continue
        try:
            value = _current_value(target, history, usd_inr_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alert eval failed for %s: %s", target.get("id"), exc)
            continue
        hit = (
            value <= target["price"]
            if target["op"] == "<="
            else value >= target["price"]
        )
        if hit:
            target["triggered_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            target["triggered_value"] = round(value, 2)
            triggered_now.append(target)
    if triggered_now:
        save_targets(targets, data_dir)
        for target in triggered_now:
            send_message(_target_message(target))
    return triggered_now


def _target_message(target: dict) -> str:
    side = "dropped to" if target["op"] == "<=" else "rose to"
    if target["currency"] == "inr":
        karat = f"{target['karat'].upper()} retail ({settings.digest_city.title()})"
        unit = f"per {target['unit'].replace('gram', 'g')}"
        amount = f"Rs {_inr(target['triggered_value'])}"
    else:
        karat = "COMEX GC=F"
        unit = "per oz"
        amount = f"${target['triggered_value']:,.2f}"
    return (
        f"GOLD ALERT\n"
        f"{karat} {side} {amount} {unit}\n"
        f"(you asked for {'<=' if target['op'] == '<=' else '>='} "
        + (_inr(target["price"]) if target["currency"] == "inr" else f"{target['price']:,.2f}")
        + ")\n"
        + (f"\nDashboard: {settings.dashboard_url}" if settings.dashboard_url else "")
    )


def _inr(value: float) -> str:
    """Indian digit grouping: 1,53,170 (lakh/crore style)."""
    s = f"{float(value):,.0f}".replace(",", "")
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = [head[::-1][i:i + 2] for i in range(0, len(head), 2)]
    return ",".join(g[::-1] for g in reversed(groups)) + "," + tail


def _fmt_pct(value) -> str:
    if value is None:
        return ""
    return f"{value:+.2f}%"


def daily_digest(
    retail: dict | None,
    usd_inr_rate: float | None,
    usd_inr_rate_date,
    forecast_info: dict | None,
    drift: dict | None = None,
    active_targets: int = 0,
) -> bool:
    """Build + send the daily digest. Returns True if sent.

    Template is retail-first: Groww city rates in INR (24K/22K/18K, per 10g and
    per gram), the estimated retail forecast, and bullion only as a reference.
    """
    city = (retail or {}).get("city") or "Pune"
    lines = [f"GOLD DAILY — {city} retail (Groww)"]
    if retail:
        lines.append(f"Rate date: {retail.get('date')}")
        lines.append("")
        per_gram = retail.get("per_gram") or {}
        per_10g = retail.get("per_10g") or {
            k: float(v) * 10 for k, v in per_gram.items()
        }
        pct = retail.get("pct_change") or {}
        lines.append(
            f"24K/10g: Rs {_inr(per_10g.get('24k', 0))}  ({_fmt_pct(pct.get('24k'))} vs prev day)"
        )
        lines.append(
            f"22K/10g: Rs {_inr(per_10g.get('22k', 0))}  ({_fmt_pct(pct.get('22k'))} vs prev day)"
        )
        lines.append(
            f"18K/10g: Rs {_inr(per_10g.get('18k', 0))}  ({_fmt_pct(pct.get('18k'))} vs prev day)"
        )
        lines.append("")
        lines.append(
            "Per gram: 24K Rs {} | 22K Rs {} | 18K Rs {}".format(
                _inr(per_gram.get("24k", 0)), _inr(per_gram.get("22k", 0)), _inr(per_gram.get("18k", 0))
            )
        )
    if usd_inr_rate:
        lines.append(f"USD/INR: {usd_inr_rate:.4f} as of {usd_inr_rate_date}")
    if forecast_info:
        lines.append("")
        lines.append(
            "Forecast {} (est. retail, 22K/10g): Rs {} ({} est.)".format(
                forecast_info.get("horizon", "1m"),
                _inr(forecast_info.get("median_inr_22k_10g", 0)),
                _fmt_pct(forecast_info.get("pct")),
            )
        )
        lines.append(
            "(estimate = TimesFM bullion forecast x current retail premium"
            + f" {forecast_info.get('premium_ratio', 0):.3f}"
            + ")"
        )
    if active_targets:
        lines.append(f"Active price alerts: {active_targets}")
    if drift and drift.get("underperforming"):
        lines.append("")
        lines.append(
            "NOTE: model currently underperforming naive baseline — "
            "treat forecasts with extra skepticism"
        )
    lines.append("")
    lines.append(
        "Retail quotes include import duty + GST + local premium and move during "
        "the day; confirm with your jeweller before buying."
    )
    if settings.dashboard_url:
        lines.append(f"Dashboard: {settings.dashboard_url}")
    lines.append("Not investment advice — forecasts are experimental.")
    return send_message("\n".join(lines))
