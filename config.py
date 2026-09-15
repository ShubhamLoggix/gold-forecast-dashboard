"""Central configuration loaded from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv()


def _fallback_hf_cache() -> None:
    """Point the HuggingFace cache at a larger drive when the system drive is full.

    Weights for TimesFM 2.5/3.0 are ~1 GB; if HF_HOME is unset and the volume
    holding the default cache has < 2 GB free, redirect to D:\\hf_cache (if D:
    exists). Override by setting HF_HOME explicitly.
    """
    if "HF_HOME" in os.environ:
        return
    try:
        import shutil

        home_drive = Path.home().drive or "C:"
        if shutil.disk_usage(f"{home_drive}\\").free < 2 * 1024**3 and Path("D:\\").exists():
            os.environ["HF_HOME"] = "D:\\hf_cache"
    except Exception:  # noqa: BLE001 - best-effort convenience only
        pass


_fallback_hf_cache()


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    model_version: str = field(
        default_factory=lambda: os.environ.get("MODEL_VERSION", "2.5").strip()
    )
    non_commercial: bool = field(
        default_factory=lambda: _env_bool("NON_COMMERCIAL", "false")
    )
    context_length: int = field(
        default_factory=lambda: int(os.environ.get("CONTEXT_LENGTH", "512"))
    )
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "auto"))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")))
    metals_api_key: str = field(
        default_factory=lambda: os.environ.get("METALS_API_KEY", "").strip()
    )
    history_start: str = field(
        default_factory=lambda: os.environ.get("HISTORY_START", "2016-01-01")
    )
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", ""))
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip()
        for o in os.environ.get(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if o.strip()
    ])
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    enable_scheduler: bool = field(
        default_factory=lambda: _env_bool("ENABLE_SCHEDULER", "1")
    )
    torch_compile: bool = field(
        default_factory=lambda: _env_bool("TIMESFM_TORCH_COMPILE", "0")
    )
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    )
    telegram_chat_ids: list[str] = field(default_factory=lambda: [
        c.strip()
        for c in os.environ.get("TELEGRAM_CHAT_IDS", "").split(",")
        if c.strip()
    ])
    dashboard_url: str = field(
        default_factory=lambda: os.environ.get("DASHBOARD_URL", "").strip()
    )
    digest_city: str = field(
        default_factory=lambda: os.environ.get("DIGEST_CITY", "pune").strip()
    )

    @property
    def alerts_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_ids)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def backtest_dir(self) -> Path:
        return self.data_dir / "backtests"


settings = Settings()
