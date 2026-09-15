import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings  # noqa: E402


def make_history(
    n: int = 300,
    start: str = "2024-01-01",
    seed: int = 42,
    base: float = 2000.0,
    drift: float = 0.5,
    vol: float = 5.0,
) -> pd.DataFrame:
    """Synthetic gold-like daily series. UNIT TEST FIXTURE ONLY - not real prices."""
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(drift / 21, vol, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=n),
            "open": close + rng.normal(0, 0.5, n),
            "high": close + np.abs(rng.normal(0, 1, n)),
            "low": close - np.abs(rng.normal(0, 1, n)),
            "close": close,
            "volume": rng.integers(0, 100_000, n).astype(float),
            "source": "fixture",
        }
    )


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    """Point global settings at a tmp dir so tests never touch real data/."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return settings
