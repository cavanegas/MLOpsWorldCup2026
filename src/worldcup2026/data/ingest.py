"""Download raw datasets from the public sources declared in the project brief.

Sources:
    * International Football Results (martj42) — 1872..today, >49k rows.
    * The Fjelstul World Cup Database — clean relational WC 1930..2022.

Both are static CSVs on GitHub raw, so ingestion is deterministic and offline
after the first run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import FJELSTUL_BASE_URL, FJELSTUL_FILES, MARTJ42_RESULTS_URL, PATHS
from ..logging_utils import get_logger

log = get_logger(__name__)

_TIMEOUT = 60


def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.info("cache hit: %s", dest.name)
        return dest
    log.info("downloading %s -> %s", url, dest.name)
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download_martj42_results(force: bool = False) -> Path:
    """Download the cumulative international results CSV."""
    dest = PATHS.raw / "martj42_results.csv"
    return _download(MARTJ42_RESULTS_URL, dest, force=force)


def download_fjelstul(force: bool = False) -> dict[str, Path]:
    """Download the subset of Fjelstul's WorldCup database we rely on."""
    out: dict[str, Path] = {}
    for name in FJELSTUL_FILES:
        out[name] = _download(FJELSTUL_BASE_URL + name, PATHS.raw / f"fjelstul_{name}", force)
    return out


def load_martj42() -> pd.DataFrame:
    """Return the international results table with parsed dtypes."""
    path = download_martj42_results()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"home_team": "home", "away_team": "away"})
    # Basic sanity — some historical rows have NaN goals when match was abandoned.
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    return df