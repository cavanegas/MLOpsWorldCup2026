"""Normalise and harmonise the raw match history into a single long table.

Output schema (one row per match):
    date, home, away, home_score, away_score, neutral, tournament,
    is_worldcup, is_knockout, country
"""

from __future__ import annotations

import pandas as pd

from ..config import PATHS
from ..logging_utils import get_logger
from .ingest import load_martj42

log = get_logger(__name__)

# Country name aliases so Fjelstul, martj42, and FIFA strings align.
_COUNTRY_ALIASES = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "Czechia": "Czech Republic",
    "Republic of Ireland": "Ireland",
    "DR Congo": "Democratic Republic of the Congo",
}

_WORLD_CUP_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
}

_KNOCKOUT_HINTS = (
    "World Cup",
    "Copa América",
    "UEFA Euro",
    "African Cup of Nations",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
)


def _canonical(name: str) -> str:
    return _COUNTRY_ALIASES.get(name, name)


def build_match_history() -> pd.DataFrame:
    """Return the cleaned long history used for Elo + feature engineering."""
    df = load_martj42().copy()
    df["home"] = df["home"].map(_canonical)
    df["away"] = df["away"].map(_canonical)

    df["is_worldcup"] = df["tournament"].isin(_WORLD_CUP_TOURNAMENTS)
    df["is_knockout"] = df["tournament"].apply(
        lambda t: any(h in t for h in _KNOCKOUT_HINTS)
    )
    df = df.sort_values("date").reset_index(drop=True)

    out_path = PATHS.processed / "match_history.parquet"
    df.to_parquet(out_path, index=False)
    log.info("wrote %s rows -> %s", len(df), out_path)
    return df
