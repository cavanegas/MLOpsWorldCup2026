"""Rolling Elo rating builder.

We implement an eloratings.net-style algorithm:

    R'_home = R_home + K * g(d) * (W - W_e)

where ``W`` is the actual result (1/0.5/0), ``W_e`` the expected result from the
logistic of the rating differential (with home advantage), ``g(d)`` the goal
difference multiplier, and K is weighted up for World Cup / knockout matches.

Ratings are replayed forward over the full international history so that every
match has an *ex-ante* pair (home_elo_before, away_elo_before) — which is what
the model trains on.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import ELO_HOME_ADV, ELO_INITIAL, ELO_K, ELO_WC_WEIGHT, PATHS
from ..logging_utils import get_logger

log = get_logger(__name__)


def _goal_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def _match_weight(row: pd.Series) -> float:
    if row["is_worldcup"]:
        return ELO_K + ELO_WC_WEIGHT
    if row["is_knockout"]:
        return ELO_K + ELO_WC_WEIGHT / 2
    if row["tournament"] == "Friendly":
        return ELO_K * 0.6
    return ELO_K


def compute_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Return *matches* augmented with pre-match Elo ratings for both sides."""
    required = {"date", "home", "away", "home_score", "away_score", "neutral",
                "tournament", "is_worldcup", "is_knockout"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"match history is missing columns: {sorted(missing)}")

    matches = matches.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)

    home_elo = np.empty(len(matches), dtype=float)
    away_elo = np.empty(len(matches), dtype=float)

    for idx, row in matches.iterrows():
        home, away = row["home"], row["away"]
        r_home, r_away = ratings[home], ratings[away]
        home_elo[idx] = r_home
        away_elo[idx] = r_away

        adv = 0.0 if row["neutral"] else ELO_HOME_ADV
        diff = (r_home + adv) - r_away
        expected_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            actual_home = 1.0
        elif hs == as_:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = _match_weight(row) * _goal_multiplier(hs - as_)
        delta = k * (actual_home - expected_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    out = matches.copy()
    out["home_elo_before"] = home_elo
    out["away_elo_before"] = away_elo
    out["elo_diff"] = home_elo - away_elo

    path = PATHS.processed / "matches_with_elo.parquet"
    out.to_parquet(path, index=False)
    log.info("wrote %d rows with Elo -> %s", len(out), path)

    latest = pd.Series(ratings, name="elo").sort_values(ascending=False)
    latest_path = PATHS.processed / "latest_elo.parquet"
    latest.rename_axis("team").reset_index().to_parquet(latest_path, index=False)
    log.info("snapshot: top-5 current Elo %s", latest.head(5).to_dict())
    return out


def load_latest_elo() -> pd.Series:
    path = PATHS.processed / "latest_elo.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "latest_elo.parquet missing — run the training pipeline first."
        )
    df = pd.read_parquet(path)
    return pd.Series(df["elo"].to_numpy(), index=df["team"])
