"""Shared pytest fixtures — synthetic match history so tests never hit the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D", "E", "F"]
    rows = []
    start = pd.Timestamp("2010-01-01")
    for i in range(200):
        home, away = rng.choice(teams, size=2, replace=False)
        hs = int(rng.poisson(1.5))
        as_ = int(rng.poisson(1.2))
        rows.append({
            "date": start + pd.Timedelta(days=i * 7),
            "home": str(home),
            "away": str(away),
            "home_score": hs,
            "away_score": as_,
            "tournament": "Friendly" if i % 3 else "FIFA World Cup qualification",
            "city": "Town",
            "country": "Nowhere",
            "neutral": bool(i % 5 == 0),
        })
    df = pd.DataFrame(rows)
    df["is_worldcup"] = df["tournament"] == "FIFA World Cup"
    df["is_knockout"] = False
    return df
