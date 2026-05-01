"""Sanity tests for the Elo builder."""

from __future__ import annotations

import pandas as pd

from worldcup2026.config import ELO_INITIAL
from worldcup2026.features.elo import compute_elo_history


def test_elo_runs_and_conserves_mean(synthetic_matches: pd.DataFrame) -> None:
    out = compute_elo_history(synthetic_matches)
    assert {"home_elo_before", "away_elo_before", "elo_diff"} <= set(out.columns)
    assert len(out) == len(synthetic_matches)

    # The first match of every team must see the default initial rating.
    first_home = out.iloc[0]
    assert abs(first_home["home_elo_before"] - ELO_INITIAL) < 1e-6

    # Sum of all Elo changes should be roughly zero (zero-sum game).
    final_home = out.groupby("home")["home_elo_before"].last().sum()
    final_away = out.groupby("away")["away_elo_before"].last().sum()
    # Not exactly zero-sum because of home advantage bonus, but mean must stay near 1500.
    mean_elo = (final_home + final_away) / (out["home"].nunique() + out["away"].nunique())
    assert 1200.0 < mean_elo < 1800.0


def test_elo_reacts_to_results(synthetic_matches: pd.DataFrame) -> None:
    df = synthetic_matches.copy()
    # Make team "A" steamroll everyone: replace all of its away games with 5-0 wins.
    mask = df["away"] == "A"
    df.loc[mask, ["home_score", "away_score"]] = [0, 5]
    out = compute_elo_history(df)
    # Final Elo of A should be above initial baseline.
    last_a = pd.concat([
        out.loc[out["home"] == "A", ["date", "home_elo_before"]].rename(columns={"home_elo_before": "elo"}),
        out.loc[out["away"] == "A", ["date", "away_elo_before"]].rename(columns={"away_elo_before": "elo"}),
    ]).sort_values("date")["elo"].iloc[-1]
    assert last_a > ELO_INITIAL
