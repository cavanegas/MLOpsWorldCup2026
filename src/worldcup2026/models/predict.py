
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import HOSTS_2026, PATHS
from ..features.build_features import FEATURE_COLUMNS
from ..features.elo import load_latest_elo


def load_artifact(path: Path | None = None) -> dict[str, Any]:
    path = path or PATHS.model_artifact
    if not Path(path).exists():
        raise FileNotFoundError(
            f"model artifact not found at {path}. Run the training pipeline first."
        )
    return joblib.load(path)


def build_match_features(
    home: str,
    away: str,
    neutral: bool,
    elo: pd.Series,
    form: pd.DataFrame | None = None,
    is_worldcup: bool = True,
    is_knockout: bool = False,
) -> pd.DataFrame:
    """Build one feature row given latest team stats.

    ``form`` is optional; when missing we fall back to Elo-implied averages so the
    model still gets reasonable inputs (this is what enables a cold-start prediction).
    """
    home_elo = float(elo.get(home, 1500.0))
    away_elo = float(elo.get(away, 1500.0))

    def _form_lookup(team: str, col: str, fallback: float) -> float:
        if form is None or team not in form.index:
            return fallback
        return float(form.loc[team, col])

    row = {
        "home_elo_before": home_elo,
        "away_elo_before": away_elo,
        "elo_diff": home_elo - away_elo,
        "neutral": int(neutral),
        "home_is_host2026": int(home in HOSTS_2026),
        "away_is_host2026": int(away in HOSTS_2026),
        "home_form5": _form_lookup(home, "form_pts", 1.4),
        "away_form5": _form_lookup(away, "form_pts", 1.4),
        "home_goals_for_form5": _form_lookup(home, "form_gf", 1.3),
        "away_goals_for_form5": _form_lookup(away, "form_gf", 1.3),
        "home_goals_against_form5": _form_lookup(home, "form_ga", 1.3),
        "away_goals_against_form5": _form_lookup(away, "form_ga", 1.3),
        "is_worldcup": int(is_worldcup),
        "is_knockout": int(is_knockout),
    }
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))


def predict_match(
    home: str,
    away: str,
    neutral: bool = True,
    artifact: dict[str, Any] | None = None,
    is_worldcup: bool = True,
    is_knockout: bool = False,
) -> dict[str, float]:
    artifact = artifact or load_artifact()
    elo = load_latest_elo()
    X = build_match_features(home, away, neutral, elo,
                             is_worldcup=is_worldcup, is_knockout=is_knockout)
    proba = artifact["pipeline"].predict_proba(X[artifact["features"]])[0]
    return {cls: float(p) for cls, p in zip(artifact["classes"], proba, strict=True)}


def predict_match_probs(
    home: str,
    away: str,
    neutral: bool,
    elo: pd.Series,
    artifact: dict[str, Any],
    is_knockout: bool = False,
) -> np.ndarray:
    """Vector of [home_win, draw, away_win] probabilities — simulator fast path."""
    X = build_match_features(home, away, neutral, elo,
                             is_worldcup=True, is_knockout=is_knockout)
    return artifact["pipeline"].predict_proba(X[artifact["features"]])[0]
