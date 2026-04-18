"""Build the feature matrix that feeds the classifier.

Target: ``result ∈ {home_win, draw, away_win}`` encoded as ``{0, 1, 2}``.
Features are all computed with *ex-ante* information only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import HOSTS_2026, PATHS
from ..logging_utils import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "neutral",
    "home_is_host2026",
    "away_is_host2026",
    "home_form5",
    "away_form5",
    "home_goals_for_form5",
    "away_goals_for_form5",
    "home_goals_against_form5",
    "away_goals_against_form5",
    "is_worldcup",
    "is_knockout",
)


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def _rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """For each team compute form (points per game) and goals over the last N matches."""
    long = pd.concat(
        [
            df[["date", "home", "home_score", "away_score"]]
            .rename(columns={"home": "team", "home_score": "gf", "away_score": "ga"})
            .assign(side="home"),
            df[["date", "away", "away_score", "home_score"]]
            .rename(columns={"away": "team", "away_score": "gf", "home_score": "ga"})
            .assign(side="away"),
        ],
        ignore_index=True,
    ).sort_values(["team", "date"])

    long["points"] = np.where(
        long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0)
    )

    grouped = long.groupby("team", group_keys=False)
    # shift so row t sees only matches strictly before t.
    long["form_pts"] = grouped["points"].apply(lambda s: s.shift().rolling(window, min_periods=1).mean())
    long["form_gf"] = grouped["gf"].apply(lambda s: s.shift().rolling(window, min_periods=1).mean())
    long["form_ga"] = grouped["ga"].apply(lambda s: s.shift().rolling(window, min_periods=1).mean())

    return long[["date", "team", "side", "form_pts", "form_gf", "form_ga"]]


def build_dataset(matches_with_elo: pd.DataFrame) -> Dataset:
    """Return the training dataset (features + label + meta)."""
    df = matches_with_elo.copy()
    df = df[df["date"] >= "1994-01-01"].reset_index(drop=True)

    form = _rolling_form(df, window=5)
    home_form = form[form["side"] == "home"].drop(columns=["side"])
    away_form = form[form["side"] == "away"].drop(columns=["side"])
    df = df.merge(home_form, left_on=["date", "home"], right_on=["date", "team"], how="left").drop(columns="team")
    df = df.rename(columns={
        "form_pts": "home_form5",
        "form_gf": "home_goals_for_form5",
        "form_ga": "home_goals_against_form5",
    })
    df = df.merge(away_form, left_on=["date", "away"], right_on=["date", "team"], how="left").drop(columns="team")
    df = df.rename(columns={
        "form_pts": "away_form5",
        "form_gf": "away_goals_for_form5",
        "form_ga": "away_goals_against_form5",
    })

    df["home_is_host2026"] = df["home"].isin(HOSTS_2026).astype(int)
    df["away_is_host2026"] = df["away"].isin(HOSTS_2026).astype(int)

    df["result"] = np.where(
        df["home_score"] > df["away_score"], 0,
        np.where(df["home_score"] == df["away_score"], 1, 2),
    )

    df = df.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)

    meta_cols = ["date", "home", "away", "home_score", "away_score", "tournament"]
    features = df[list(FEATURE_COLUMNS)].copy()
    features["neutral"] = features["neutral"].astype(int)
    features["is_worldcup"] = features["is_worldcup"].astype(int)
    features["is_knockout"] = features["is_knockout"].astype(int)

    ds = Dataset(X=features, y=df["result"], meta=df[meta_cols])
    out = pd.concat([ds.meta.reset_index(drop=True), ds.X.reset_index(drop=True),
                     ds.y.reset_index(drop=True).rename("result")], axis=1)
    out.to_parquet(PATHS.processed / "training_dataset.parquet", index=False)
    log.info("training dataset: %s rows, %s features", len(out), ds.X.shape[1])
    return ds


def load_training_dataset() -> Dataset:
    path = PATHS.processed / "training_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError("run the pipeline first — training_dataset.parquet missing")
    df = pd.read_parquet(path)
    X = df[list(FEATURE_COLUMNS)]
    y = df["result"]
    meta = df.drop(columns=[*FEATURE_COLUMNS, "result"])
    return Dataset(X=X, y=y, meta=meta)
