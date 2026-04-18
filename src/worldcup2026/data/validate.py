"""Lightweight data-quality checks for the match history.

Runs as part of the Prefect flow after ``preprocess`` and raises a
``DataValidationError`` when any hard constraint is violated. Soft warnings
are logged but do not stop the pipeline — the goal is a visible, declarative
contract on the schema.
"""

from __future__ import annotations

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


class DataValidationError(Exception):
    pass


REQUIRED_COLUMNS: tuple[str, ...] = (
    "date", "home", "away", "home_score", "away_score",
    "tournament", "neutral", "is_worldcup", "is_knockout",
)


def validate_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` unchanged after enforcing the schema & basic sanity rules."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    if df.empty:
        raise DataValidationError("match history is empty")

    if df[["home_score", "away_score"]].isna().any().any():
        raise DataValidationError("scores contain NaN after preprocessing")

    if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
        raise DataValidationError("negative scores detected")

    if df["date"].isna().any():
        raise DataValidationError("null dates detected")

    if not df["date"].is_monotonic_increasing:
        log.warning("match history is not sorted by date — sorting in place")
        df = df.sort_values("date").reset_index(drop=True)

    # Soft checks
    dup = df.duplicated(subset=["date", "home", "away"]).sum()
    if dup:
        log.warning("found %d duplicated (date, home, away) rows", dup)

    future = (df["date"] > pd.Timestamp.utcnow().tz_localize(None)).sum()
    if future:
        log.warning("found %d rows with future dates — likely pipeline error", future)

    log.info("match history validated: %d rows, %d teams",
             len(df), df[["home", "away"]].stack().nunique())
    return df
