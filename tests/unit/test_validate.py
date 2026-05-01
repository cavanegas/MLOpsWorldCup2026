"""Schema/sanity checks."""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup2026.data.validate import DataValidationError, validate_match_history


def test_validate_happy_path(synthetic_matches: pd.DataFrame) -> None:
    out = validate_match_history(synthetic_matches)
    assert len(out) == len(synthetic_matches)


def test_validate_missing_column(synthetic_matches: pd.DataFrame) -> None:
    df = synthetic_matches.drop(columns=["neutral"])
    with pytest.raises(DataValidationError):
        validate_match_history(df)


def test_validate_negative_scores(synthetic_matches: pd.DataFrame) -> None:
    df = synthetic_matches.copy()
    df.loc[0, "home_score"] = -1
    with pytest.raises(DataValidationError):
        validate_match_history(df)
