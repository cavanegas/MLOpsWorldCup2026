"""Feature matrix builder tests."""

from __future__ import annotations

import pandas as pd

from worldcup2026.features.build_features import FEATURE_COLUMNS, build_dataset
from worldcup2026.features.elo import compute_elo_history


def test_build_dataset_shape(synthetic_matches: pd.DataFrame) -> None:
    # Shift synthetic dates into the modelled window.
    df = synthetic_matches.copy()
    df["date"] = df["date"] + pd.Timedelta(days=365 * 15)
    matches_with_elo = compute_elo_history(df)

    ds = build_dataset(matches_with_elo)

    assert list(ds.X.columns) == list(FEATURE_COLUMNS)
    assert len(ds.X) == len(ds.y) == len(ds.meta)
    assert set(ds.y.unique()) <= {0, 1, 2}
    assert ds.X["elo_diff"].notna().all()
