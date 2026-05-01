"""PSI drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.monitoring.drift import compute_drift


def test_drift_is_stable_on_same_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"home_elo_before": rng.normal(1500, 100, size=2000),
                        "elo_diff": rng.normal(0, 80, size=2000)})
    cur = pd.DataFrame({"home_elo_before": rng.normal(1500, 100, size=2000),
                        "elo_diff": rng.normal(0, 80, size=2000)})
    report = compute_drift(ref, cur, columns=list(ref.columns))
    assert report.status in {"OK", "WARNING"}
    assert all(v < 0.25 for v in report.psi.values())


def test_drift_is_flagged_on_shifted_distribution() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"home_elo_before": rng.normal(1500, 100, size=2000)})
    cur = pd.DataFrame({"home_elo_before": rng.normal(1800, 150, size=2000)})
    report = compute_drift(ref, cur, columns=["home_elo_before"])
    assert report.status == "ALERT"
    assert report.psi["home_elo_before"] > 0.25
