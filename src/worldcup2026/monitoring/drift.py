"""Lightweight drift monitoring based on Population Stability Index (PSI).

PSI is the de-facto monitoring metric in credit risk and works well for any
numeric feature:

    PSI = Σ (p_ref - p_curr) * ln(p_ref / p_curr)

Thresholds (industry convention):
    PSI < 0.10     -> stable
    0.10..0.25    -> moderate shift, investigate
    > 0.25         -> significant shift, retrain
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from ..features.build_features import FEATURE_COLUMNS
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class DriftReport:
    psi: dict[str, float] = field(default_factory=dict)
    verdict: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if any(v == "drift" for v in self.verdict.values()):
            return "ALERT"
        if any(v == "warning" for v in self.verdict.values()):
            return "WARNING"
        return "OK"

    def to_dict(self) -> dict:
        return {"status": self.status, "psi": self.psi, "verdict": self.verdict}


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Return the PSI between two 1-D arrays. Safe against 0-mass bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_p = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))


def _classify(psi: float) -> str:
    if psi > 0.25:
        return "drift"
    if psi > 0.10:
        return "warning"
    return "stable"


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  columns: list[str] | None = None) -> DriftReport:
    cols = columns or [c for c in FEATURE_COLUMNS if c in reference.columns and c in current.columns]
    report = DriftReport()
    for col in cols:
        psi = _psi(reference[col].to_numpy(), current[col].to_numpy())
        report.psi[col] = psi
        report.verdict[col] = _classify(psi)
    log.info("drift status=%s", report.status)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PSI drift check.")
    parser.add_argument("--reference", type=str,
                        default=str(PATHS.processed / "training_dataset.parquet"))
    parser.add_argument("--current", type=str, default=None,
                        help="Parquet with recent observations; defaults to last 20% of reference.")
    parser.add_argument("--out", type=str, default=str(PATHS.processed / "drift_report.json"))
    args = parser.parse_args()

    ref = pd.read_parquet(args.reference)
    if args.current:
        cur = pd.read_parquet(args.current)
    else:
        cutoff = int(len(ref) * 0.80)
        ref, cur = ref.iloc[:cutoff], ref.iloc[cutoff:]

    report = compute_drift(ref, cur)
    Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
