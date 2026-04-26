"""End-to-end Prefect flow: ingest -> preprocess -> features -> train -> simulate.

Run it locally with::

    python -m worldcup2026.pipeline.flow

or through Prefect::

    prefect deployment build src/worldcup2026/pipeline/flow.py:worldcup2026_pipeline -n local
"""

from __future__ import annotations

import json
from typing import Any

from prefect import flow, task

from ..config import PATHS
from ..data.ingest import ingest_all
from ..data.preprocess import build_match_history
from ..data.validate import validate_match_history
from ..features.build_features import build_dataset
from ..features.elo import compute_elo_history
from ..logging_utils import get_logger
from ..models.train import train_model
from ..simulation.tournament import simulate_tournament

log = get_logger(__name__)


@task(name="ingest", retries=2, retry_delay_seconds=30)
def ingest_task(force: bool = False) -> dict[str, str]:
    return {k: str(v) for k, v in ingest_all(force=force).items()}


@task(name="preprocess")
def preprocess_task() -> int:
    df = build_match_history()
    return len(df)


@task(name="validate_data")
def validate_task() -> int:
    from ..data.preprocess import load_match_history
    df = load_match_history()
    df = validate_match_history(df)
    return len(df)


@task(name="elo")
def elo_task() -> int:
    from ..data.preprocess import load_match_history
    df = load_match_history()
    out = compute_elo_history(df)
    return len(out)


@task(name="features")
def features_task() -> int:
    from ..features.elo import load_latest_elo  # noqa: F401 - ensures file exists
    import pandas as pd
    df = pd.read_parquet(PATHS.processed / "matches_with_elo.parquet")
    ds = build_dataset(df)
    return len(ds.X)


@task(name="train")
def train_task(
    model_name: str = "gbt",
    params: dict[str, Any] | None = None,
    promote_to_stage: str | None = None,
) -> dict[str, Any]:
    from ..features.build_features import load_training_dataset
    ds = load_training_dataset()
    trained = train_model(
        ds, model_name=model_name, params=params, register=False,
        promote_to_stage=promote_to_stage,
    )
    return {"model_name": trained.model_name, "metrics": trained.metrics,
            "run_id": trained.run_id}


@task(name="simulate")
def simulate_task(n_simulations: int = 2000) -> dict[str, Any]:
    result = simulate_tournament(n_simulations=n_simulations)
    table = result.probability_table()
    out = PATHS.processed / "champion_probabilities.csv"
    table.to_csv(out, index=False)
    top = table.head(10).to_dict(orient="records")
    (PATHS.processed / "top10_champion_probabilities.json").write_text(
        json.dumps(top, indent=2)
    )
    log.info("top 5 champions: %s", top[:5])
    return {"top10": top, "n_simulations": n_simulations, "csv": str(out)}


@flow(name="worldcup2026_pipeline")
def worldcup2026_pipeline(
    force_download: bool = False,
    model_name: str = "gbt",
    params: dict[str, Any] | None = None,
    n_simulations: int = 2000,
    promote_to_stage: str | None = None,
) -> dict[str, Any]:
    ingest_task(force=force_download)
    preprocess_task()
    validate_task()
    elo_task()
    features_task()
    metrics = train_task(model_name=model_name, params=params,
                         promote_to_stage=promote_to_stage)
    simulation = simulate_task(n_simulations=n_simulations)
    return {"training": metrics, "simulation": simulation}


if __name__ == "__main__":
    worldcup2026_pipeline()
