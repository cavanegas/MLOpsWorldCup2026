"""Typer-based CLI exposing the most common MLOps actions."""

from __future__ import annotations

import json
from typing import Optional

import typer

from .data.ingest import ingest_all
from .data.preprocess import build_match_history
from .features.build_features import build_dataset, load_training_dataset
from .features.elo import compute_elo_history
from .logging_utils import get_logger
from .models.predict import predict_match
from .models.train import train_model
from .simulation.tournament import simulate_tournament

app = typer.Typer(help="CLI for the WorldCup 2026 prediction pipeline.")
log = get_logger(__name__)


@app.command()
def ingest(force: bool = typer.Option(False, help="Re-download even if cached.")) -> None:
    """Download every raw dataset used by the project."""
    paths = ingest_all(force=force)
    typer.echo(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


@app.command()
def preprocess() -> None:
    """Normalise raw data into match_history.parquet."""
    df = build_match_history()
    typer.echo(f"wrote {len(df)} rows")


@app.command()
def build_elo() -> None:
    """Replay Elo ratings and write matches_with_elo.parquet."""
    from .data.preprocess import load_match_history
    df = load_match_history()
    out = compute_elo_history(df)
    typer.echo(f"wrote {len(out)} rows with Elo")


@app.command()
def features() -> None:
    """Build the training dataset."""
    import pandas as pd
    from .config import PATHS
    df = pd.read_parquet(PATHS.processed / "matches_with_elo.parquet")
    ds = build_dataset(df)
    typer.echo(f"features={ds.X.shape[1]} rows={len(ds.X)}")


@app.command()
def train(
    model: str = typer.Option("gbt", help="One of: logreg, gbt, xgb"),
    n_estimators: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> None:
    """Train the match-outcome classifier with MLflow tracking."""
    ds = load_training_dataset()
    params: dict = {}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    trained = train_model(ds, model_name=model, params=params or None, register=False)
    typer.echo(json.dumps({"metrics": trained.metrics, "run_id": trained.run_id}, indent=2))


@app.command()
def simulate(n: int = typer.Option(2000, help="Number of Monte Carlo simulations")) -> None:
    """Simulate the WC 2026 bracket and write champion probabilities."""
    result = simulate_tournament(n_simulations=n)
    table = result.probability_table()
    typer.echo(table.head(10).to_string(index=False))


@app.command()
def predict(
    home: str = typer.Argument(..., help="Home team name"),
    away: str = typer.Argument(..., help="Away team name"),
    neutral: bool = typer.Option(True, help="Is it played at a neutral venue?"),
) -> None:
    """Predict the outcome of one match."""
    probs = predict_match(home, away, neutral=neutral)
    typer.echo(json.dumps(probs, indent=2))


@app.command("run-pipeline")
def run_pipeline(
    force_download: bool = False,
    model: str = "gbt",
    simulations: int = 2000,
) -> None:
    """Run the full Prefect flow (ingest -> train -> simulate)."""
    from .pipeline.flow import worldcup2026_pipeline
    result = worldcup2026_pipeline(
        force_download=force_download,
        model_name=model,
        n_simulations=simulations,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    app()
