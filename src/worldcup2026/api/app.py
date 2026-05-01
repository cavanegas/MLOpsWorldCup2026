"""FastAPI service exposing match-level predictions and tournament probabilities."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import PATHS
from ..logging_utils import get_logger
from ..models.predict import load_artifact, predict_match
from ..monitoring.metrics import Timer, render as render_metrics
from ..simulation.tournament import simulate_tournament

log = get_logger(__name__)

app = FastAPI(
    title="WorldCup 2026 Prediction API",
    version="0.1.0",
    description="Predict match outcomes and the FIFA World Cup 2026 champion.",
)


class MatchRequest(BaseModel):
    home: str = Field(..., examples=["Argentina"])
    away: str = Field(..., examples=["France"])
    neutral: bool = True
    is_knockout: bool = False


class MatchResponse(BaseModel):
    home: str
    away: str
    probabilities: dict[str, float]


class SimulationRequest(BaseModel):
    n_simulations: int = Field(500, ge=50, le=20000)
    teams: list[str] | None = None
    seed: int = 42


class SimulationResponse(BaseModel):
    n_simulations: int
    top10: list[dict]


def _cached_artifact() -> dict:
    if not hasattr(app.state, "artifact"):
        app.state.artifact = load_artifact()
    return app.state.artifact


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"model_loaded": Path(PATHS.model_artifact).exists()}


@app.post("/predict/match", response_model=MatchResponse)
def predict_match_endpoint(req: MatchRequest) -> MatchResponse:
    with Timer("predict_match"):
        try:
            probs = predict_match(
                home=req.home,
                away=req.away,
                neutral=req.neutral,
                artifact=_cached_artifact(),
                is_knockout=req.is_knockout,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return MatchResponse(home=req.home, away=req.away, probabilities=probs)


@app.post("/predict/tournament", response_model=SimulationResponse)
def predict_tournament(req: SimulationRequest) -> SimulationResponse:
    with Timer("predict_tournament"):
        try:
            result = simulate_tournament(
                n_simulations=req.n_simulations,
                artifact=_cached_artifact(),
                teams=req.teams,
                seed=req.seed,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        table = result.probability_table().head(10).to_dict(orient="records")
        return SimulationResponse(n_simulations=req.n_simulations, top10=table)


@app.get("/metrics")
def metrics() -> Response:
    body, ct = render_metrics()
    return Response(content=body, media_type=ct)


@app.get("/champion_probabilities")
def champion_probabilities_batch() -> dict:
    """Return the cached batch output from the last pipeline run."""
    path = PATHS.processed / "champion_probabilities.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No batch output yet — run the pipeline.")
    import pandas as pd
    df = pd.read_csv(path).head(25)
    return json.loads(df.to_json(orient="records"))
