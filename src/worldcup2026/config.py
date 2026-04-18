"""Central configuration for the WorldCup 2026 MLOps project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def _env_path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_dir: Path = _env_path("DATA_DIR", str(PROJECT_ROOT / "data"))
    raw: Path = field(init=False)
    processed: Path = field(init=False)
    external: Path = field(init=False)
    models: Path = _env_path("MODELS_DIR", str(PROJECT_ROOT / "models_store"))
    mlruns: Path = _env_path("MLFLOW_TRACKING_DIR", str(PROJECT_ROOT / "mlruns"))
    model_artifact: Path = _env_path(
        "MODEL_PATH", str(PROJECT_ROOT / "models_store" / "champion_model.joblib")
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", self.data_dir / "raw")
        object.__setattr__(self, "processed", self.data_dir / "processed")
        object.__setattr__(self, "external", self.data_dir / "external")
        for p in (self.raw, self.processed, self.external, self.models, self.mlruns):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlruns.db'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "worldcup2026")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Data sources -------------------------------------------------------------
MARTJ42_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
FJELSTUL_BASE_URL = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/"
)
FJELSTUL_FILES = (
    "tournaments.csv",
    "matches.csv",
    "teams.csv",
    "host_countries.csv",
)

# --- 2026 World Cup structural knowledge --------------------------------------
HOSTS_2026: tuple[str, ...] = ("United States", "Mexico", "Canada")

# Pre-qualified + strong contenders for a 48-team WC simulation. Keep the list
# explicit so the project is deterministic even if the bracket isn't final yet.
WORLDCUP_2026_CONTENDERS: tuple[str, ...] = (
    # Hosts
    "United States", "Mexico", "Canada",
    # UEFA
    "France", "England", "Spain", "Germany", "Portugal", "Bosnia and Herzegovina", "Netherlands",
    "Belgium", "Croatia", "Switzerland", "Czech Republic", "Scotland", "Austria", "Turkey", "Sweden", "Norway",
    # CONMEBOL
    "Brazil", "Argentina", "Uruguay", "Colombia", "Ecuador", "Paraguay",
    # AFC
    "Japan", "Korea Republic", "Iran", "Saudi Arabia", "Australia", "Qatar", "Jordan", "Uzbekistan", "Iraq"
    # CAF
    "Morocco", "Senegal", "Egypt", "South Africa", "Algeria", "Cape Verde", "Ghana",
    "Ivory Coast", "Tunisia", "Republic of the Congo"
    # CONCACAF
    "Curacao", "Panama", "Haiti",
    # OFC
    "New Zealand"
)

# Elo model hyperparameters (chess-style adaptation used by eloratings.net)
ELO_INITIAL = 1500.0
ELO_K = 30.0           # base K-factor
ELO_HOME_ADV = 100.0   # points added to home team's pre-match rating
ELO_WC_WEIGHT = 60.0   # extra weight for knockout / world-cup matches
