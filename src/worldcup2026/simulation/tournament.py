"""Monte Carlo simulator for the 48-team FIFA World Cup 2026 format.

The 2026 bracket is 12 groups of 4, followed by a 32-team knockout round (round
of 32). To keep the project deterministic without hard-wiring the live draw, we
approximate the bracket by ordering qualified teams by current Elo rating and
drawing groups serpentine-style. The user can override the seeding list via the
``teams`` argument.

Each match is sampled from the classifier's probability output. In knockout
rounds, draws are resolved by re-sampling a win/loss outcome biased by the
pre-match Elo differential (mimicking penalty shootouts).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import HOSTS_2026, WORLDCUP_2026_CONTENDERS
from ..features.elo import load_latest_elo
from ..logging_utils import get_logger
from ..models.predict import load_artifact, predict_match_probs

log = get_logger(__name__)


@dataclass
class SimulationResult:
    championship_counts: Counter
    finalist_counts: Counter
    semifinal_counts: Counter
    n_simulations: int
    seeds: list[str] = field(default_factory=list)

    def probability_table(self) -> pd.DataFrame:
        rows = []
        for team in sorted(self.seeds):
            rows.append({
                "team": team,
                "champion_prob": self.championship_counts[team] / self.n_simulations,
                "final_prob": self.finalist_counts[team] / self.n_simulations,
                "semifinal_prob": self.semifinal_counts[team] / self.n_simulations,
            })
        return (pd.DataFrame(rows)
                .sort_values("champion_prob", ascending=False)
                .reset_index(drop=True))


def _serpentine_groups(teams: list[str], n_groups: int = 12) -> list[list[str]]:
    """Distribute a seeded team list into N groups, snaking through seed bands."""
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, team in enumerate(teams):
        band = i // n_groups
        idx = i % n_groups
        if band % 2 == 1:
            idx = n_groups - 1 - idx
        groups[idx].append(team)
    return groups


def _play(
    home: str,
    away: str,
    neutral: bool,
    artifact: dict[str, Any],
    elo: pd.Series,
    rng: random.Random,
    knockout: bool,
) -> tuple[str, str | None]:
    """Sample a match outcome. Returns (winner, loser) — loser is None on a draw."""
    probs = predict_match_probs(home, away, neutral, elo, artifact, is_knockout=knockout)
    r = rng.random()
    if r < probs[0]:
        return home, away
    if r < probs[0] + probs[1]:
        if not knockout:
            return "draw", None
        # Coin flip biased by Elo for shootouts.
        p_home = 1 / (1 + 10 ** (-(elo.get(home, 1500) - elo.get(away, 1500)) / 400.0))
        return (home, away) if rng.random() < p_home else (away, home)
    return away, home


def _play_group_stage(
    groups: list[list[str]],
    artifact: dict[str, Any],
    elo: pd.Series,
    rng: random.Random,
) -> list[str]:
    """Return the 32 teams that advance (top-2 per group + 8 best thirds)."""
    qualified: list[str] = []
    thirds: list[tuple[str, int, int]] = []  # (team, points, goal_diff proxy)
    for g in groups:
        table: dict[str, int] = defaultdict(int)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                home, away = g[i], g[j]
                # Hosts always play at home.
                neutral = home not in HOSTS_2026 and away not in HOSTS_2026
                winner, loser = _play(home, away, neutral, artifact, elo, rng, knockout=False)
                if winner == "draw":
                    table[home] += 1
                    table[away] += 1
                else:
                    table[winner] += 3
        ranking = sorted(g, key=lambda t: (table[t], rng.random()), reverse=True)
        qualified.extend(ranking[:2])
        if len(ranking) >= 3:
            thirds.append((ranking[2], table[ranking[2]], 0))

    thirds.sort(key=lambda r: (r[1], random.random()), reverse=True)
    qualified.extend(team for team, *_ in thirds[:8])
    return qualified


def _play_knockout(
    teams: list[str],
    artifact: dict[str, Any],
    elo: pd.Series,
    rng: random.Random,
    milestone_counts: dict[str, Counter] | None = None,
) -> str:
    """Bracket-style single elimination. ``teams`` must have a power-of-two length."""
    current = list(teams)
    while len(current) > 1:
        round_size = len(current)
        if milestone_counts is not None:
            if round_size == 4:
                for t in current:
                    milestone_counts["semifinal"][t] += 1
            if round_size == 2:
                for t in current:
                    milestone_counts["final"][t] += 1
        next_round: list[str] = []
        for i in range(0, round_size, 2):
            home, away = current[i], current[i + 1]
            neutral = home not in HOSTS_2026 and away not in HOSTS_2026
            winner, _ = _play(home, away, neutral, artifact, elo, rng, knockout=True)
            next_round.append(winner)
        current = next_round
    return current[0]


def _seed_teams(elo: pd.Series, teams: list[str] | None) -> list[str]:
    contenders = list(teams) if teams else list(WORLDCUP_2026_CONTENDERS)
    # Guarantee hosts are included.
    for host in HOSTS_2026:
        if host not in contenders:
            contenders.append(host)
    ranked = sorted(contenders, key=lambda t: -float(elo.get(t, 1500.0)))
    return ranked[:48]


def simulate_tournament(
    n_simulations: int = 2000,
    artifact: dict[str, Any] | None = None,
    teams: list[str] | None = None,
    seed: int = 42,
) -> SimulationResult:
    artifact = artifact or load_artifact()
    elo = load_latest_elo()
    rng = random.Random(seed)
    np.random.seed(seed)

    seeds = _seed_teams(elo, teams)
    groups_template = _serpentine_groups(seeds, n_groups=12)
    log.info("seeds=%s...  top8=%s", len(seeds), seeds[:8])

    champion_counts: Counter = Counter()
    milestones: dict[str, Counter] = {"semifinal": Counter(), "final": Counter()}

    for sim in range(n_simulations):
        qualified = _play_group_stage(groups_template, artifact, elo, rng)
        # Re-rank advancing teams by Elo to emulate a seed-preserving bracket.
        qualified = sorted(qualified, key=lambda t: -float(elo.get(t, 1500.0)))[:32]
        champion = _play_knockout(qualified, artifact, elo, rng,
                                  milestone_counts=milestones)
        champion_counts[champion] += 1
        if (sim + 1) % max(1, n_simulations // 10) == 0:
            log.info("sim %d/%d — leader=%s", sim + 1, n_simulations,
                     champion_counts.most_common(1)[0])

    return SimulationResult(
        championship_counts=champion_counts,
        finalist_counts=milestones["final"],
        semifinal_counts=milestones["semifinal"],
        n_simulations=n_simulations,
        seeds=seeds,
    )
