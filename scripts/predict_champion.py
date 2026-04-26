"""Batch prediction script — writes champion_probabilities.csv and prints top 10."""

from __future__ import annotations

import argparse

from worldcup2026.config import PATHS
from worldcup2026.simulation.tournament import simulate_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the FIFA World Cup 2026")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = simulate_tournament(n_simulations=args.simulations, seed=args.seed)
    table = result.probability_table()
    out = PATHS.processed / "champion_probabilities.csv"
    table.to_csv(out, index=False)
    print(table.head(10).to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
