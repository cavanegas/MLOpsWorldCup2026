"""Structural tests for the tournament simulator."""

from __future__ import annotations

from worldcup2026.simulation.tournament import _serpentine_groups


def test_serpentine_groups_sizes() -> None:
    teams = [f"T{i:02d}" for i in range(48)]
    groups = _serpentine_groups(teams, n_groups=12)
    assert len(groups) == 12
    assert all(len(g) == 4 for g in groups)
    flat = [t for g in groups for t in g]
    assert set(flat) == set(teams)


def test_serpentine_groups_top_seeds_spread() -> None:
    teams = [f"S{i:02d}" for i in range(48)]
    groups = _serpentine_groups(teams, n_groups=12)
    # Each group must have exactly one "top 12" seed.
    for g in groups:
        first_band = [t for t in g if int(t[1:]) < 12]
        assert len(first_band) == 1
