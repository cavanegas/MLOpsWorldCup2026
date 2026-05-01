"""Smoke tests for the FastAPI app (no model required for /health)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from worldcup2026.api.app import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()
