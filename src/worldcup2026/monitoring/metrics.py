"""Prometheus-style metrics exposed by the FastAPI app.

We implement a minimal in-process counter so the project has a working
``/metrics`` endpoint that Grafana/Prometheus can scrape, without adding a
new dependency on prometheus_client. If the latter is available we prefer it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

try:  # pragma: no cover - soft dependency
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _HAS_PROMETHEUS = True
except Exception:
    _HAS_PROMETHEUS = False


if _HAS_PROMETHEUS:
    PREDICT_COUNTER = Counter("wc2026_predict_total", "Total predictions served",
                              ["endpoint"])
    PREDICT_LATENCY = Histogram("wc2026_predict_latency_seconds",
                                "Prediction latency in seconds",
                                ["endpoint"])

    def observe(endpoint: str, elapsed: float) -> None:
        PREDICT_COUNTER.labels(endpoint=endpoint).inc()
        PREDICT_LATENCY.labels(endpoint=endpoint).observe(elapsed)

    def render() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST

else:  # fallback: in-memory pseudo prometheus output
    _LOCK = Lock()
    _COUNTS: dict[str, int] = defaultdict(int)
    _LATENCIES: dict[str, list[float]] = defaultdict(list)

    def observe(endpoint: str, elapsed: float) -> None:
        with _LOCK:
            _COUNTS[endpoint] += 1
            _LATENCIES[endpoint].append(elapsed)

    def render() -> tuple[bytes, str]:
        lines = ["# HELP wc2026_predict_total Total predictions served",
                 "# TYPE wc2026_predict_total counter"]
        with _LOCK:
            for ep, n in _COUNTS.items():
                lines.append(f'wc2026_predict_total{{endpoint="{ep}"}} {n}')
            lines.append("# HELP wc2026_predict_latency_seconds Prediction latency")
            lines.append("# TYPE wc2026_predict_latency_seconds summary")
            for ep, values in _LATENCIES.items():
                if not values:
                    continue
                avg = sum(values) / len(values)
                lines.append(f'wc2026_predict_latency_seconds_sum{{endpoint="{ep}"}} {sum(values):.6f}')
                lines.append(f'wc2026_predict_latency_seconds_count{{endpoint="{ep}"}} {len(values)}')
                lines.append(f'wc2026_predict_latency_avg{{endpoint="{ep}"}} {avg:.6f}')
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/plain; version=0.0.4"


class Timer:
    """Context manager that reports to `observe(endpoint, elapsed)` on exit."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        observe(self.endpoint, time.perf_counter() - self._t0)
