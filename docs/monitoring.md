# Monitoring Plan

## 1. Data drift (input monitoring)

`src/worldcup2026/monitoring/drift.py` computes **Population Stability Index
(PSI)** between a reference dataset (the last training snapshot) and the most
recent window.

```bash
python -m worldcup2026.monitoring.drift \
    --reference data/processed/training_dataset.parquet \
    --current data/processed/last_30d_matches.parquet \
    --out logs/drift_report.json
```

Thresholds:

| PSI           | Action               |
|---------------|----------------------|
| < 0.10        | stable — no action   |
| 0.10 – 0.25   | warning — investigate|
| > 0.25        | retrain the model    |

## 2. Concept drift (output monitoring)

After every FIFA window we compare the predicted probabilities against the
actual results.

- **Brier score** rolling 30 days — primary signal.
- **Log-loss** rolling 30 days — secondary.
- Alert when Brier grows ≥ 20% week over week.

## 3. Operational metrics

Exposed at `GET /metrics` in Prometheus format:
- `wc2026_predict_total{endpoint}` — request counter.
- `wc2026_predict_latency_seconds{endpoint}` — latency histogram.

### Proposed dashboard (Grafana)
1. Stacked bar — PSI by feature, refreshed daily.
2. Line — monthly Brier / log-loss (target ≤ 1.0).
3. p95 latency — alert if > 400 ms.
4. Error rate (5xx) — alert if > 1% / 5 min.

## 4. Alerting
- PagerDuty / Slack webhook fired by Grafana when any of:
  - `PSI > 0.25` on any feature for 3 consecutive days.
  - `brier_score` weekly delta > 20%.
  - `5xx rate > 1% over 5 min`.
