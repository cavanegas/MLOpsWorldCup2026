# Deployment Guide

## Local — FastAPI
```bash
pip install -e ".[dev]"
python scripts/run_pipeline.py      # trains & produces the joblib artefact
uvicorn worldcup2026.api.app:app --reload --port 8000
```

Open http://localhost:8000/docs for the OpenAPI UI.

## Docker
```bash
docker build -t worldcup2026:local .
docker run -p 8000:8000 worldcup2026:local
```

## Docker Compose (API + MLflow server)
```bash
docker compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

## Prefect schedule
```bash
prefect server start                  # terminal 1
python scripts/deploy_prefect.py      # terminal 2 — serves a weekly cron
```

## Cloud targets (nice-to-have)
- **GCP Cloud Run** — push the Docker image to Artifact Registry and
  `gcloud run deploy worldcup2026 --image=...`.
- **AWS ECS Fargate** — same image, define a task with port 8000.
- **Azure Container Instances** — `az container create --image ...`.
- **Vertex AI Pipelines** — wrap the Prefect flow as KFP components.

## CI/CD
`.github/workflows/ci.yml` runs ruff, black and pytest on each push. Tagging
a release (`git tag v0.1.0 && git push --tags`) triggers `deploy.yml`, which
builds and (optionally) pushes the image to GHCR.
