FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

COPY scripts ./scripts
COPY tests ./tests

RUN mkdir -p data/raw data/processed data/external models_store mlruns

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=/app/mlruns \
    MODEL_PATH=/app/models_store/champion_model.joblib \
    DATA_DIR=/app/data

CMD ["uvicorn", "worldcup2026.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
