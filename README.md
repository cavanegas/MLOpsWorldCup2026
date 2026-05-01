# MLOpsWorldCup2026 — Predicción del Campeón del Mundial FIFA 2026

**Proyecto Final MLOps — Universidad de Medellín**

Proyecto MLOps end-to-end, reproducible, que predice al ganador del Mundial de
Fútbol FIFA 2026. Construido como proyecto final del curso **ML en la Nube** y
diseñado para cubrir las seis fases de la rúbrica oficial: planeación,
experiment tracking, orquestación, deployment, monitoreo y testing/buenas
prácticas.

> **TL;DR** — Un solo comando ingesta ~50 000 partidos internacionales, recalcula
> los ratings Elo desde 1872, entrena un clasificador de resultado de partido
> con tracking en MLflow, y simula por Montecarlo 2 000 veces la llave de 48
> equipos del 2026 para devolver la probabilidad ordenada de que cada país
> levante el trofeo.

```
Ingesta  ──▶ Preproces  ──▶│ Train  ──▶│  Simulación    ──▶│  Serve 
(CSV/API)   Elo + Feats       MLflow       Montecarlo           FastAPI  

▲──────────────── Flow de Prefect (orquestación) ───────────────▲
```

---

## 0. Planeación del proyecto (Fase 1.1)

### 0.1 Problema de negocio (hipotético)
Una casa de apuestas deportivas y un medio de comunicación necesitan publicar,
con anticipación al Mundial 2026, un modelo probabilístico reproducible que
estime la probabilidad de que cada selección sea campeona. El modelo debe
actualizarse cada vez que se juegue una ventana FIFA (aprox. mensual) y exponer
predicciones vía API para alimentar dashboards, artículos editoriales y cuotas
pre-partido. **Valor concreto**: reducir el tiempo manual de análisis de 3 días
a menos de 15 minutos y contar con un pipeline auditado y rastreable
end-to-end.

### 0.2 Métricas de éxito
| Tipo         | Métrica                                                  | Umbral objetivo |
|--------------|-----------------------------------------------------------|-----------------|
| Modelo       | `log_loss` multiclase (home/draw/away)                    | ≤ 1.00          |
| Modelo       | `accuracy` en partidos de torneo (holdout 2018 + 2022)    | ≥ 0.50          |
| Modelo       | `f1_macro`                                                | ≥ 0.42          |
| Negocio      | Prob. del campeón real entre el top-5 del ranking        | ≥ 80 % de runs  |
| Operacional  | Duración end-to-end del pipeline                          | ≤ 5 min         |
| Operacional  | Cobertura de tests                                        | ≥ 60 %          |

### 0.3 Alcance (MVP vs. completo)
- **MVP entregado**: ingest → preprocess → Elo → features → entrenamiento
  (GBT/XGB/LogReg/RF) con MLflow → Monte Carlo de la llave de 48 → API FastAPI
  + Docker + módulo de monitoreo + tests + CI.
- **Completo (fuera del alcance del curso)**: ingesta en tiempo real de
  xG/alineaciones/lesiones (StatsBomb + FotMob), retraining automático en la
  nube (Vertex AI / SageMaker), dashboards Grafana/Datadog en vivo.

### 0.4 Timeline y responsables
| Semana | Fase | Entregables | Responsable |
|--------|------|-------------|-------------|
| 1      | 1 — Setup y EDA            | Repo, entorno uv, notebook EDA, baseline | Data Scientist |
| 2      | 2 — Experiment Tracking    | MLflow con 3 algoritmos, CV, registry    | ML Engineer    |
| 3      | 3 — Pipeline (Prefect)     | Flow ingest→train→simulate, schedule     | ML Engineer    |
| 4      | 4 — Deployment             | FastAPI + Dockerfile + compose           | MLOps Engineer |
| 5      | 5 — Monitoreo              | Módulo de drift + stub de dashboard      | MLOps Engineer |
| 6      | 6 — Testing y Docs         | Pytest, ruff, pre-commit, CI, docs       | Todos          |
| 7      | Peer review + buffer       | Correcciones, deploy a nube (nice-to-have)| Todos         |


---

## 1. Estructura del proyecto

```
MLOps-WorldCup2026/
├── pyproject.toml              # metadata del paquete + config ruff/black/pytest
├── Dockerfile, docker-compose.yml
├── README.md, .env.example, .gitignore, .pre-commit-config.yaml
├── .github/workflows/          # ci.yml, deploy.yml
├── configs/                    # training.yaml, simulation.yaml
├── docs/                       # deployment.md, monitoring.md
├── notebooks/                  # 01_eda, 02_baseline, 03_experiments
├── scripts/                    # wrappers finos invocables desde CI/cron
│   ├── run_pipeline.py
│   ├── predict_champion.py
│   └── deploy_prefect.py
├── src/worldcup2026/
│   ├── config.py               # PATHS, hiperparámetros, anfitriones 2026 y contendientes
│   ├── logging_utils.py
│   ├── cli.py                  # CLI Typer (`wc2026 …`)
│   ├── data/                   # ingest.py, preprocess.py, validate.py
│   ├── features/               # elo.py, build_features.py
│   ├── models/                 # train.py (MLflow + CV + staging), predict.py
│   ├── simulation/tournament.py
│   ├── pipeline/flow.py        # flow de Prefect
│   ├── monitoring/             # drift.py (PSI) + metrics.py (Prometheus)
│   └── api/app.py              # servicio FastAPI
├── tests/                      # pytest + TestClient de FastAPI (unit/ + integration/)
├── data/                       # raw / processed / external (gitignored)
├── models_store/               # artefacto joblib persistido
└── mlruns/                     # almacén file-backed de MLflow
```

---

## 2. Fuentes de datos (todas públicas y determinísticas)

| Dataset | URL | Rol |
|---|---|---|
| International Football Results (martj42) | `github.com/martj42/international_results` | Historia primaria de partidos 1872 → hoy |
| Fjelstul World Cup Database | `github.com/jfjelstul/worldcup` | Contexto relacional limpio de WC 1930-2022 |
| Ratings Elo calculados | derivados de los anteriores | Predictor longitudinal más robusto |
| Open-Meteo / Transfermarkt / StatsBomb | (enriquecedores opcionales) | Documentados en `config.py`, no conectados por defecto para mantener el baseline ligero y ejecutable sin red |

Todas las llamadas externas pasan por `src/worldcup2026/data/ingest.py`, que
cachea los CSV crudos bajo `data/raw/` — de modo que la segunda corrida es
**sin red**.

---

## 3. Inicio rápido — checklist de reproducibilidad

### 3.1 Clonar e instalar con `uv` (recomendado)

```bash
git clone <este-repo>
cd MLOps-WorldCup2026

# Con uv (compatible con PEP 723):
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3.2 …o con `pip` tradicional

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -e ".[dev]"
```

### 3.3 Correr el **pipeline completo** en un solo comando

```bash
python scripts/run_pipeline.py
```

o equivalentemente vía la CLI:

```bash
wc2026 run-pipeline --simulations 2000
```

Lo que hace, de arriba abajo:

1. Descarga y cachea los CSV crudos desde GitHub.
2. Limpia/armoniza nombres de países, marca partidos WC y knockouts.
3. Replaya un algoritmo Elo estilo eloratings.net sobre ~50k partidos.
4. Arma la matriz de features (Elo, forma de 5 partidos, flag anfitrión, …).
5. **Entrena** un clasificador Gradient-Boosted con tracking completo en MLflow
   (parámetros, métricas, artefacto del modelo, dataset de entrada).
6. **Simula** por Montecarlo la llave de 48 equipos del 2026 N veces y escribe
   `data/processed/champion_probabilities.csv`.

Duración típica en un portátil: **~1 minuto** en la primera corrida (descargas
incluidas), **~20 segundos** en corridas posteriores gracias a los cachés
parquet.

### 3.4 Inspeccionar MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
# abre http://localhost:5000  →  experimento "worldcup2026"
```
uv run mlflow ui --backend-store-uri ./mlruns --port 5000
---

## 4. Cobertura de la rúbrica de peer review

### ✅ 1. Reproducibilidad
- Un `pip install -e .` o `uv pip install -e .` deja un entorno funcional.
- Toda corrida lleva semilla fija (`seed=42`); los cachés son archivos parquet
  deterministas.
- `python scripts/run_pipeline.py` ejecuta el pipeline completo sin intervención
  humana.
- Las fuentes son CSV estáticos en GitHub raw — no se requieren API keys.

### ✅ 2. Experiment Tracking (MLflow)
Implementado en `src/worldcup2026/models/train.py`:
- `mlflow.set_experiment("worldcup2026")`.
- Loguea **parámetros**: nombre de modelo, todos los hiperparámetros, fracción
  de test, conteos de filas.
- Loguea **métricas**: `accuracy`, `f1_macro`, `log_loss` y sus análogas en
  cross-validation temporal (`cv_accuracy`, `cv_f1_macro`, `cv_log_loss`,
  `cv_log_loss_std`).
- Loguea **artefactos**: el pipeline scikit-learn (`mlflow.sklearn.log_model`),
  el dump joblib y el dataset parquet de entrenamiento.
- Registra el modelo en el Model Registry como `wc2026_match_classifier` y
  opcionalmente lo promueve a `Staging` o `Production` con tags por algoritmo.

### ✅ 3. Pipeline / Orquestación (Prefect)
`src/worldcup2026/pipeline/flow.py` define un flow
(`worldcup2026_pipeline`) con siete tareas: `ingest`, `preprocess`,
`validate_data`, `elo`, `features`, `train`, `simulate`. Cada paso es un
módulo aislado — se pueden correr individualmente:

```bash
wc2026 ingest
wc2026 preprocess
wc2026 build-elo
wc2026 features
wc2026 train --model gbt
wc2026 simulate --n 2000
```

Para registrar el flow con cron semanal:

```bash
python scripts/deploy_prefect.py
# (usa CronSchedule "0 6 * * MON", zona horaria America/Bogota)
```

### ✅ 4. Deployment
Tres modos de consumo vienen listos:

**a) Servicio FastAPI** — `src/worldcup2026/api/app.py`
```bash
uvicorn worldcup2026.api.app:app --reload --port 8000
# o:
docker compose up api
```
Endpoints:
- `GET /health` — liveness probe.
- `GET /ready` — verifica que haya artefacto en disco.
- `POST /predict/match` — `{home, away, neutral, is_knockout}` → probabilidades por clase.
- `POST /predict/tournament` — corre un Monte Carlo pequeño bajo demanda.
- `GET /champion_probabilities` — salida batch de la última corrida del pipeline.
- `GET /metrics` — métricas Prometheus (latencia + contadores).

**b) Predicciones batch (CSV)**
```bash
python scripts/predict_champion.py --simulations 5000
# escribe data/processed/champion_probabilities.csv
```

**c) CLI (`typer`)**
```bash
wc2026 predict "Argentina" "France" --neutral
```

**d) Docker**
```bash
docker build -t worldcup2026 .
docker run -p 8000:8000 worldcup2026
# o para MLflow + API juntos:
docker compose up
```

### ✅ 5. Monitoreo (Fase 5 — Diseño)
`src/worldcup2026/monitoring/` contiene el diseño y scripts base:

- **Data drift** — `monitoring/drift.py` compara la distribución de Elo y
  forma reciente entre el dataset de entrenamiento y los datos nuevos usando
  un **Population Stability Index (PSI)** por feature.
- **Concept drift** — tras cada ventana FIFA se comparan las probabilidades
  predichas contra los resultados reales (`brier_score`, `log_loss` móvil de
  30 días).
- **Operacional** — latencia de `/predict/match`, tasa de errores 5xx, uso de
  CPU/RAM del contenedor Docker.
- **Transporte** — las métricas se exponen en formato Prometheus en
  `GET /metrics` (listo para scraping). Dashboard propuesto: **Grafana** con 3
  paneles: PSI por feature, Brier score por mes, p95 de latencia del API.
- **Alertas sugeridas** — si `PSI > 0.25` en cualquier feature → retrain; si
  `brier_score` sube 20 % semana a semana → page al ML Engineer.

Ejecutar el chequeo de drift:
```bash
python -m worldcup2026.monitoring.drift --reference data/processed/training_dataset.parquet
```

### ✅ 6. Calidad de código y documentación
- Organización en 7 sub-paquetes enfocados (ver árbol arriba).
- `ruff` + `black` configurados en `pyproject.toml`:
  ```bash
  ruff check src tests
  black --check src tests
  ```
- Suite `pytest` bajo `tests/unit/` cubriendo Elo, features, tournament, API,
  validate y drift:
  ```bash
  pytest -q
  ```
- `.pre-commit-config.yaml` configurado (ruff + black + trailing-whitespace):
  ```bash
  pre-commit install && pre-commit run --all-files
  ```
- CI en `.github/workflows/ci.yml` — corre `ruff`, `black --check` y `pytest`
  en cada push/PR, más `docker build` como smoke test.
- CLI Typer con help en cada comando: `wc2026 --help`.
- Guías adicionales en `docs/`: `deployment.md`, `monitoring.md`.

---

## 5. Respondiendo "¿Quién ganará el Mundial 2026?"

El simulador de torneo en `src/worldcup2026/simulation/tournament.py` replaya
la llave de 48 equipos N veces. Cada resultado de partido se muestrea del
vector de probabilidades del clasificador; los empates en fase de eliminación
se deciden con una moneda cargada por el diferencial Elo (simula tanda de
penales). Tras `N` simulaciones se obtiene una tabla ranqueada de
probabilidades de campeonato. Un top-10 típico se ve así:

| equipo     | prob_campeón | prob_final | prob_semifinal |
|------------|---------------|-------------|-----------------|
| Argentina  | 0.138         | 0.24        | 0.38            |
| Francia    | 0.124         | 0.22        | 0.37            |
| Brasil     | 0.111         | 0.20        | 0.34            |
| España     | 0.094         | 0.18        | 0.31            |
| Inglaterra | 0.088         | 0.17        | 0.29            |
| …          | …             | …           | …               |

Los números dependen de la semilla, el modelo y la lista de equipos
clasificados — ajusta `WORLDCUP_2026_CONTENDERS` en `config.py` una vez se
confirme la lista oficial de 48.

---

## 6. Reproducir una corrida específica del peer review

Copia estos comandos tal cual al formulario del peer review:

```bash
# 1. Clonar
git clone <este-repo> && cd MLOps-WorldCup2026

# 2. Instalar
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Correr el pipeline completo
python scripts/run_pipeline.py

# 4. Inspeccionar MLflow
mlflow ui --backend-store-uri ./mlruns

# 5. Levantar el API
uvicorn worldcup2026.api.app:app --reload

# 6. Pedirle una predicción de partido
curl -X POST http://localhost:8000/predict/match \
    -H "Content-Type: application/json" \
    -d '{"home":"Argentina","away":"France","neutral":true,"is_knockout":true}'

# 7. Tests y linter
pytest -q
ruff check src tests
```

---

## 7. Limitaciones conocidas

- La aproximación de la llave de 48 usa seeding serpenteado por Elo, no el
  sorteo oficial de la FIFA (que no se conoce hasta ~dic-2025).
- xG de StatsBomb/Understat está referenciado en la arquitectura pero no está
  cableado en el feature set por defecto para mantener el pipeline
  reproducible sin red.
- El clasificador es un baseline — cambiar `gbt` por `xgb` (o correr un
  barrido) es cuestión de una línea vía MLflow.

## 8. Licencia

MIT. Ver [`pyproject.toml`](pyproject.toml).
