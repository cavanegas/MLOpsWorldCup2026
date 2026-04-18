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

