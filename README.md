# ParkIQ v2 — AI-Driven Parking Congestion Intelligence

## Setup
```bash
pip install -r requirements.txt
python run_pipeline.py      # runs all 8 modules (~3–5 min)
streamlit run dashboard.py
```

## Architecture (8 Modules)

| Module | What it does |
|--------|-------------|
| 0 — Ingest & CIS | Parse dataset, compute Causal Impact Score (proximity × vehicle weight × time) |
| 1 — GRU Stress | Per-junction GRU neural net predicts next-hour violation stress (0–1) |
| 2 — Spillover Graph | NetworkX directed graph + PageRank centrality of congestion propagation |
| 3 — Enforcement Priority | Multi-factor priority score + 2-opt TSP patrol routing across 3 officers |
| 4 — XGBoost + SHAP | Attribution engine: links parking events to congestion % via SHAP explainability |
| 5 — Policy Simulator | What-if: restrict zone X by Y% → congestion delta |
| 6 — Anomaly + Hotspots | Isolation Forest anomaly detection + persistent ~100m grid hotspot scoring |
| 7 — Revenue & ROI | Fine revenue projection, congestion savings, patrol ROI, weekly shift scheduler |

## Dashboard Pages
- 📊 Overview — KPIs, hourly/weekday patterns, CIS heatmap
- 🗺️ Zone Heatmap — Violation density / priority / anomaly map tabs
- 🔮 Junction Stress — GRU predictions vs historical stress
- 🚨 Anomaly Detection — Isolation Forest flagged events
- 📍 Repeat Hotspots — Persistent grid-cell hotspot tiers (Critical/High/Medium/Low)
- 📈 Congestion Attribution — XGBoost + SHAP feature importance
- 🚔 Enforcement Routing — Multi-officer patrol map with 2-opt optimised routes
- 💰 Revenue & ROI — Fine revenue, congestion savings, strategic ROI quadrant
- 🗓️ Shift Scheduler — Weekly officer deployment calendar
- 🧪 Policy Simulator — Live what-if restriction testing
- 🔗 Spillover Graph — Junction congestion propagation network

## Tech Stack
Python · PyTorch (GRU) · XGBoost · SHAP · scikit-learn · GeoPandas · NetworkX  
Streamlit · Plotly · FastAPI-ready · PostgreSQL+PostGIS ready
