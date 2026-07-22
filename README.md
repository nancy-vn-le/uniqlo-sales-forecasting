# Uniqlo Sydney CBD — Sales Forecasting System

> Built a three-layer sales forecasting system for a Sydney CBD flagship retail store using synthetic data
> calibrated to real trading patterns. Layer 1 forecasts division-level demand using Prophet/XGBoost with
> SHAP-attributed drivers (temperature, events, seasonality); Layer 2 quantifies forecast confidence and
> flags low-reliability periods; Layer 3 detects systematic forecast bias via rolling error tracking.
> Deployed an interactive Streamlit dashboard with live what-if scenario controls.
>
> — Viet Ngan Le

**Live Demo:** _[Link available after Streamlit Cloud deployment]_

---

## Three-Layer Architecture

What makes this project different from a standard forecasting repo:

| Layer | Question Answered | Output |
|---|---|---|
| **Layer 1 — Base Forecast** | What will happen? | Division-level daily sales prediction + SHAP driver breakdown |
| **Layer 2 — Uncertainty** | How sure are we? | 70% CI band + HIGH/MEDIUM/LOW reliability flag |
| **Layer 3 — Self-Monitoring** | Where has the model been wrong? | Rolling bias tracker — flags systematic errors automatically |

Most forecasting portfolios stop at Layer 1. Layers 2 and 3 demonstrate a model that knows its own uncertainty and catches its own recurring mistakes.

---

## Dashboard — 4 Pages

| Page | Content |
|---|---|
| **Overview** | KPI cards, daily sales trend, division revenue share, day-of-week distribution |
| **Division Forecast** | Division selector + what-if sliders (temperature, event toggle) → SHAP waterfall chart |
| **Forecast Confidence** | CI band chart, reliability flag (HIGH/MEDIUM/LOW), MAPE heatmap by context |
| **Model Self-Check** | Rolling bias chart per division, systematic error flags, error heatmap |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Database | SQLite (`data/uniqlo.db`) via SQLAlchemy — schema is PostgreSQL-compatible |
| Forecasting | statsmodels (ARIMA), Prophet, XGBoost, LightGBM (quantile regression) |
| Explainability | SHAP (XGBoost feature attribution — waterfall charts) |
| Segmentation | scikit-learn K-means + SQL RFM scoring |
| Visualisation | Plotly (all charts interactive) |
| Dashboard | Streamlit — deployed on Streamlit Cloud |
| Language | Python 3.11+ |

---

## How to Run Locally

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd project-retail-uniqlo
pip install -r requirements.txt

# 2. Generate synthetic datasets (takes ~5–10 min for full transaction tables)
python src/data_generation/generate_data.py

# 3. Load all CSVs into SQLite
python src/data_generation/load_db.py

# 4. Run notebooks in order (to train models and save artifacts)
# notebooks/01_eda.ipynb
# notebooks/02_segmentation_lite.ipynb
# notebooks/03_forecasting_layer1.ipynb    ← trains and saves XGBoost + SHAP
# notebooks/04_forecasting_layer2_uncertainty.ipynb
# notebooks/05_forecasting_layer3_selfmonitor.ipynb

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## Repository Structure

```
├── data/
│   ├── uniqlo.db                  # SQLite database
│   └── raw/                       # CSV exports
│       ├── daily_sales.csv        # 730 rows (2024–2025)
│       ├── division_daily.csv     # 14,600 rows (20 divisions × 730 days)
│       ├── transactions.csv       # ~1.5M rows
│       ├── transaction_items.csv  # ~3.3M rows
│       ├── customers.csv          # ~80,000 rows
│       ├── products.csv           # ~200 SKUs
│       ├── inventory.csv          # ~20,800 rows
│       └── forecast_log.csv       # Generated in Layer 3 notebook
├── sql/
│   ├── schema.sql                 # CREATE TABLE, indexes, FKs
│   ├── eda_queries.sql            # Aggregations, summaries
│   ├── window_functions.sql       # Rolling avg, rankings, running totals
│   ├── rfm_segmentation.sql       # Lightweight RFM scoring
│   └── division_performance.sql   # Division rankings, mix-shift queries
├── notebooks/
│   ├── 01_eda.ipynb               # SQL + Plotly — data foundation
│   ├── 02_segmentation_lite.ipynb # Lightweight RFM + K-means
│   ├── 03_forecasting_layer1.ipynb
│   ├── 04_forecasting_layer2_uncertainty.ipynb
│   └── 05_forecasting_layer3_selfmonitor.ipynb
├── src/
│   ├── data_generation/
│   │   ├── generate_data.py       # Synthetic data generator
│   │   └── load_db.py             # Load CSVs into SQLite
│   ├── features/
│   │   └── feature_engineering.py # Lag features, event encoding
│   ├── models/                    # Saved model artifacts (.pkl)
│   └── utils/
│       └── db_connect.py          # SQLAlchemy engine helper
├── dashboard/
│   └── app.py                     # Streamlit — 4 pages
├── docs/
│   └── project_brief.md           # Full domain knowledge document
├── requirements.txt
└── README.md
```

---

## Dataset Note

Synthetic dataset generated to reflect realistic Australian retail trading patterns for a CBD fashion retailer. Product taxonomy, pricing tiers, seasonal demand curves, and customer behaviour parameters are based on the author's professional retail experience and publicly available information. No confidential data is reproduced.
