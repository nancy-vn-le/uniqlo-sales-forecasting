# Uniqlo Sydney CBD - Sales Forecasting System
## Project Report

---

## 1. Executive Summary

This project builds a three-layer retail sales forecasting system for a synthetic Uniqlo Sydney CBD flagship store. It was developed as a data science portfolio piece to demonstrate production-quality skills across the full data lifecycle: synthetic data generation, SQL analytics, time series forecasting with uncertainty quantification, SHAP explainability, and an interactive Streamlit dashboard.

The dataset covers two full years (2024-2025), 731 trading days, 21 merchandise divisions, 151 SKUs, 80,000 customer profiles, and 1.5 million transactions. The three-layer architecture separates point forecasts (Layer 1), confidence intervals (Layer 2), and rolling bias detection (Layer 3) into distinct, independently auditable components.

---

## 2. Project Background

**Why synthetic data?**  
Real store transaction data is commercially sensitive and not available for public portfolio work. A synthetic dataset lets us demonstrate exactly the same analytical techniques while controlling the data generating process so ground-truth relationships can be verified. All dollar amounts are scaled to be clearly distinct from any real Uniqlo store's reported figures.

**Store profile (simulated)**  
- Location: Sydney CBD, New South Wales, Australia
- Format: Multi-level flagship, trading 7 days a week
- Trading hours: 9 hours (Mon-Wed, Fri-Sat), 12 hours (Thursday late night), 7 hours (Sunday)
- Customer mix: loyalty members, casual shoppers, international tourists

**Tech stack**  
Python 3.12 (Anaconda), SQLite + SQLAlchemy, XGBoost 2.1.3, SHAP 0.46.0, Prophet, ARIMA (statsmodels), Streamlit, Plotly.

---

## 3. Dataset Overview

| Table | Rows | Description |
|---|---|---|
| `daily_sales` | 731 | One row per trading day, store-level totals |
| `division_daily` | 15,351 | Sales per division per day (21 divisions x 731 days) |
| `transactions` | 1,501,669 | Individual customer transactions |
| `transaction_items` | 3,393,425 | Line items within each transaction |
| `products` | 151 | SKU catalogue with price tiers and division mapping |
| `customers` | 80,000 | Customer profiles with segment and demographic flags |
| `inventory` | 15,855 | Weekly stock-on-hand and replenishment records |

**Total store revenue (2024-2025):** $130.73M  
**Average gross profit ratio:** 62.1%  
**Date range:** 2024-01-01 to 2025-12-31  
**Price range:** $13 (entry basics) to $199 (premium outerwear)  
**Average unit price across all SKUs:** $49.71

---

## 4. Data Generation Methodology

### 4.1 Temperature Curve (Southern Hemisphere)

Sydney's climate was modelled as a cosine curve calibrated to the Southern Hemisphere, where January is the hottest month (~28°C) and July is the coldest (~13°C):

```
baseline_temp = 20.5 + 7.5 * cos(2*pi * (day_of_year - 15) / 365)
```

Daily noise (+/- 3°C) was added to simulate weather variation.

### 4.2 Daily Sales Targets

Base targets by day type, reflecting the higher footfall pattern of a CBD location:

| Day | Base target range | Event-inflated target |
|---|---|---|
| Mon-Wed | $126K - $140K | up to ~$225K (Arigato) |
| Thursday | $175K - $224K | up to ~$310K (Arigato) |
| Friday | $154K - $168K | up to ~$240K (Arigato) |
| Saturday | $175K - $238K | up to ~$350K (Arigato) |
| Sunday | $168K - $196K | up to ~$290K (Arigato) |

Achievement: 90% of days reach at least 100% of target (1.00-1.12x), the remaining 10% fall short (0.84-0.99x).

### 4.3 Event Calendar

Six event categories were defined based on known Sydney retail and Uniqlo-specific trading patterns:

| Event | Days/year | Avg daily sales | Lift vs normal |
|---|---|---|---|
| Arigato Sale | 14 | $269,493 | +57% |
| Christmas | 8 | $249,066 | +45% |
| EOFY (End of Financial Year) | 52 | $211,509 | +23% |
| Long Weekend | 22 | $200,691 | +17% |
| Vivid Sydney | 24 | $198,381 | +16% |
| No event | 611 | $171,510 | - |

Vivid Sydney activates only when temperature is above 14°C (nights warm enough for outdoor attendance).

### 4.4 Division Seasonal Weight Curves

Each division receives a share of the day's total sales via a seasonal weight function driven by temperature and time of year. Key behaviours:

- **Cut & Sewn (Men's div 24, Women's div 34):** Flat year-round, largest revenue share at ~15.8% each
- **Outerwear (div 21, 31):** Near-zero in summer, peaks when temp drops below 15°C; minimum tourist floor of 18% applied (tourists from cold climates buy winter items even in Sydney summer)
- **Women's Dress (div 29):** Spring-Summer weighted, peaks above 22°C; minimum tourist floor of 12% applied
- **Inner/Heattech (div 27, 37):** Winter-weighted with a gradual autumn ramp
- **Knits (div 25, 35):** Autumn-Winter weighted
- **Kids, Accessories, Homewear:** Relatively flat throughout the year

### 4.5 Customer Generation

80,000 customer profiles were created across three segments:

| Segment | Share | Avg transaction | Notes |
|---|---|---|---|
| Loyalty | 43% | $120.63 | Registered members, higher frequency |
| Tourist | 29% | $120.60 | International visitors, higher basket in winter items |
| Casual | 28% | $120.56 | Walk-in shoppers |

Tourist volume is weighted higher during December-January (peak inbound travel) and event weeks.

---

## 5. Exploratory Findings

### 5.1 Day-of-Week Revenue Pattern

| Day | Avg daily sales | Avg customers | Avg achievement |
|---|---|---|---|
| Saturday | $227,564 | 2,408 | 104.9% |
| Thursday | $215,060 | 2,299 | 104.2% |
| Sunday | $197,033 | 2,097 | 104.0% |
| Friday | $178,623 | 2,215 | 106.1% |
| Tuesday | $145,718 | 1,800 | 105.5% |
| Wednesday | $144,455 | 1,787 | 104.9% |
| Monday | $144,416 | 1,782 | 104.8% |

Saturday is the top trading day. Thursday outperforms Friday despite fewer trading hours because extended hours (9am-9pm) attract after-work shoppers. All days consistently exceed their daily target by ~4-6%, reflecting a well-performing flagship.

### 5.2 Seasonal Division Patterns

Seasonal sensitivity is most visible in temperature-driven categories:

| Division | Winter (Jun-Aug) | Summer (Dec-Feb) | Ratio |
|---|---|---|---|
| Men's Outerwear | $5,891/day | $1,330/day | 4.4x |
| Women's Outerwear | $5,891/day | $1,330/day | 4.4x |
| Women's Dress | $924/day | $4,403/day | 4.8x (inverted) |
| Men's Cut & Sewn | $26,406/day | $28,622/day | flat |

The Outerwear tourist floor is visible: even in Sydney summer, Outerwear averages $1,330/day because international visitors from cold climates purchase items for their home-country seasons.

### 5.3 Revenue Mix by Division

The top 10 divisions account for 78% of total revenue:

| Division | 2-year total | Share |
|---|---|---|
| Women's Cut & Sewn | $20.69M | 15.8% |
| Men's Cut & Sewn | $20.69M | 15.8% |
| Women's Bottoms | $8.01M | 6.1% |
| Men's Bottoms | $8.01M | 6.1% |
| Women's Inner | $6.54M | 5.0% |
| Men's Inner | $6.54M | 5.0% |
| Women's Shirt | $6.01M | 4.6% |
| Men's Shirt | $6.01M | 4.6% |
| Women's Knit | $5.92M | 4.5% |
| Men's Knit | $5.92M | 4.5% |

Cut & Sewn (basics - t-shirts, polo, long sleeve) dominates as the highest-frequency, year-round replenishment category for all customer types.

### 5.4 Temperature and Sales

| Temperature band | Days | Avg daily sales |
|---|---|---|
| Cold (<15°C) | 172 | $186,242 |
| Mild (15-22°C) | 240 | $179,219 |
| Warm (>22°C) | 319 | $174,568 |

Counter-intuitively, cold days drive slightly higher total store revenue. This is because colder weather brings in Outerwear, Knitwear, and Heattech shoppers with higher basket values, while warm-weather categories (Dress, lightweight basics) have lower average prices.

---

## 6. Three-Layer Forecasting Architecture

### Layer 1 - Point Forecast

Three models were trained on the `division_daily` table at the division x day level:

**ARIMA** (baseline)  
Captures autocorrelation and trend in the sales time series for a single division. Used as a statistical baseline to compare against machine learning approaches.

**Prophet** (seasonal baseline)  
Facebook Prophet with weekly and annual seasonality plus event regressors (Arigato, Christmas, EOFY, Vivid, Long Weekend dummies). Better than ARIMA at capturing multi-period seasonality.

**XGBoost** (primary model)  
Gradient boosted trees on the full feature set. Selected as the primary model because it incorporates cross-division patterns, temperature, and lag features simultaneously.

Feature set used by XGBoost:

| Feature | Description |
|---|---|
| `division_code` | Which merchandise division |
| `temperature_index` | Sydney daily temperature (°C) |
| `trading_hours` | 7, 9, or 12 depending on day type |
| `is_public_holiday` | Binary flag |
| `promo_flag` | Promotional activity on non-event days |
| `dow_num` | Day of week (0=Monday) |
| `month` | Calendar month |
| `day_of_year` | 1-365, captures annual seasonality |
| `event_Arigato` ... `event_LongWeekend` | One-hot event flags |
| `lag_7`, `lag_14`, `lag_28` | Sales same division, 1/2/4 weeks ago |
| `rolling_7_mean`, `rolling_14_mean` | Rolling average sales past 7/14 days |

### Layer 2 - Confidence Intervals

70% confidence intervals are generated using XGBoost quantile regression:

- Lower bound: XGBoost trained with `objective="reg:quantileerror"`, `quantile_alpha=0.15`
- Upper bound: XGBoost trained with `quantile_alpha=0.85`

A reliability flag (HIGH / MEDIUM / LOW) is assigned per forecast based on historical Mean Absolute Percentage Error (MAPE) for that combination of day-of-week and event type:

| Reliability | MAPE threshold |
|---|---|
| HIGH | MAPE < 5% |
| MEDIUM | 5% <= MAPE < 12% |
| LOW | MAPE >= 12% or event week |

### Layer 3 - Rolling Bias Self-Monitor

The model is run in walk-forward validation (train on weeks 1-n, predict week n+1, store predicted vs actual) over 549 evaluation dates, producing 11,529 forecast log rows. A rolling N-week mean error is computed per division x day-of-week combination. Any context window showing |rolling bias| > 5% for 4 or more consecutive weeks is flagged.

---

## 7. Model Performance

### 7.1 Accuracy by Context (MAPE)

Accuracy varies significantly by day type and event. The model is most reliable on structured event days (Arigato, Christmas) because the training data provides clear signal. It struggles more with opportunistic events (EOFY, Vivid) where the sales lift is more variable.

Selected results (MAPE lower is better):

| Day | Event | MAPE |
|---|---|---|
| Thursday | Arigato | 8.2% |
| Thursday | Christmas | 7.4% |
| Thursday | EOFY | 14.0% |
| Saturday | Arigato | 9.3% |
| Saturday | EOFY | 13.6% |
| Tuesday | Vivid | 27.0% |
| Wednesday | Vivid | 20.3% |

A MAPE of 8-14% on daily division-level forecasts is consistent with published accuracy benchmarks for fashion retail forecasting, where style turnover and event unpredictability are known to limit forecast precision.

### 7.2 Active Bias Flags (Layer 3)

At the end of the evaluation period, Layer 3 detected 5 systematic bias patterns:

| Division | Day | Direction | Consecutive weeks | Bias |
|---|---|---|---|---|
| Men's Outerwear | Saturday | Overestimating | 21 weeks | -5.5% |
| Women's Dress | Thursday | Underestimating | 10 weeks | +10.2% |
| Men's Outerwear | Wednesday | Overestimating | 5 weeks | -8.0% |
| Women's Dress | Sunday | Underestimating | 5 weeks | +6.3% |
| Women's Outerwear | Wednesday | Overestimating | 4 weeks | -6.6% |

The Outerwear overestimation on weekends and midweek reflects the model smoothing temperature effects too aggressively. When temperatures rise faster than the model expects in spring, Outerwear drops faster than predicted. The Women's Dress underestimation on Thursdays points to an emerging demand pattern not yet fully captured by the lag features.

These flags are surfaced in the dashboard's Model Self-Check page so the forecasting team can investigate before a bias compounds into significant stock or staffing errors.

---

## 8. SHAP Driver Attribution

SHAP (SHapley Additive exPlanations) TreeExplainer values are computed for every forecast on the Division Forecast page. Each prediction is decomposed into:

- **Base value:** the model's average prediction across all divisions and contexts
- **Division contribution:** how much this specific category typically deviates from the store average
- **Temperature contribution:** the temperature's push or pull on demand (negative for Outerwear in warm weather, positive for Dress)
- **Event contribution:** the lift from Arigato, Christmas, or other event flags
- **Lag/rolling contribution:** whether recent sales trend is above or below average
- **Day-of-week contribution:** the structural pattern (Saturday premium, Monday baseline)

This makes each forecast auditable: a store manager can see exactly why the model predicts $28,000 for Men's Outerwear on a cold Thursday versus $8,000 on a warm Monday.

---

## 9. Dashboard Pages

The dashboard is built in Streamlit and connects directly to the SQLite database at runtime. No CSV files are read. All queries use `pd.read_sql()` to ensure the dashboard reflects the current database state.

**Page 1 - Overview**  
Store-level KPI cards (total revenue, GP%, average daily customers, YoY growth), daily and weekly sales trend with Plotly range slider, and division revenue share chart. A time range selector (Last 30 days / Last 90 days / Last 6 months / All time) persists across chart refreshes.

**Page 2 - Division Forecast**  
A "scenario planner" interface. The user selects a division and a target month, and the model provides a forecast for a representative day in that month. Temperature is pre-suggested based on Sydney's seasonal curve for that month. What-if sliders allow overriding temperature and event type to explore scenarios. SHAP waterfall chart shows driver attribution for each forecast.

**Page 3 - Forecast Confidence**  
70% confidence interval band chart (Layer 2 quantile bounds), reliability flag (HIGH/MEDIUM/LOW) with plain-language explanation, and a historical accuracy table showing MAPE by day-of-week and event context for the selected division.

**Page 4 - Model Self-Check**  
Rolling forecast bias chart per division over time, with flagged weeks highlighted. Active bias alert cards are surfaced at the top for any context window exceeding 5% systematic error for 4+ consecutive weeks. This page answers the question: "Is the model still accurate, or is it drifting?"

---

## 10. Key Design Decisions

**Single Python environment.** All data generation, model training, and dashboard serving run under the same Anaconda Python 3.12 binary. This eliminates the class of bug where a model trained by one Python/XGBoost version gives silently wrong predictions when loaded by a different version.

**XGBoost native format (.ubj).** Models are saved in XGBoost's binary format rather than pickle. This makes them portable across minor Python versions and avoids pickle security and compatibility issues.

**SHAP explainer recreated at runtime.** The SHAP TreeExplainer is not pickled; it is recreated from the loaded XGBoost model at dashboard startup. Only the expected value scalar is saved to disk. This avoids version-mismatch errors between SHAP and XGBoost.

**No double event multiplier.** A common synthetic data pitfall is applying an event lift twice: once to the planning target and again to actual sales. This project applies event multipliers only to the base daily target; the achievement percentage (whether the store beats or misses that target) is the only additional factor. This keeps event-day sales in plausible ranges.

**Tourist sales floor.** Outerwear and Women's Dress maintain minimum sales floors regardless of temperature (18% base for Outerwear, 12% for Dress). This reflects the reality that tourists from cold climates purchase winter items in Sydney summer, and lightweight categories retain some demand in colder months. Without this floor, these divisions would show $0 sales in off-season months, which is not realistic for an international flagship.

---

## 11. How to Reproduce

**Requirements:** Anaconda Python 3.12, all packages in `requirements.txt`.

```bash
# Step 1 - Install dependencies
pip install -r requirements.txt

# Step 2 - Generate synthetic data (~2 minutes)
python src/data_generation/generate_data.py

# Step 3 - Load into SQLite database
PYTHONPATH=. python src/data_generation/load_db.py

# Step 4 - Train all models (~3-5 minutes)
PYTHONPATH=. python src/train_models.py

# Step 5 - Launch dashboard
PYTHONPATH=. streamlit run dashboard/app.py
```

Model artifacts are saved to `src/models/`. The database is created at `data/uniqlo.db`. Both are excluded from version control (see `.gitignore`) and must be regenerated locally.

---

## 12. Repository Structure

```
project-retail-uniqlo/
- data/raw/               Regenerated CSVs (daily_sales, division_daily, etc.)
- sql/                    Portfolio SQL files (schema, EDA, window functions, RFM)
- notebooks/              5 Jupyter notebooks (EDA, segmentation, 3 forecast layers)
- src/
  - data_generation/      generate_data.py, load_db.py
  - features/             Feature engineering
  - models/               Trained model artifacts
  - utils/                db_connect.py
- dashboard/
  - app.py                Streamlit application (4 pages)
- docs/
  - project_report.md     This document
- requirements.txt
- README.md
- project_brief.md        Full data specification and design decisions
```

---

## 13. Limitations and Future Work

**Synthetic data constraints.** Because the data is generated from parametric rules, the model is in some sense fitting to its own data-generating process. In a real deployment, model performance would depend on how well the feature set captures actual demand drivers not present in historical data (weather forecasts, competitor promotions, social media trends).

**Single-store scope.** The architecture forecasts at the division level for one store. A natural extension is multi-store hierarchy forecasting with store-level fixed effects.

**No stockout modelling.** The inventory table records stock-on-hand but the forecasting models do not incorporate stockout risk as a feature. In practice, out-of-stock events suppress sales in ways that corrupt the demand signal.

**Retraining cadence.** The bias detection layer (Layer 3) identifies drift but the current implementation does not trigger automatic retraining. A production system would wire the bias flags to a retraining job, with human review before the new model is promoted to serving.

**ARIMA and Prophet as reference only.** ARIMA and Prophet are trained and evaluated in the notebooks but XGBoost is the sole serving model in the dashboard. A rigorous ensemble or model selection step would be a straightforward extension.
