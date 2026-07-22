"""
Run all model training logic from notebooks 01–05.
Saves artifacts to src/models/ so the Streamlit dashboard is fully functional.

Usage:
    python src/train_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.db_connect import get_engine
from src.features.feature_engineering import build_features

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

engine = get_engine()

FEATURE_COLS = [
    "division_code", "temperature_index", "trading_hours",
    "is_public_holiday", "promo_flag", "dow_num", "month", "day_of_year",
    "event_Arigato", "event_Christmas", "event_EOFY", "event_Vivid", "event_LongWeekend",
    "lag_7", "lag_14", "lag_28", "rolling_7_mean", "rolling_14_mean",
]

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

def load_data():
    print("Loading data from SQLite...")
    df = pd.read_sql("""
        SELECT dd.date, dd.division_code, dd.division_name, dd.department,
               dd.sales_amt, dd.gp_amt, dd.gp_ratio, dd.num_transactions,
               ds.day_of_week, ds.event_flag, ds.temperature_index,
               ds.is_public_holiday, ds.trading_hours, ds.promo_flag
        FROM division_daily dd
        JOIN daily_sales ds ON dd.date = ds.date
        ORDER BY dd.date, dd.division_code
    """, engine, parse_dates=["date"])
    df = build_features(df)
    print(f"  {len(df):,} rows | {df['division_code'].nunique()} divisions | {df['date'].nunique()} days")
    return df


def train_test_split(df, test_weeks=8):
    split_date = df["date"].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df["date"] <= split_date].copy()
    test  = df[df["date"] >  split_date].copy()
    print(f"  Train: {train['date'].min().date()} → {train['date'].max().date()} ({len(train):,} rows)")
    print(f"  Test:  {test['date'].min().date()} → {test['date'].max().date()} ({len(test):,} rows)")
    return train, test


# ---------------------------------------------------------------------------
# 2. ARIMA baseline (demo division only — saved separately)
# ---------------------------------------------------------------------------

def train_arima(train, test, demo_div=34):
    from statsmodels.tsa.arima.model import ARIMA
    print(f"\n[Layer 1] ARIMA — Div {demo_div} baseline...")
    div_train = train[train["division_code"] == demo_div].set_index("date")["sales_amt"]
    div_test  = test[test["division_code"]   == demo_div].set_index("date")["sales_amt"]
    model = ARIMA(div_train, order=(7, 1, 1))
    fit = model.fit()
    pred = fit.forecast(steps=len(div_test))
    pred.index = div_test.index
    mape = np.mean(np.abs((div_test - pred) / div_test)) * 100
    rmse = np.sqrt(np.mean((div_test - pred) ** 2))
    print(f"  ARIMA MAPE: {mape:.2f}%  RMSE: ${rmse:,.0f}")
    with open(MODEL_DIR / f"arima_div{demo_div}.pkl", "wb") as f:
        pickle.dump(fit, f)
    return mape, rmse


# ---------------------------------------------------------------------------
# 3. Prophet (per division, event regressors)
# ---------------------------------------------------------------------------

def train_prophet(train, test, demo_div=34):
    from prophet import Prophet
    print(f"\n[Layer 1] Prophet — Div {demo_div}...")

    EVENT_COLS = ["Arigato", "Christmas", "EOFY", "Vivid", "LongWeekend"]

    def make_prophet_df(src, div_code):
        d = src[src["division_code"] == div_code][["date", "sales_amt", "event_flag"]].copy()
        d = d.rename(columns={"date": "ds", "sales_amt": "y"})
        for ev in EVENT_COLS:
            d[ev] = (d["event_flag"] == ev).astype(int)
        return d

    d_train = make_prophet_df(train, demo_div)
    d_test  = make_prophet_df(test,  demo_div)

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, interval_width=0.70)
    for ev in EVENT_COLS:
        m.add_regressor(ev)
    m.fit(d_train)

    future = d_test[["ds"] + EVENT_COLS]
    forecast = m.predict(future)
    pred = forecast.set_index("ds")["yhat"]
    actual = d_test.set_index("ds")["y"]
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    print(f"  Prophet MAPE: {mape:.2f}%  RMSE: ${rmse:,.0f}")

    with open(MODEL_DIR / f"prophet_div{demo_div}.pkl", "wb") as f:
        pickle.dump(m, f)

    return m, mape, rmse


# ---------------------------------------------------------------------------
# 4. XGBoost point forecast (all divisions)
# ---------------------------------------------------------------------------

def train_xgb_point(train, test):
    import xgboost as xgb
    print("\n[Layer 1] XGBoost point forecast (all divisions)...")

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["sales_amt"]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test["sales_amt"]

    # Simple 85/15 internal val split for early stopping
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42,
        early_stopping_rounds=20, eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

    pred = model.predict(X_test)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    print(f"  XGBoost MAPE: {mape:.2f}%  RMSE: ${rmse:,.0f}")

    # Save in XGBoost native format so it loads correctly in any Python version.
    # pickle is not portable across Python/XGBoost version combinations.
    model.save_model(MODEL_DIR / "xgb_layer1.ubj")
    return model, X_train, y_train, X_test, y_test, mape, rmse


# ---------------------------------------------------------------------------
# 5. SHAP explainer
# ---------------------------------------------------------------------------

def compute_shap(model, X_test):
    import shap
    print("\n[Layer 1] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLS)
    print("  Top 5 features by |SHAP|:")
    print(mean_abs.sort_values(ascending=False).head(5).to_string())
    # Save expected_value separately; explainer itself is recreated at load time
    # from the model to avoid cross-Python pickle incompatibility.
    np.save(MODEL_DIR / "shap_expected_value.npy", np.array([float(explainer.expected_value)]))
    np.save(MODEL_DIR / "feature_cols.npy", np.array(FEATURE_COLS))
    print("  SHAP expected_value saved.")
    return explainer


# ---------------------------------------------------------------------------
# 6. Quantile regression (Layer 2 — 70% CI)
# ---------------------------------------------------------------------------

def train_quantile(X_train, y_train, X_test, y_test):
    import xgboost as xgb
    print("\n[Layer 2] XGBoost quantile regression (70% CI)...")

    results = {}
    for alpha, label in [(0.15, "lower"), (0.85, "upper")]:
        m = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            objective="reg:quantileerror", quantile_alpha=alpha,
            random_state=42, verbosity=0,
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[label] = (m, pred)
        m.save_model(MODEL_DIR / f"xgb_quantile_{label}.ubj")

    lower_pred = results["lower"][1]
    upper_pred = results["upper"][1]
    coverage = np.mean((y_test.values >= lower_pred) & (y_test.values <= upper_pred))
    print(f"  70% CI coverage on test set: {coverage:.1%} (target ~70%)")
    return results


# ---------------------------------------------------------------------------
# 7. Historical accuracy by context (Layer 2)
# ---------------------------------------------------------------------------

def build_accuracy_lookup(model, train, X_train):
    print("\n[Layer 2] Building accuracy lookup by context...")
    train_pred = model.predict(X_train.fillna(0))
    err_df = train.copy()
    err_df["predicted"] = train_pred
    err_df["abs_pct_error"] = (
        np.abs(err_df["sales_amt"] - err_df["predicted"]) / err_df["sales_amt"] * 100
    )
    lookup = (
        err_df.groupby(["day_of_week", "event_flag"])["abs_pct_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_pct_error": "mean_mape"})
    )
    lookup.to_csv(MODEL_DIR / "accuracy_lookup.csv", index=False)
    print(f"  Accuracy lookup saved ({len(lookup)} context rows).")
    print("  Highest-error contexts:")
    print(lookup.sort_values("mean_mape", ascending=False).head(5).to_string(index=False))
    return lookup


# ---------------------------------------------------------------------------
# 8. Rolling-window forecast log (Layer 3)
# ---------------------------------------------------------------------------

def build_forecast_log(df):
    import xgboost as xgb
    print("\n[Layer 3] Building rolling-window forecast log...")
    print("  (This will take several minutes for 2 years of daily predictions)")

    all_dates = sorted(df["date"].unique())
    min_train_date = all_dates[0] + pd.Timedelta(weeks=26)
    predict_dates = [d for d in all_dates if d >= min_train_date]
    print(f"  {len(predict_dates)} prediction dates (from {min_train_date.date()})")

    log_rows = []

    for i, target_date in enumerate(predict_dates):
        train_mask = df["date"] < target_date
        pred_mask  = df["date"] == target_date

        X_tr = df[train_mask][FEATURE_COLS].fillna(0)
        y_tr = df[train_mask]["sales_amt"]
        X_pr = df[pred_mask][FEATURE_COLS].fillna(0)
        y_pr = df[pred_mask]["sales_amt"]

        if len(X_pr) == 0 or len(X_tr) < 100:
            continue

        m = xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)

        m_lo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                                  objective="reg:quantileerror", quantile_alpha=0.15,
                                  random_state=42, verbosity=0)
        m_hi = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                                  objective="reg:quantileerror", quantile_alpha=0.85,
                                  random_state=42, verbosity=0)
        m_lo.fit(X_tr, y_tr)
        m_hi.fit(X_tr, y_tr)

        preds   = m.predict(X_pr)
        lowers  = m_lo.predict(X_pr)
        uppers  = m_hi.predict(X_pr)
        actuals = y_pr.values
        meta    = df[pred_mask][["division_code", "day_of_week", "event_flag"]].values

        fc_date = target_date - pd.Timedelta(days=7)

        for j in range(len(preds)):
            actual = float(actuals[j]) if j < len(actuals) else None
            err_pct = ((actual - preds[j]) / actual) if actual and actual != 0 else None
            log_rows.append({
                "forecast_date":   str(fc_date.date()),
                "target_date":     str(target_date.date()),
                "division_code":   int(meta[j][0]),
                "predicted_sales": round(float(preds[j]), 2),
                "lower_bound":     round(float(lowers[j]), 2),
                "upper_bound":     round(float(uppers[j]), 2),
                "actual_sales":    round(actual, 2) if actual else None,
                "error_pct":       round(float(err_pct), 4) if err_pct else None,
                "context_flags":   f"{meta[j][1]}/{meta[j][2]}",
            })

        if i % 50 == 0:
            print(f"  {i}/{len(predict_dates)} dates processed ({len(log_rows):,} log rows)")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(RAW_DIR / "forecast_log.csv", index=False)
    log_df.to_sql("forecast_log", engine, if_exists="replace", index=False, chunksize=5_000)
    print(f"  forecast_log: {len(log_df):,} rows saved.")
    return log_df


# ---------------------------------------------------------------------------
# 9. Bias detection (Layer 3)
# ---------------------------------------------------------------------------

def detect_bias(log_df):
    print("\n[Layer 3] Detecting systematic forecast biases...")
    BIAS_THRESHOLD = 5.0
    CONSECUTIVE = 4

    # Join division names
    div_names = pd.read_sql(
        "SELECT DISTINCT division_code, division_name FROM division_daily", engine
    )
    log_df = log_df.merge(div_names, on="division_code", how="left")

    # Parse day_of_week from context_flags (format: "Thursday/None")
    log_df["day_of_week"] = log_df["context_flags"].str.split("/").str[0]
    log_df["target_date"] = pd.to_datetime(log_df["target_date"])
    log_df["signed_error_pct"] = (
        (log_df["actual_sales"] - log_df["predicted_sales"])
        / log_df["actual_sales"] * 100
    )

    log_df = log_df.sort_values(["division_code", "day_of_week", "target_date"])
    log_df["rolling_6w_bias"] = (
        log_df.groupby(["division_code", "day_of_week"])["signed_error_pct"]
        .transform(lambda x: x.rolling(6, min_periods=4).mean())
    )

    def consec_flagged(series, thr):
        count = 0
        for v in reversed(series.dropna().values):
            if abs(v) > thr:
                count += 1
            else:
                break
        return count

    bias_flags = []
    for (div_code, dow), group in log_df.groupby(["division_code", "day_of_week"]):
        g = group.sort_values("target_date").dropna(subset=["rolling_6w_bias"])
        if len(g) < CONSECUTIVE:
            continue
        n = consec_flagged(g["rolling_6w_bias"], BIAS_THRESHOLD)
        if n >= CONSECUTIVE:
            latest = g["rolling_6w_bias"].iloc[-1]
            direction = "underestimating" if latest > 0 else "overestimating"
            div_name = g["division_name"].iloc[-1] if "division_name" in g.columns else str(div_code)
            bias_flags.append({
                "division_code": div_code,
                "division_name": div_name,
                "day_of_week": dow,
                "consecutive_flagged_periods": n,
                "latest_rolling_bias_pct": round(latest, 2),
                "direction": direction,
                "alert": (
                    f"Heads up: {n} consecutive weeks {direction} {dow} "
                    f"sales in {div_name} by ~{abs(latest):.1f}%."
                ),
            })

    bias_df = pd.DataFrame(bias_flags)
    bias_df.to_csv(MODEL_DIR / "bias_flags.csv", index=False)

    if len(bias_df):
        print(f"  {len(bias_df)} systematic bias flag(s) detected:")
        for _, row in bias_df.iterrows():
            print(f"    ⚠  {row['alert']}")
    else:
        print("  No systematic biases detected above threshold.")
    return bias_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Uniqlo Sales Forecasting — Model Training Pipeline")
    print("=" * 60)

    df = load_data()
    train, test = train_test_split(df)

    # Layer 1
    train_arima(train, test)
    train_prophet(train, test)
    xgb_model, X_train, y_train, X_test, y_test, xgb_mape, xgb_rmse = train_xgb_point(train, test)
    compute_shap(xgb_model, X_test)

    # Layer 2
    train_quantile(X_train, y_train, X_test, y_test)
    build_accuracy_lookup(xgb_model, train, X_train)

    # Layer 3
    log_df = build_forecast_log(df)
    detect_bias(log_df)

    print("\n" + "=" * 60)
    print("Training complete. Artifacts saved to src/models/:")
    for p in sorted(MODEL_DIR.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name}  ({size/1024:.1f} KB)")
    print("=" * 60)
    print("Dashboard will now show all 4 pages. Reload http://localhost:8503")
