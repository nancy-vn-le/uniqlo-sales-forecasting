"""
Uniqlo Sydney CBD — Sales Forecasting Dashboard
4 pages: Overview | Division Forecast | Forecast Confidence | Model Self-Check
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings("ignore")

from src.utils.db_connect import get_engine
from src.features.feature_engineering import build_features

st.set_page_config(
    page_title="Uniqlo Sydney CBD — Sales Forecast",
    page_icon="🏪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Color palette — single source of truth for all chart colors
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":   "#378ADD",
    "target":    "#EF9F27",
    "positive":  "#5DCAA5",
    "negative":  "#E24B4A",
    "neutral":   "#9da3b4",
    "rolling":   "#A78BFA",
    "ci_fill":   "rgba(55,138,221,0.15)",
    "ci_line":   "rgba(55,138,221,0)",
    "grid":      "rgba(255,255,255,0.05)",
    "women":     "#E8A0BF",
    "men":       "#87BFEA",
    "kids":      "#9ED4A0",
    "others":    "#F5E08A",
    "card_bg":   "#161b27",
    "page_bg":   "#0e1117",
    "border":    "#252c3d",
    "text":      "#e4e8f0",
    "subtext":   "#9da3b4",
}

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
/* Page background */
.stApp {{ background-color: {PALETTE['page_bg']}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: #10151f;
    border-right: 1px solid {PALETTE['border']};
}}
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {PALETTE['subtext']};
    font-size: 0.8rem;
}}

/* Sidebar radio — selected item accent */
div[data-testid="stRadio"] label:has(input:checked) {{
    border-left: 3px solid {PALETTE['primary']} !important;
    padding-left: 10px !important;
    color: {PALETTE['primary']} !important;
    font-weight: 600;
}}
div[data-testid="stRadio"] label {{
    border-left: 3px solid transparent;
    padding-left: 10px;
    transition: border-color 0.15s, color 0.15s;
}}

/* Metric cards */
.metric-row {{
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
}}
.metric-card {{
    flex: 1;
    background: {PALETTE['card_bg']};
    border: 1px solid {PALETTE['border']};
    border-top: 3px solid var(--accent, {PALETTE['primary']});
    border-radius: 8px;
    padding: 16px 18px 14px;
    min-width: 0;
}}
.metric-label {{
    font-size: 0.72rem;
    color: {PALETTE['subtext']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.metric-value {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {PALETTE['text']};
    line-height: 1.15;
    white-space: nowrap;
}}
.metric-delta {{
    font-size: 0.8rem;
    margin-top: 6px;
    font-weight: 500;
}}
.delta-positive {{ color: {PALETTE['positive']}; }}
.delta-negative {{ color: {PALETTE['negative']}; }}
.delta-neutral  {{ color: {PALETTE['neutral']}; }}

/* Range selector row — collapse label gap */
div[data-testid="stRadio"] > label:first-child {{ display: none; }}

/* Horizontal radio pills */
div[data-testid="stRadio"] > div {{
    display: flex !important;
    flex-direction: row !important;
    gap: 6px;
    flex-wrap: wrap;
}}
div[data-testid="stRadio"] > div > label {{
    border: 1px solid {PALETTE['border']} !important;
    border-left: 1px solid {PALETTE['border']} !important;
    border-radius: 5px !important;
    padding: 3px 12px !important;
    font-size: 0.8rem !important;
    cursor: pointer;
    background: transparent;
    transition: background 0.15s;
}}
div[data-testid="stRadio"] > div > label:has(input:checked) {{
    background: {PALETTE['primary']}22 !important;
    border-color: {PALETTE['primary']} !important;
    color: {PALETTE['primary']} !important;
    font-weight: 600 !important;
    padding-left: 12px !important;
}}

/* Dividers */
hr {{ border-color: {PALETTE['border']}; margin: 18px 0; }}

/* Remove excess top padding */
.block-container {{ padding-top: 1.5rem !important; }}

/* Reliability flag chips */
.flag-chip {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 10px;
}}
.flag-high     {{ background: rgba(93,202,165,0.15); color: {PALETTE['positive']}; border: 1px solid {PALETTE['positive']}; }}
.flag-medium   {{ background: rgba(239,159,39,0.15);  color: {PALETTE['target']};   border: 1px solid {PALETTE['target']};   }}
.flag-low      {{ background: rgba(226,75,74,0.15);   color: {PALETTE['negative']}; border: 1px solid {PALETTE['negative']}; }}

/* Bias flag alerts */
.bias-alert {{
    background: rgba(226,75,74,0.08);
    border: 1px solid rgba(226,75,74,0.3);
    border-left: 4px solid {PALETTE['negative']};
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #e4e8f0;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly chart theme
# ---------------------------------------------------------------------------

_CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["subtext"], size=12),
    title_font=dict(color=PALETTE["text"], size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["border"],
                font=dict(color=PALETTE["subtext"])),
    hoverlabel=dict(bgcolor="#1a2035", bordercolor=PALETTE["border"],
                    font=dict(color=PALETTE["text"])),
    margin=dict(l=4, r=10, t=42, b=8),
    xaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["border"],
               tickfont=dict(color=PALETTE["subtext"]), zeroline=False),
    yaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["border"],
               tickfont=dict(color=PALETTE["subtext"]), zeroline=False),
)
_RANGESLIDER = dict(
    visible=True,
    thickness=0.055,
    bgcolor=PALETTE["card_bg"],
    bordercolor=PALETTE["border"],
    borderwidth=1,
)


def apply_theme(fig, height=400, rangeslider=True):
    fig.update_layout(**_CHART_BASE, height=height, hovermode="x unified")
    if rangeslider:
        fig.update_xaxes(rangeslider=_RANGESLIDER, type="date")
    return fig


# ---------------------------------------------------------------------------
# Time-range helpers
# ---------------------------------------------------------------------------

_RANGES = {"7D": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "All": None}


def time_range_selector(key: str) -> str:
    """Render horizontal range-pill radio; persists in session_state. Default 3M."""
    if key not in st.session_state:
        st.session_state[key] = "3M"
    result = st.radio(
        "Range",
        list(_RANGES.keys()),
        horizontal=True,
        index=list(_RANGES.keys()).index(st.session_state[key]),
        key=f"__tr_{key}",
        label_visibility="collapsed",
    )
    st.session_state[key] = result
    return result


def filter_range(df: pd.DataFrame, date_col: str, label: str) -> pd.DataFrame:
    days = _RANGES[label]
    if days is None:
        return df
    cutoff = df[date_col].max() - pd.Timedelta(days=days)
    return df[df[date_col] >= cutoff]


# ---------------------------------------------------------------------------
# Metric card HTML helpers
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, delta: str = "",
                 delta_dir: str = "neutral", accent: str = PALETTE["primary"]) -> str:
    icon = {"positive": "▲ ", "negative": "▼ "}.get(delta_dir, "")
    delta_html = (
        f'<div class="metric-delta delta-{delta_dir}">{icon}{delta}</div>' if delta else ""
    )
    return (
        f'<div class="metric-card" style="--accent:{accent};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f"{delta_html}</div>"
    )


def render_metric_row(cards: list[dict]):
    st.markdown(
        '<div class="metric-row">' + "".join(_metric_card(**c) for c in cards) + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent.parent / "src" / "models"
engine = get_engine()


@st.cache_data
def load_daily_sales():
    return pd.read_sql("SELECT * FROM daily_sales ORDER BY date",
                       engine, parse_dates=["date"])


@st.cache_data
def load_division_daily():
    return pd.read_sql("""
        SELECT dd.*, ds.day_of_week, ds.event_flag, ds.temperature_index,
               ds.is_public_holiday, ds.trading_hours, ds.promo_flag
        FROM division_daily dd
        JOIN daily_sales ds ON dd.date = ds.date
        ORDER BY dd.date, dd.division_code
    """, engine, parse_dates=["date"])


@st.cache_data
def load_forecast_log():
    try:
        return pd.read_sql("""
            SELECT fl.*, dd.division_name, ds.day_of_week AS dow
            FROM forecast_log fl
            JOIN division_daily dd
              ON fl.target_date = dd.date AND fl.division_code = dd.division_code
            JOIN daily_sales ds ON fl.target_date = ds.date
            WHERE fl.actual_sales IS NOT NULL
        """, engine, parse_dates=["target_date", "forecast_date"])
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_models():
    import xgboost as xgb
    import shap as _shap
    m = {}

    # XGBoost models: load via native format so predictions are correct
    # regardless of which Python version trained them.
    for name in ["xgb_layer1", "xgb_quantile_lower", "xgb_quantile_upper"]:
        p = MODEL_DIR / f"{name}.ubj"
        if p.exists():
            reg = xgb.XGBRegressor()
            reg.load_model(str(p))
            m[name] = reg

    # SHAP explainer: recreate from the loaded point-forecast model.
    # We never pickle it - TreeExplainer is rebuilt from the model's tree
    # structure, so results are identical to what was computed at training time.
    if "xgb_layer1" in m:
        ev_path = MODEL_DIR / "shap_expected_value.npy"
        explainer = _shap.TreeExplainer(m["xgb_layer1"])
        if ev_path.exists():
            # Override with the training-set expected_value for a stable baseline.
            explainer.expected_value = float(np.load(ev_path)[0])
        m["shap_explainer"] = explainer

    # Legacy pkl fallback for ARIMA / Prophet (these are not the broken models)
    for name in ["arima_div34", "prophet_div34"]:
        p = MODEL_DIR / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                m[name] = pickle.load(f)

    for name, fname in [("accuracy_lookup", "accuracy_lookup.csv"),
                         ("bias_flags", "bias_flags.csv")]:
        p = MODEL_DIR / fname
        if p.exists():
            m[name] = pd.read_csv(p)
    return m


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "division_code", "temperature_index", "trading_hours",
    "is_public_holiday", "promo_flag", "dow_num", "month", "day_of_year",
    "event_Arigato", "event_Christmas", "event_EOFY", "event_Vivid", "event_LongWeekend",
    "lag_7", "lag_14", "lag_28", "rolling_7_mean", "rolling_14_mean",
]

DIVISIONS = [
    (11, "Boys"), (12, "Girls"), (18, "Baby"),
    (21, "Women's Outerwear"), (22, "Women's Bottoms"), (23, "Women's Shirt"),
    (24, "Women's Cut & Sewn"), (25, "Women's Knit"), (26, "Women's Accessories"),
    (27, "Women's Inner"), (28, "Women's Loungewear"), (29, "Women's Dress"),
    (31, "Men's Outerwear"), (32, "Men's Bottoms"), (33, "Men's Shirt"),
    (34, "Men's Cut & Sewn"), (35, "Men's Knit"), (36, "Men's Accessories"),
    (37, "Men's Inner"), (38, "Men's Loungewear"), (88, "Miscellaneous"),
]
DIV_CODE_TO_NAME = dict(DIVISIONS)
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Sydney average temperature by month — drives what-if default
_SYDNEY_TEMP = {1:26, 2:26, 3:24, 4:20, 5:16, 6:13,
                7:12, 8:14, 9:16, 10:19, 11:22, 12:25}
# Mid-point day-of-year for each month (temporal context for model)
_MONTH_DOY   = {1:15, 2:46, 3:74, 4:105, 5:135, 6:166,
                7:196, 8:227, 9:258, 10:288, 11:319, 12:349}
_MONTH_NAMES = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"<h2 style='color:{PALETTE['text']};margin-bottom:2px'>Uniqlo Sydney CBD</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{PALETTE['subtext']};font-size:0.82rem;margin-top:0'>Sales Forecasting System</p>",
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Overview", "Division Forecast", "Forecast Confidence", "Model Self-Check"],
        index=0,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Synthetic dataset calibrated to real retail trading patterns.")

# ===========================================================================
# PAGE 1: Overview
# ===========================================================================

if page == "Overview":
    st.markdown(f"<h1 style='color:{PALETTE['text']}'>Store Overview</h1>", unsafe_allow_html=True)

    daily = load_daily_sales()

    # ── KPI computation ──────────────────────────────────────────────────────
    d24 = daily[daily["date"].dt.year == 2024]
    d25 = daily[daily["date"].dt.year == 2025]

    total_sales_24 = d24["actual_sales"].sum()
    total_sales_25 = d25["actual_sales"].sum()
    total_sales    = daily["actual_sales"].sum()

    avg_gp_24 = d24["gross_profit_ratio"].mean()
    avg_gp_25 = d25["gross_profit_ratio"].mean()

    avg_cust_24 = d24["num_customers"].mean()
    avg_cust_25 = d25["num_customers"].mean()

    yoy_avg = d25["yoy_pct"].mean()
    yoy_pct_change = (total_sales_25 - total_sales_24) / total_sales_24 * 100

    gp_delta = (avg_gp_25 - avg_gp_24) * 100
    cust_delta = (avg_cust_25 - avg_cust_24) / avg_cust_24 * 100

    render_metric_row([
        dict(label="Total Revenue (2024–2025)", value=f"${total_sales/1e6:.1f}M",
             delta=f"+{yoy_pct_change:.1f}% YoY", delta_dir="positive",
             accent=PALETTE["primary"]),
        dict(label="Avg Gross Profit Ratio", value=f"{avg_gp_25*100:.1f}%",
             delta=f"{gp_delta:+.2f}pp vs 2024",
             delta_dir="positive" if gp_delta >= 0 else "negative",
             accent=PALETTE["positive"]),
        dict(label="Avg Daily Customers", value=f"{avg_cust_25:,.0f}",
             delta=f"{cust_delta:+.1f}% vs 2024",
             delta_dir="positive" if cust_delta >= 0 else "negative",
             accent=PALETTE["rolling"]),
        dict(label="2025 YoY Growth", value=f"{yoy_avg*100:.1f}%",
             delta="vs same day prior year", delta_dir="neutral",
             accent=PALETTE["target"]),
    ])

    # ── Daily Sales Trend ────────────────────────────────────────────────────
    tr_label = time_range_selector("ov_trend")
    daily_filtered = filter_range(daily, "date", tr_label)
    daily_filtered = daily_filtered.copy()
    daily_filtered["rolling_7d"] = daily_filtered["actual_sales"].rolling(7, min_periods=1).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_filtered["date"], y=daily_filtered["actual_sales"],
        mode="lines", name="Daily Sales",
        line=dict(color=PALETTE["primary"], width=1), opacity=0.45,
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily_filtered["date"], y=daily_filtered["rolling_7d"],
        mode="lines", name="7-Day Rolling Avg",
        line=dict(color=PALETTE["primary"], width=2.5),
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily_filtered["date"], y=daily_filtered["target"],
        mode="lines", name="Target",
        line=dict(color=PALETTE["target"], dash="dash", width=1.5),
    ))
    apply_theme(fig_trend, height=390)
    fig_trend.update_layout(title="Daily Sales vs Target")
    fig_trend.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Bottom row ───────────────────────────────────────────────────────────
    col_dow, _ = st.columns(2)
    with col_dow:
        dow_avg = (daily.groupby("day_of_week")["actual_sales"]
                   .mean().reindex(DOW_ORDER))
        bar_colors = [PALETTE["primary"]] * 7
        bar_colors[DOW_ORDER.index("Thursday")] = PALETTE["target"]
        bar_colors[DOW_ORDER.index("Saturday")] = PALETTE["target"]
        fig_dow = go.Figure(go.Bar(
            x=DOW_ORDER, y=dow_avg.values,
            marker_color=bar_colors,
        ))
        apply_theme(fig_dow, height=380, rangeslider=False)
        fig_dow.update_layout(title="Avg Sales by Day of Week", showlegend=False)
        fig_dow.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig_dow, use_container_width=True)

    # ── Division Pulse ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin:20px 0 12px;padding-top:18px;"
        f"border-top:1px solid {PALETTE['border']};font-size:0.8rem;"
        f"color:{PALETTE['subtext']};text-transform:uppercase;"
        f"letter-spacing:.08em'>Division Revenue Pulse — 24-month avg daily sales</div>",
        unsafe_allow_html=True,
    )
    div_monthly_df = pd.read_sql("""
        SELECT division_code, division_name, department,
               strftime('%Y-%m', date) AS month,
               AVG(sales_amt)          AS avg_daily
        FROM   division_daily
        GROUP  BY division_code, division_name, department, month
        ORDER  BY division_code, month
    """, engine)
    div_rev_df = pd.read_sql("""
        SELECT division_code, SUM(sales_amt) AS total_sales
        FROM   division_daily
        GROUP  BY division_code
    """, engine)
    total_rev_map = dict(zip(div_rev_df.division_code, div_rev_df.total_sales))

    _DEPT_COLORS = {
        "Women": PALETTE["women"], "Men": PALETTE["men"],
        "Kids":  PALETTE["kids"],  "Others": PALETTE["others"],
    }
    _DEPT_ORDER = ["Women", "Men", "Kids", "Others"]

    def _sparkline(values, color, uid):
        W, H = 130, 32
        mn, mx = min(values), max(values)
        rng = mx - mn or 1
        pts = [(i / (len(values) - 1) * W,
                H - ((v - mn) / rng) * (H - 4) - 2)
               for i, v in enumerate(values)]
        line = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        area = f"M{pts[0][0]:.1f},{H} L{pts[0][0]:.1f},{pts[0][1]:.1f}"
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]; x1, y1 = pts[i]; cx = (x0 + x1) / 2
            seg = f" C{cx:.1f},{y0:.1f} {cx:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
            line += seg
            area += seg
        area += f" L{pts[-1][0]:.1f},{H} Z"
        c = color.replace('#', '')
        return (
            f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" '
            f'style="width:100%;height:{H}px;display:block">'
            f'<defs><linearGradient id="sg{c}{uid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0.02"/>'
            f'</linearGradient></defs>'
            f'<path d="{area}" fill="url(#sg{c}{uid})"/>'
            f'<path d="{line}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
            f'</svg>'
        )

    _css = (
        f"<style>"
        f".dpgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px}}"
        f".dplabel{{font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;"
        f"padding-bottom:6px;border-bottom:1px solid {PALETTE['border']};"
        f"margin-top:16px;margin-bottom:4px}}"
        f".dpcard{{background:{PALETTE['card_bg']};border:1px solid {PALETTE['border']};"
        f"border-left:3px solid var(--dc);border-radius:7px;padding:10px 12px 8px}}"
        f".dpname{{font-size:.64rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.05em;color:{PALETTE['subtext']};"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}"
        f".dptotal{{font-size:1.05rem;font-weight:700;color:{PALETTE['text']};margin:3px 0 6px}}"
        f"</style>"
    )
    parts = [_css, '<div class="dpgrid">']
    for dept in _DEPT_ORDER:
        dept_color = _DEPT_COLORS[dept]
        dept_codes = div_monthly_df.loc[
            div_monthly_df.department == dept, 'division_code'
        ].unique()
        ranked = (div_rev_df[div_rev_df.division_code.isin(dept_codes)]
                  .sort_values('total_sales', ascending=False)['division_code']
                  .tolist())
        parts.append(
            f'<div class="dplabel" style="color:{dept_color};grid-column:1/-1">'
            f'{dept}</div>'
        )
        for code in ranked:
            grp = (div_monthly_df[div_monthly_df.division_code == code]
                   .sort_values('month'))
            if grp.empty:
                continue
            name = grp.iloc[0]['division_name']
            monthly_vals = grp['avg_daily'].tolist()
            total_m = total_rev_map.get(code, 0) / 1e6
            svg = _sparkline(monthly_vals, dept_color, uid=str(code))
            parts.append(
                f'<div class="dpcard" style="--dc:{dept_color}">'
                f'<div class="dpname">{name}</div>'
                f'<div class="dptotal">${total_m:.1f}M</div>'
                f'{svg}</div>'
            )
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)

# ===========================================================================
# PAGE 2: Division Forecast
# ===========================================================================

elif page == "Division Forecast":
    st.markdown(f"<h1 style='color:{PALETTE['text']}'>Division Scenario Planner</h1>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{PALETTE['subtext']}'>"
        "Answer: <em>\"What would this division sell on a given day in a given month, "
        "under specific weather and event conditions?\"</em> — uses the trained XGBoost model "
        "with SHAP attribution to show <em>why</em> the forecast is what it is."
        "</p>",
        unsafe_allow_html=True,
    )

    models = load_models()
    div_df_full = load_division_daily()
    div_feat = build_features(div_df_full)

    col_ctrl, col_main = st.columns([1, 2])

    with col_ctrl:
        st.markdown(f"<p style='color:{PALETTE['subtext']};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px'>Division</p>",
                    unsafe_allow_html=True)
        selected_div_name = st.selectbox("Division", [d[1] for d in DIVISIONS],
                                          index=13, label_visibility="collapsed")
        selected_div_code = next(d[0] for d in DIVISIONS if d[1] == selected_div_name)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{PALETTE['text']};font-weight:600;margin-bottom:6px'>Scenario Inputs</p>",
                    unsafe_allow_html=True)

        # Month selector — drives day_of_year + month features and suggests a default temperature
        selected_month_name = st.selectbox("Month (planning horizon)", _MONTH_NAMES,
                                            index=6)  # default July (winter, good for Outerwear demo)
        selected_month = _MONTH_NAMES.index(selected_month_name) + 1
        default_temp   = _SYDNEY_TEMP[selected_month]

        dow_input = st.selectbox("Day of Week", DOW_ORDER, index=3)
        temp      = st.slider(
            f"Temperature (°C)  ·  Sydney avg in {selected_month_name}: {default_temp}°C",
            min_value=5, max_value=38, value=default_temp, step=1,
        )
        event = st.selectbox("Event / Promotion",
                              ["None", "Arigato", "EOFY", "Vivid", "LongWeekend", "Christmas"])

        _p  = PALETTE
        _ev = f", <strong style='color:{_p['target']}'>{event}</strong>" if event != "None" else ""
        st.markdown(
            f"<p style='color:{_p['subtext']};font-size:0.75rem;margin-top:8px'>"
            f"Scenario: a <strong style='color:{_p['text']}'>{dow_input}</strong> in "
            f"<strong style='color:{_p['text']}'>{selected_month_name}</strong> at "
            f"<strong style='color:{_p['text']}'>{temp}°C</strong>{_ev}</p>",
            unsafe_allow_html=True,
        )

    with col_main:
        if "xgb_layer1" not in models:
            st.info("Run `python src/train_models.py` first to generate model artifacts.")
        else:
            xgb_model = models["xgb_layer1"]
            dow_map    = {d: i for i, d in enumerate(DOW_ORDER)}
            trading_hrs = 12 if dow_input == "Thursday" else (7 if dow_input == "Sunday" else 9)

            div_hist = div_feat[div_feat["division_code"] == selected_div_code].sort_values("date")
            if len(div_hist) == 0:
                st.warning("No historical data for this division.")
            else:
                # Use historical averages for this division in the selected month
                # as the lag/rolling context — much more realistic than last row.
                month_hist = div_hist[div_hist["month"] == selected_month]
                ref = month_hist if len(month_hist) >= 4 else div_hist
                lag_7   = float(ref["lag_7"].mean())
                lag_14  = float(ref["lag_14"].mean())
                lag_28  = float(ref["lag_28"].mean())
                roll_7  = float(ref["rolling_7_mean"].mean())
                roll_14 = float(ref["rolling_14_mean"].mean())

                input_row = pd.DataFrame([{
                    "division_code":     selected_div_code,
                    "temperature_index": temp,
                    "trading_hours":     trading_hrs,
                    "is_public_holiday": 0,
                    "promo_flag":        0,
                    "dow_num":           dow_map[dow_input],
                    "month":             selected_month,
                    "day_of_year":       _MONTH_DOY[selected_month],
                    "event_Arigato":     int(event == "Arigato"),
                    "event_Christmas":   int(event == "Christmas"),
                    "event_EOFY":        int(event == "EOFY"),
                    "event_Vivid":       int(event == "Vivid"),
                    "event_LongWeekend": int(event == "LongWeekend"),
                    "lag_7":             lag_7,
                    "lag_14":            lag_14,
                    "lag_28":            lag_28,
                    "rolling_7_mean":    roll_7,
                    "rolling_14_mean":   roll_14,
                }])

                # Clip to ≥ 0 — sales cannot be negative
                pred  = max(0.0, float(xgb_model.predict(input_row[FEATURE_COLS])[0]))
                lower = pred * 0.88
                upper = pred * 1.12
                if "xgb_quantile_lower" in models:
                    lower = max(0.0, float(models["xgb_quantile_lower"].predict(input_row[FEATURE_COLS])[0]))
                    upper = max(0.0, float(models["xgb_quantile_upper"].predict(input_row[FEATURE_COLS])[0]))

                # Historical average for same division + dow + month — gives user a sanity baseline
                hist_avg_rows = div_hist[
                    (div_hist["month"] == selected_month) &
                    (div_hist["day_of_week"] == dow_input)
                ]["sales_amt"]
                hist_avg = hist_avg_rows.mean() if len(hist_avg_rows) > 0 else None
                vs_hist = ((pred - hist_avg) / hist_avg * 100) if hist_avg else None

                hist_label = (
                    f"{vs_hist:+.1f}% vs historical avg (${hist_avg:,.0f})"
                    if hist_avg else "No historical data for this context"
                )
                hist_dir = ("positive" if vs_hist and vs_hist >= 0 else
                            "negative" if vs_hist else "neutral")

                render_metric_row([
                    dict(
                        label=f"Forecast — {selected_div_name} · {dow_input} in {selected_month_name}",
                        value=f"${pred:,.0f}",
                        delta=f"70% CI: ${lower:,.0f} – ${upper:,.0f}",
                        delta_dir="neutral", accent=PALETTE["primary"],
                    ),
                    dict(
                        label=f"Historical Avg (same context, 2024–2025)",
                        value=f"${hist_avg:,.0f}" if hist_avg else "—",
                        delta=hist_label,
                        delta_dir=hist_dir,
                        accent=PALETTE["positive"],
                    ),
                ])

                # SHAP waterfall
                if "shap_explainer" in models:
                    explainer = models["shap_explainer"]
                    shap_vals = explainer.shap_values(input_row[FEATURE_COLS])[0]

                    top_n   = 8
                    top_idx = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
                    shap_labels = [FEATURE_COLS[i] for i in top_idx]
                    shap_values = [float(shap_vals[i]) for i in top_idx]

                    base     = float(explainer.expected_value)
                    measures = ["absolute"] + ["relative"] * len(shap_labels)

                    fig_wf = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=measures,
                        x=["Base (avg all divisions)"] + shap_labels,
                        y=[base] + shap_values,
                        connector={"line": {"color": PALETTE["border"]}},
                        increasing={"marker": {"color": PALETTE["positive"]}},
                        decreasing={"marker": {"color": PALETTE["negative"]}},
                        totals={"marker": {"color": PALETTE["primary"]}},
                    ))
                    apply_theme(fig_wf, height=360, rangeslider=False)
                    fig_wf.update_layout(
                        title=f"Why ${pred:,.0f}? — SHAP Feature Attribution for {selected_div_name}",
                    )
                    fig_wf.update_yaxes(tickformat="$,.0f")
                    st.plotly_chart(fig_wf, use_container_width=True)

                    st.markdown(
                        f"<p style='color:{PALETTE['subtext']};font-size:0.8rem'>"
                        f"<strong>How to read this:</strong> The chart starts at the model's "
                        f"average prediction across all divisions (Base = ${base:,.0f}). "
                        f"Each bar shows how much a specific feature pushes the forecast "
                        f"up (green) or down (red). The final bar is the scenario forecast."
                        "</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("SHAP explainer not found in src/models/ — run train_models.py.")

            # Historical line chart — filtered by time range
            st.markdown("<hr>", unsafe_allow_html=True)
            tr_label = time_range_selector("fc_hist")
            div_hist_filtered = filter_range(
                div_feat[div_feat["division_code"] == selected_div_code],
                "date", tr_label,
            )
            # Monthly average bars so seasonal pattern is obvious
            div_monthly = (
                div_hist_filtered.groupby(
                    div_hist_filtered["date"].dt.to_period("M")
                )["sales_amt"].mean().reset_index()
            )
            div_monthly["date"] = div_monthly["date"].dt.to_timestamp()

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=div_monthly["date"], y=div_monthly["sales_amt"],
                name="Avg Daily Sales (monthly)",
                marker_color=PALETTE["primary"], opacity=0.7,
            ))
            # Overlay actual daily
            fig_hist.add_trace(go.Scatter(
                x=div_hist_filtered["date"], y=div_hist_filtered["sales_amt"],
                mode="lines", name="Daily actual",
                line=dict(color=PALETTE["primary"], width=1), opacity=0.3,
            ))
            apply_theme(fig_hist, height=300)
            fig_hist.update_layout(
                title=f"{selected_div_name} — Historical Sales (bars = monthly avg)",
                showlegend=True, barmode="overlay",
            )
            fig_hist.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig_hist, use_container_width=True)

# ===========================================================================
# PAGE 3: Forecast Confidence
# ===========================================================================

elif page == "Forecast Confidence":
    st.markdown(f"<h1 style='color:{PALETTE['text']}'>Forecast Confidence</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{PALETTE['subtext']}'>Layer 2: Confidence intervals + reliability flag based on historical MAPE by context (day-of-week × event).</p>",
                unsafe_allow_html=True)

    models = load_models()

    if "accuracy_lookup" not in models:
        st.info("Run `python src/train_models.py` first.")
    else:
        accuracy_lookup = models["accuracy_lookup"]

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            dow_sel = st.selectbox("Day of Week", DOW_ORDER, index=3)
        with col2:
            event_sel = st.selectbox("Event Flag",
                                      ["None", "Arigato", "Christmas", "EOFY", "Vivid", "LongWeekend"])

        # Reliability flag
        match = accuracy_lookup[
            (accuracy_lookup["day_of_week"] == dow_sel) &
            (accuracy_lookup["event_flag"] == event_sel)
        ]
        if event_sel in ("Arigato", "Christmas", "Vivid"):
            flag, flag_cls = "LOW",    "flag-low"
            explanation = f"Event week ({event_sel}) — irregular events have higher forecast error historically."
        elif len(match) == 0:
            flag, flag_cls = "MEDIUM", "flag-medium"
            explanation = "Insufficient history for this context."
        else:
            mape = match.iloc[0]["mean_mape"]
            if mape < 5:
                flag, flag_cls = "HIGH",   "flag-high"
                explanation = f"Historical MAPE for {dow_sel} / {event_sel}: {mape:.1f}% — reliable context."
            elif mape < 12:
                flag, flag_cls = "MEDIUM", "flag-medium"
                explanation = f"Historical MAPE for {dow_sel} / {event_sel}: {mape:.1f}%."
            else:
                flag, flag_cls = "LOW",    "flag-low"
                explanation = f"Historical MAPE for {dow_sel} / {event_sel}: {mape:.1f}% — less reliable context."

        with col3:
            st.markdown(
                f'<div style="padding-top:6px">'
                f'<span class="flag-chip {flag_cls}">● {flag} CONFIDENCE</span>'
                f'<p style="color:{PALETTE["subtext"]};font-size:0.85rem;margin-top:4px">{explanation}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # MAPE heatmap
        pivot = accuracy_lookup.pivot_table(
            index="event_flag", columns="day_of_week", values="mean_mape"
        )
        pivot = pivot.reindex(columns=[c for c in DOW_ORDER if c in pivot.columns])

        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            labels=dict(color="MAPE %"), text_auto=".1f",
            title="Historical MAPE (%) by Day-of-Week × Event — lower is better",
        )
        fig_heat.update_layout(**{k: v for k, v in _CHART_BASE.items()
                                    if k not in ("xaxis", "yaxis")}, height=320)
        fig_heat.update_coloraxes(colorbar_ticksuffix="%")
        st.plotly_chart(fig_heat, use_container_width=True)

        # CI band chart
        fl = load_forecast_log()
        if len(fl) > 0 and "upper_bound" in fl.columns:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{PALETTE['text']};font-weight:600'>Forecast with 70% Confidence Interval</p>",
                        unsafe_allow_html=True)
            tr_label = time_range_selector("ci_chart")
            fl_filtered = filter_range(fl, "target_date", tr_label)
            fl_agg = fl_filtered.groupby("target_date").agg(
                actual=("actual_sales", "sum"),
                predicted=("predicted_sales", "sum"),
                lower=("lower_bound", "sum"),
                upper=("upper_bound", "sum"),
            ).reset_index()

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                x=fl_agg["target_date"], y=fl_agg["upper"],
                fill=None, mode="lines", line_color=PALETTE["ci_line"], showlegend=False,
            ))
            fig_ci.add_trace(go.Scatter(
                x=fl_agg["target_date"], y=fl_agg["lower"],
                fill="tonexty", mode="lines",
                line_color=PALETTE["ci_line"], fillcolor=PALETTE["ci_fill"], name="70% CI",
            ))
            fig_ci.add_trace(go.Scatter(
                x=fl_agg["target_date"], y=fl_agg["predicted"],
                mode="lines", name="Forecast",
                line=dict(color=PALETTE["primary"], dash="dash", width=1.5),
            ))
            fig_ci.add_trace(go.Scatter(
                x=fl_agg["target_date"], y=fl_agg["actual"],
                mode="lines", name="Actual",
                line=dict(color=PALETTE["positive"], width=2),
            ))
            apply_theme(fig_ci, height=400)
            fig_ci.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig_ci, use_container_width=True)

# ===========================================================================
# PAGE 4: Model Self-Check
# ===========================================================================

elif page == "Model Self-Check":
    st.markdown(f"<h1 style='color:{PALETTE['text']}'>Model Self-Check</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{PALETTE['subtext']}'>Layer 3: The model detects its own systematic forecast biases via rolling error tracking.</p>",
                unsafe_allow_html=True)

    models = load_models()
    fl     = load_forecast_log()

    if len(fl) == 0:
        st.info("Run `python src/train_models.py` first to generate the forecast log.")
    else:
        # ── Active bias flags ────────────────────────────────────────────────
        if "bias_flags" in models and len(models["bias_flags"]) > 0:
            bias_df = models["bias_flags"]
            top_flags = bias_df.sort_values(
                "latest_rolling_bias_pct", key=abs, ascending=False
            ).head(6)

            kpi_cols = st.columns(3)
            kpi_cols[0].markdown(
                _metric_card("Active Bias Flags", str(len(bias_df)),
                             delta=f"{len(bias_df[bias_df['direction']=='underestimating'])} under / "
                                   f"{len(bias_df[bias_df['direction']=='overestimating'])} over",
                             delta_dir="negative", accent=PALETTE["negative"]),
                unsafe_allow_html=True,
            )
            kpi_cols[1].markdown(
                _metric_card("Worst Bias",
                             f"{bias_df['latest_rolling_bias_pct'].abs().max():.1f}%",
                             delta=bias_df.loc[bias_df['latest_rolling_bias_pct'].abs().idxmax(),
                                               'division_name'],
                             delta_dir="negative", accent=PALETTE["negative"]),
                unsafe_allow_html=True,
            )
            kpi_cols[2].markdown(
                _metric_card("Longest Streak",
                             f"{bias_df['consecutive_flagged_periods'].max()} weeks",
                             delta=bias_df.loc[bias_df['consecutive_flagged_periods'].idxmax(),
                                               'division_name'],
                             delta_dir="negative", accent=PALETTE["target"]),
                unsafe_allow_html=True,
            )

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{PALETTE['text']};font-weight:600;margin-bottom:8px'>Top Active Flags</p>",
                        unsafe_allow_html=True)
            for _, row in top_flags.iterrows():
                icon = "📈" if row["direction"] == "underestimating" else "📉"
                st.markdown(
                    f'<div class="bias-alert">{icon} {row["alert"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No systematic biases detected above the 5% threshold.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Rolling bias chart ───────────────────────────────────────────────
        available_divs = sorted(fl["division_code"].unique())
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            selected_div = st.selectbox(
                "Division",
                available_divs,
                format_func=lambda x: f"{DIV_CODE_TO_NAME.get(x, str(x))}",
                index=min(3, len(available_divs) - 1),
            )

        fl = fl.copy()
        fl["signed_error_pct"] = (
            (fl["actual_sales"] - fl["predicted_sales"]) / fl["actual_sales"] * 100
        )
        fl_sorted = fl.sort_values(["division_code", "dow", "target_date"])
        fl_sorted["rolling_6w_bias"] = (
            fl_sorted.groupby(["division_code", "dow"])["signed_error_pct"]
            .transform(lambda x: x.rolling(6, min_periods=4).mean())
        )

        tr_label = time_range_selector("sc_bias")
        div_fl = fl_sorted[fl_sorted["division_code"] == selected_div].dropna(subset=["rolling_6w_bias"])
        div_fl = filter_range(div_fl, "target_date", tr_label)

        # DOW colors from palette + extras
        dow_colors = [PALETTE["primary"], PALETTE["positive"], PALETTE["target"],
                      PALETTE["negative"], PALETTE["rolling"], "#F9A825", "#80CBC4"]

        fig_bias = go.Figure()
        for i, dow in enumerate(DOW_ORDER):
            d = div_fl[div_fl["dow"] == dow]
            if len(d) == 0:
                continue
            fig_bias.add_trace(go.Scatter(
                x=d["target_date"], y=d["rolling_6w_bias"],
                mode="lines", name=dow,
                line=dict(color=dow_colors[i], width=1.8),
            ))
        fig_bias.add_hline(y=0,  line_dash="dash",  line_color=PALETTE["neutral"],
                           line_width=1)
        fig_bias.add_hline(y=5,  line_dash="dot",   line_color=PALETTE["target"],
                           annotation_text="+5% threshold",
                           annotation_font_color=PALETTE["target"])
        fig_bias.add_hline(y=-5, line_dash="dot",   line_color=PALETTE["target"],
                           annotation_text="-5% threshold",
                           annotation_font_color=PALETTE["target"])

        div_label = DIV_CODE_TO_NAME.get(selected_div, str(selected_div))
        apply_theme(fig_bias, height=420)
        fig_bias.update_layout(title=f"Rolling 6-Week Bias by Day of Week — {div_label}")
        fig_bias.update_yaxes(title_text="Mean Signed Error", ticksuffix="%")
        st.plotly_chart(fig_bias, use_container_width=True)

        # ── Error heatmap ────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{PALETTE['text']};font-weight:600'>Mean Signed Error: All Divisions × Day of Week</p>",
                    unsafe_allow_html=True)

        err_heat = (
            fl_sorted.groupby(["division_name", "dow"])["signed_error_pct"]
            .mean().reset_index()
        )
        pivot = err_heat.pivot(index="division_name", columns="dow",
                               values="signed_error_pct")
        pivot = pivot.reindex(columns=[c for c in DOW_ORDER if c in pivot.columns])

        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdBu",
            color_continuous_midpoint=0, text_auto=".1f",
            title="Signed Error %  ·  Blue = Under-forecast  ·  Red = Over-forecast",
        )
        fig_heat.update_layout(**{k: v for k, v in _CHART_BASE.items()
                                    if k not in ("xaxis", "yaxis")}, height=660)
        fig_heat.update_coloraxes(colorbar_ticksuffix="%")
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Bias table ───────────────────────────────────────────────────────
        if "bias_flags" in models and len(models["bias_flags"]) > 0:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{PALETTE['text']};font-weight:600'>Systematic Bias Table</p>",
                        unsafe_allow_html=True)
            st.dataframe(
                models["bias_flags"][
                    ["division_name", "day_of_week",
                     "consecutive_flagged_periods", "latest_rolling_bias_pct", "direction"]
                ].rename(columns={
                    "division_name": "Division",
                    "day_of_week": "Day",
                    "consecutive_flagged_periods": "Consecutive Weeks Flagged",
                    "latest_rolling_bias_pct": "Latest Bias %",
                    "direction": "Direction",
                }).sort_values("Latest Bias %", key=abs, ascending=False),
                use_container_width=True,
                hide_index=True,
            )
