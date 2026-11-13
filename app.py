# app.py
# Unified Gold & Bitcoin market intelligence dashboard.
# Live market quotes, explainable multi-factor forecasts and immutable forecast tracking.
# Other application modules (Jobs, Real Estate, Task Planner and Newsletter) remain separate.
#
# Author: ChatGPT (generated for user)
# Date: 2025-10-03
# ------------------------------------------------------------------------------

import os
import yaml
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

# -------------------------------------------------------------------
# APP CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Optional Auto Refresh Setup ---
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# --- Auto-refresh every 10 minutes (single timer for the whole application) ---
if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=600000, key="datarefresh")  # 10 min in milliseconds
else:
    st.sidebar.error("Rafraîchissement automatique indisponible : dépendance streamlit-autorefresh manquante. Vérifier requirements.txt et le déploiement.")

# --- Manual Refresh Button ---
st.sidebar.markdown("### 🔁 Data Refresh")
if st.sidebar.button("Refresh Now"):
    st.success("Refreshing dashboard... please wait ⏳")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Dernier recalcul de la page : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} · rafraîchissement automatique toutes les 10 min")

# Preserve original external imports but provide safe fallbacks
try:
    from jobs_app import jobs_dashboard
except Exception:
    def jobs_dashboard():
        st.warning("jobs_app not found — Jobs dashboard unavailable.")

try:
    from real_estate_bot import real_estate_dashboard
except Exception:
    def real_estate_dashboard():
        st.warning("real_estate_bot not found — Real Estate Bot unavailable.")

from tasks_planner import render_task_planner
from newsletter_app import newsletter_dashboard


#from Formation_Word_Excel_PowerPoint import render_training_dashboard

from market_forecast import render_market_forecast_dashboard
from financial_intelligence_agent import render_financial_intelligence_dashboard

# -------------------------------------------------------------------
# FILE PATHS AND DATA DIRECTORIES
# -------------------------------------------------------------------
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass  # downstream writes will warn if they fail

PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# -------------------------------------------------------------------
# UTILITIES: FILE I/O (robust)
# -------------------------------------------------------------------
def load_csv_safe(path: str, default_cols: List[str], parse_ts: bool = True) -> pd.DataFrame:
    """
    Load a CSV safely. If the file doesn't exist, returns an empty DataFrame with default_cols.
    Attempts to parse 'timestamp' to datetime when parse_ts=True.
    """
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if parse_ts and "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
                except Exception:
                    # fallback: leave as-is
                    pass
            return df
        except Exception as e:
            st.warning(f"Failed to read {path}: {e}")
            return pd.DataFrame(columns=default_cols)
    else:
        return pd.DataFrame(columns=default_cols)


def save_df_to_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save a DataFrame to CSV, with defensive error handling.
    """
    try:
        df.to_csv(path, index=index)
    except Exception as e:
        st.warning(f"Failed to save CSV to {path}: {e}")


def append_prediction_to_log(path: str, row: Dict[str, Any], dedupe_on: Optional[List[str]] = None):
    """
    Append a single-row dict to CSV at 'path'. If file doesn't exist, create it.
    If dedupe_on is provided, drop duplicates keeping the last timestamp per key.
    """
    try:
        new_row = pd.DataFrame([row])
        if os.path.exists(path):
            existing = pd.read_csv(path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        # Attempt to parse timestamp column if present
        if "timestamp" in combined.columns:
            try:
                combined["timestamp"] = pd.to_datetime(combined["timestamp"], infer_datetime_format=True)
            except Exception:
                pass

        if dedupe_on:
            # sort by timestamp so keep last
            if "timestamp" in combined.columns:
                combined.sort_values("timestamp", inplace=True)
            combined = combined.drop_duplicates(subset=dedupe_on, keep="last")

        combined.to_csv(path, index=False)
    except Exception as e:
        st.warning(f"Could not append to log {path}: {e}")

# -------------------------------------------------------------------
# PORTFOLIO ENGINE + EUR/AED MONITORING (ROBUST FINAL FIX)
# -------------------------------------------------------------------

import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from datetime import datetime, timedelta

# -------------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------------

PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_grid.csv")
DYNAMIC_PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_dynamic.csv")
FX_HISTORY_FILE = os.path.join(DATA_DIR, "eur_aed_history.csv")


# -------------------------------------------------------------------
# FX LIVE RATE
# -------------------------------------------------------------------

def get_eur_aed_rate():
    try:
        url = "https://open.er-api.com/v6/latest/EUR"
        r = requests.get(url, timeout=10)
        data = r.json()
        return round(float(data["rates"]["AED"]), 4)

    except Exception as e:
        st.warning(f"EUR/AED API error: {e}")
        return None


# -------------------------------------------------------------------
# BUILD REAL 90D HISTORY (IMPORTANT FIX)
# -------------------------------------------------------------------

def fetch_eur_aed_90d():
    """
    Real historical daily FX data (NOT single-day API).
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=90)

        url = (
            f"https://api.frankfurter.app/"
            f"{start.strftime('%Y-%m-%d')}..{end.strftime('%Y-%m-%d')}"
            "?from=EUR&to=AED"
        )

        r = requests.get(url, timeout=15)
        data = r.json()

        rates = data.get("rates", {})

        rows = []
        for date, value in rates.items():
            if isinstance(value, dict) and "AED" in value:
                rows.append({
                    "timestamp": pd.to_datetime(date),
                    "eur_aed": float(value["AED"])
                })

        df = pd.DataFrame(rows)

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("timestamp")
        df = df.drop_duplicates()
        return df

    except Exception as e:
        st.warning(f"FX history fetch failed: {e}")
        return pd.DataFrame()


# -------------------------------------------------------------------
# SAVE LOCAL HISTORY (APPEND LIVE POINT)
# -------------------------------------------------------------------

def save_fx_history(rate):

    if rate is None:
        return

    try:
        new_row = pd.DataFrame([{
            "timestamp": datetime.now(),
            "eur_aed": float(rate)
        }])

        if os.path.exists(FX_HISTORY_FILE):
            old = pd.read_csv(FX_HISTORY_FILE)
        else:
            old = pd.DataFrame(columns=["timestamp", "eur_aed"])

        df = pd.concat([old, new_row], ignore_index=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["eur_aed"] = pd.to_numeric(df["eur_aed"], errors="coerce")

        df = df.dropna().drop_duplicates()
        df = df.sort_values("timestamp")

        df.to_csv(FX_HISTORY_FILE, index=False)

    except Exception as e:
        st.warning(f"FX save error: {e}")


# -------------------------------------------------------------------
# LOAD HISTORY (SAFE)
# -------------------------------------------------------------------

def load_fx_history():
    try:
        if not os.path.exists(FX_HISTORY_FILE):
            return pd.DataFrame()

        df = pd.read_csv(FX_HISTORY_FILE)

        if df.empty:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["eur_aed"] = pd.to_numeric(df["eur_aed"], errors="coerce")

        df = df.dropna().sort_values("timestamp")
        return df

    except Exception:
        return pd.DataFrame()


# -------------------------------------------------------------------
# PORTFOLIO LOADER
# -------------------------------------------------------------------

def load_portfolio():
    try:
        df = pd.read_csv(PORTFOLIO_FILE)

        for col in ["Base_Allocation", "Now_Pct", "Next_Pct", "Dip_Pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Portfolio loading error: {e}")
        return pd.DataFrame()


# -------------------------------------------------------------------
# SAVE DYNAMIC PORTFOLIO
# -------------------------------------------------------------------

def save_dynamic_portfolio(df):
    try:
        df.to_csv(DYNAMIC_PORTFOLIO_FILE, index=False)
    except Exception as e:
        st.warning(f"Portfolio save error: {e}")


# -------------------------------------------------------------------
# PORTFOLIO ENGINE (UNCHANGED LOGIC)
# -------------------------------------------------------------------

def dynamic_portfolio_adjustment(df, btc_df, gold_df):

    if df.empty:
        return df

    adjusted = df.copy()

    btc_signal = btc_df["signal"].iloc[-1] if not btc_df.empty else "Hold"
    gold_signal = gold_df["signal"].iloc[-1] if not gold_df.empty else "Hold"

    vix = st.session_state.get("vix_adj", 20)
    inflation = st.session_state.get("inflation_adj", 2.5)
    oil = st.session_state.get("oil_adj", 0)

    # stress
    if vix >= 40:
        adjusted.loc[adjusted["Bucket"] == "DEFENSIVE", "Base_Allocation"] *= 1.20
        adjusted.loc[adjusted["Bucket"] == "CASH", "Base_Allocation"] *= 1.30
        adjusted.loc[adjusted["Bucket"].isin(["SPECIAL", "GEO"]), "Base_Allocation"] *= 0.80

    # gold
    if gold_signal == "Buy":
        adjusted.loc[adjusted["Ticker"] == "GLD", "Base_Allocation"] *= 1.15

    # btc
    if btc_signal == "Buy":
        adjusted.loc[adjusted["Bucket"] == "SPECIAL", "Base_Allocation"] *= 1.10

    # inflation
    if inflation >= 5:
        adjusted.loc[adjusted["Bucket"] == "ENERGY", "Base_Allocation"] *= 1.12

    # oil
    if oil >= 15:
        adjusted.loc[adjusted["Bucket"] == "ENERGY", "Base_Allocation"] *= 1.08

    elif oil <= -15:
        adjusted.loc[adjusted["Bucket"] == "ENERGY", "Base_Allocation"] *= 0.92

    # FX LIVE
    eur_aed = get_eur_aed_rate()
    adjusted["EUR_AED"] = eur_aed

    if eur_aed:
        if eur_aed <= 3.8:
            adjusted.loc[adjusted["Bucket"] == "CASH", "Base_Allocation"] *= 1.10
        elif eur_aed >= 4.2:
            adjusted.loc[adjusted["Bucket"] == "SPECIAL", "Base_Allocation"] *= 1.05

    # safety
    adjusted["Base_Allocation"] = adjusted["Base_Allocation"].clip(3, 25)

    total = adjusted["Base_Allocation"].sum()

    adjusted["Dynamic_Allocation"] = (adjusted["Base_Allocation"] / total * 100).round(2)

    adjusted["Last_Update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_dynamic_portfolio(adjusted)

    return adjusted


# -------------------------------------------------------------------
# EUR/AED MONITOR (FULL FIXED VERSION)
# -------------------------------------------------------------------

def render_eur_aed_monitor():

    st.subheader("💱 EUR/AED — Dubai Payment Risk Monitor")

    # LIVE RATE
    current = get_eur_aed_rate()

    if current is None:
        st.error("Cannot fetch EUR/AED rate")
        return

    save_fx_history(current)

    # REAL 90D DATA (THIS IS THE KEY FIX)
    history = fetch_eur_aed_90d()

    # fallback to local file if API fails
    if history.empty:
        history = load_fx_history()

    # final fallback
    if history.empty:
        st.warning("Building initial FX dataset...")
        history = pd.DataFrame([{
            "timestamp": datetime.now(),
            "eur_aed": current
        }])

    # ensure last 90 days only
    cutoff = datetime.now() - timedelta(days=90)
    history = history[history["timestamp"] >= cutoff]

    history = history.sort_values("timestamp")

    # METRICS
    start = history["eur_aed"].iloc[0]
    last = history["eur_aed"].iloc[-1]

    pct = ((last - start) / start) * 100

    high = history["eur_aed"].max()
    low = history["eur_aed"].min()

    # ALERTS
    if pct <= -10:
        st.error(f"🚨 EUR -{pct:.2f}% → Dubai more expensive")
    elif pct >= 10:
        st.success(f"✅ EUR +{pct:.2f}% → Dubai cheaper")
    else:
        st.warning(f"EUR/AED: {pct:.2f}%")

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.metric("EUR/AED", f"{last:.4f}")
    c2.metric("90D Change", f"{pct:.2f}%")
    c3.metric("Range", f"{low:.4f} → {high:.4f}")

    # CHART
    color = "red" if pct <= -10 else "green" if pct >= 10 else "orange"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["timestamp"],
        y=history["eur_aed"],
        mode="lines",
        line=dict(color=color, width=3),
        fill="tozeroy"
    ))

    fig.add_hline(y=start * 1.10, line_dash="dash", line_color="green")
    fig.add_hline(y=start * 0.90, line_dash="dash", line_color="red")

    fig.update_layout(
        title="EUR/AED — Last 90 Days",
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)



# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])

# normalize timestamp columns where present
def ensure_timestamp(df: pd.DataFrame, col: str = "timestamp"):
    if df is None or df.empty:
        return
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        except Exception:
            pass

ensure_timestamp(df_pred)
ensure_timestamp(df_actual)

# Load weights (assumptions)
if os.path.exists(WEIGHT_FILE):
    try:
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f) or {}
    except Exception:
        weights = {"gold": {}, "bitcoin": {}}
else:
    weights = {"gold": {}, "bitcoin": {}}


# -------------------------------------------------------------------
# UI THEMES & ICONS
# -------------------------------------------------------------------
INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

ASSET_THEMES = {
    "Gold": {
        "buy": "#FFF9C4", "sell": "#FFE0B2", "hold": "#E0E0E0",
        "target_bg": "#FFFDE7", "target_text": "black",
        "assumption_pos": "#FFD54F", "assumption_neg": "#FFAB91",
        "chart_actual": "#FBC02D", "chart_pred": "#FFCC80", "chart_ai": "#FF6F61"
    },
    "Bitcoin": {
        "buy": "#BBDEFB", "sell": "#FFCDD2", "hold": "#CFD8DC",
        "target_bg": "#E3F2FD", "target_text": "black",
        "assumption_pos": "#64B5F6", "assumption_neg": "#EF9A9A",
        "chart_actual": "#42A5F5", "chart_pred": "#81D4FA", "chart_ai": "#FF6F61"
    }
}


# -------------------------------------------------------------------
# PRESENTATIONAL HELPERS
# -------------------------------------------------------------------
def alert_badge(signal: str, asset_name: str) -> str:
    theme = ASSET_THEMES.get(asset_name, ASSET_THEMES["Gold"])
    color = theme["buy"] if signal == "Buy" else theme["sell"] if signal == "Sell" else theme["hold"]
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'


def target_price_card(price: Any, asset_name: str, horizon: str):
    theme = ASSET_THEMES.get(asset_name, ASSET_THEMES["Gold"])
    st.markdown(f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price} <br>⏳ Horizon: {horizon}
        </div>
        """, unsafe_allow_html=True)


def pretty_datetime(dt: Optional[datetime]) -> str:
    if dt is None or pd.isna(dt):
        return "-"
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except Exception:
            return str(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------------------------
# ASSUMPTIONS & EXPLANATION CARDS
# -------------------------------------------------------------------
def explanation_card(asset_df: pd.DataFrame, asset_name: str):
    st.subheader(f"📖 Explanation for {asset_name}")
    assumptions = {}
    if "assumptions" in asset_df.columns and not asset_df.empty:
        try:
            assumptions_str = asset_df["assumptions"].iloc[-1]
            assumptions = eval(assumptions_str) if isinstance(assumptions_str, str) else assumptions_str or {}
        except Exception:
            assumptions = {}

    if assumptions:
        main_indicator = max(assumptions.items(), key=lambda x: abs(x[1]))
        indicator, value = main_indicator
        direction = "rising 📈" if value > 0 else "falling 📉" if value < 0 else "stable ⚖️"
        explanation_text = f"The latest forecast for **{asset_name}** was mainly driven by **{indicator.replace('_',' ').title()}**, which is {direction} ({value:.2f})."
    else:
        explanation_text = f"No clear driver identified for {asset_name}."

    st.markdown(f"""
        <div style="background-color:#FDF6EC; padding:12px; border-radius:12px; 
        box-shadow:0px 1px 3px rgba(0,0,0,0.1); font-size:16px; line-height:1.6;">
            {explanation_text}
        </div>
    """, unsafe_allow_html=True)


def assumptions_card(asset_df: pd.DataFrame, asset_name: str):
    st.subheader(f"📖 Assumptions for {asset_name}")

    if "assumptions" in asset_df.columns and not asset_df.empty:
        try:
            assumptions_str = asset_df["assumptions"].iloc[-1]
        except Exception:
            assumptions_str = "{}"
    else:
        assumptions_str = "{}"

    if "target_horizon" in asset_df.columns and not asset_df.empty:
        try:
            target_horizon = asset_df["target_horizon"].iloc[-1]
        except Exception:
            target_horizon = "Days"
    else:
        target_horizon = "Days"

    try:
        assumptions = eval(assumptions_str)
    except Exception:
        assumptions = {}

    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return

    # determine main factor
    max_factor = max(assumptions.items(), key=lambda x: abs(x[1]))
    factor_name, factor_value = max_factor
    if factor_value > 0:
        explanation = f"Predictions for **{asset_name}** are mainly supported by positive trends in **{factor_name.replace('_',' ')}**."
    elif factor_value < 0:
        explanation = f"Predictions for **{asset_name}** are mainly pressured by negative trends in **{factor_name.replace('_',' ')}**."
    else:
        explanation = f"Predictions for **{asset_name}** show no dominant influencing factor."

    st.markdown(f"""
        <div style="background-color:#FDF6EC; padding:12px; border-radius:12px; 
                    box-shadow:0px 1px 3px rgba(0,0,0,0.1);">
            {explanation}<br>
            <i>Forecast horizon: {target_horizon}</i>
        </div>
    """, unsafe_allow_html=True)

    indicators = list(assumptions.keys())
    values = [assumptions[k] for k in indicators]
    icons = [INDICATOR_ICONS.get(k, "❔") for k in indicators]
    theme = ASSET_THEMES.get(asset_name, ASSET_THEMES["Gold"])
    colors = [theme["assumption_pos"] if v > 0 else theme["assumption_neg"] if v < 0 else theme["hold"] for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(x=[f"{icon} {ind}"], y=[val], marker_color=color, text=[f"{val:.2f}"], textposition='auto'))

    fig.update_layout(
        title=f"{asset_name} Assumptions & Target ({target_horizon})",
        yaxis_title="Weight / Impact",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# -------------------------------------------------------------------
def merge_actual_pred(asset_name: str, actual_col: str) -> pd.DataFrame:
    """
    Merge predictions file with actuals file to generate signal, trend, assumptions and target horizon.
    """
    global df_pred, df_actual, weights

    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    if "timestamp" in asset_pred.columns and "timestamp" in df_actual.columns:
        try:
            asset_pred["timestamp"] = pd.to_datetime(asset_pred["timestamp"], infer_datetime_format=True)
            df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"], infer_datetime_format=True)
        except Exception:
            pass

    if actual_col in df_actual.columns:
        try:
            asset_pred = pd.merge_asof(
                asset_pred.sort_values("timestamp"),
                df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("1D")
            )
        except Exception:
            asset_pred = pd.merge_asof(
                asset_pred.sort_values("timestamp"),
                df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
                on="timestamp",
                direction="backward"
            )
    else:
        asset_pred["actual"] = None

    asset_pred["predicted_price"] = pd.to_numeric(asset_pred.get("predicted_price", pd.Series(dtype=float)), errors="coerce")
    asset_pred["actual"] = pd.to_numeric(asset_pred.get("actual", pd.Series(dtype=float)), errors="coerce")

    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if pd.notna(row["predicted_price"]) and pd.notna(row["actual"]) and row["predicted_price"] > row["actual"]
        else ("Sell" if pd.notna(row["predicted_price"]) and pd.notna(row["actual"]) and row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    # Trend: last 3 predicted points monotonicity
    asset_pred["trend"] = ""
    try:
        if len(asset_pred) >= 3:
            last3 = asset_pred["predicted_price"].tail(3)
            if last3.is_monotonic_increasing:
                asset_pred["trend"] = "Bullish 📈"
            elif last3.is_monotonic_decreasing:
                asset_pred["trend"] = "Bearish 📉"
            else:
                asset_pred["trend"] = "Neutral ⚖️"
    except Exception:
        asset_pred["trend"] = "Neutral ⚖️"

    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))

    # Horizon based on volatility
    horizon = "Days"
    if "volatility" in asset_pred.columns and not asset_pred["volatility"].isna().all():
        try:
            avg_vol = asset_pred["volatility"].mean()
            if avg_vol < 0.02:
                horizon = "Years"
            elif avg_vol < 0.05:
                horizon = "Months"
            else:
                horizon = "Days"
        except Exception:
            horizon = "Days"

    asset_pred["target_horizon"] = horizon
    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"] == "Buy" else (row["predicted_price"] if row["signal"] == "Sell" else row["actual"]),
        axis=1
    )
    return asset_pred


gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")


# -------------------------------------------------------------------
# What-if controls removed from the sidebar; base forecasts remain unchanged.
# Portfolio stress inputs continue to use their neutral defaults.
# -------------------------------------------------------------------


def generate_summary(asset_df: pd.DataFrame, asset_name: str) -> str:
    if asset_df is None or asset_df.empty:
        return f"No data for {asset_name}"
    last_row = asset_df.iloc[-1]
    return f"**{asset_name} Market Summary:** Signal: {last_row.get('signal', 'N/A')} | Trend: {last_row.get('trend', 'N/A')} | Target Price: {last_row.get('target_price', 'N/A')}"


# -------------------------------------------------------------------
# The Gold & Bitcoin menu now delegates all market analysis and forecasting
# to market_forecast.py. Legacy AI Forecast and synthetic Candlestick menus are retired.


# -------------------------------------------------------------------
# MAIN MENU: one unified Gold & Bitcoin market module
# -------------------------------------------------------------------
menu = st.sidebar.radio(
    "📊 Choose Dashboard",
    ["Financial Intelligence", "Gold & Bitcoin", "Jobs", "Real Estate Bot", "Formation Word & Excel", "Task Planner", "Newsletter"]
)


# small header helper
def render_header(title: str, subtitle: Optional[str] = None):
    st.title(title)
    if subtitle:
        st.markdown(f"_{subtitle}_")


# -------------------------------------------------------------------
# UNIFIED GOLD & BITCOIN MARKET INTELLIGENCE
# -------------------------------------------------------------------
if menu == "Financial Intelligence":
    render_financial_intelligence_dashboard()

elif menu == "Gold & Bitcoin":
    render_market_forecast_dashboard()


# -------------------------------------------------------------------
# JOBS MENU
# -------------------------------------------------------------------
elif menu == "Jobs":
    try:
        jobs_dashboard()
    except Exception as e:
        st.error(f"Error running jobs_dashboard(): {e}")


# -------------------------------------------------------------------
# REAL ESTATE BOT MENU
# -------------------------------------------------------------------
elif menu == "Real Estate Bot":
    try:
        real_estate_dashboard()
    except Exception as e:
        st.error(f"Error running real_estate_dashboard(): {e}")


# -------------------------------------------------------------------
# Training MENU
# -------------------------------------------------------------------
#elif menu == "Formation Word & Excel":
#       render_training_dashboard()

# -------------------------------------------------------------------
# Task planner MENU
# -------------------------------------------------------------------
elif menu == "Task Planner":
    render_task_planner()
# Newesletter
# -------------------------------------------------------------------
elif menu == "Newsletter":
    newsletter_dashboard()
# -------------------------------------------------------------------
