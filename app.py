# app.py
# Rewritten full application file with separate "Candlestick Predictions" menu and
# next-week synthetic candlestick generation moved to candlestick_predictions.py.
#
# Features:
#  - Gold & Bitcoin dashboards
#  - AI Forecast dashboard (candlestick-related predictions removed from here)
#  - Dedicated: Candlestick Predictions menu (calls render_candlestick_dashboard)
#  - Jobs & Real Estate Bot placeholders preserved
#  - Logging of predicted daily values into data/ai_predictions_log.csv
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


# --- Auto-refresh every 30 minutes (1,800,000 ms) ---
st_autorefresh(interval=1800000, key="datarefresh")

# --- Manual Refresh Button ---
st.sidebar.markdown("### 🔁 Data Refresh")
if st.sidebar.button("Refresh Now"):
    st.success("Refreshing dashboard... please wait ⏳")
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Preserve original external imports but provide safe fallbacks
try:
    from jobs_app import jobs_dashboard
except Exception:
    def jobs_dashboard():
        st.warning("jobs_app not found — Jobs dashboard unavailable.")

try:
    from ai_predictor import predict_next_n, compare_predictions_vs_actuals
except Exception:
    def predict_next_n(asset_name="Bitcoin", n_steps=7):
        # fallback: no AI predictions available
        return pd.DataFrame()

    def compare_predictions_vs_actuals(*args, **kwargs):
        return {}

try:
    from real_estate_bot import real_estate_dashboard
except Exception:
    def real_estate_dashboard():
        st.warning("real_estate_bot not found — Real Estate Bot unavailable.")

from tasks_planner import render_task_planner

#from Formation_Word_Excel_PowerPoint import render_training_dashboard

# -------------------------------------------------------------------
# NEW: candlestick module import (all candlestick-specific logic moved there)
# -------------------------------------------------------------------
#try:
#    from candlestick_predictions import render_candlestick_dashboard
#except Exception as e:
#    def render_candlestick_dashboard(df_actual):
#        st.error(f"Candlestick module not found: {e}\nPlease ensure candlestick_predictions.py exists.")
# -------------------------------------------------------------------
# Candlestick Predictions import
# -------------------------------------------------------------------

from candlestick_predictions import render_candlestick_dashboard
from candlestick_predictions import render_daily_candlestick_dashboard
# -------------------------------------------------------------------
# APP CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide", initial_sidebar_state="expanded")

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
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
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
# LOAD DATA
# -------------------------------------------------------------------
df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price", "method"])

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
ensure_timestamp(df_ai_pred_log)

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
# WHAT-IF SLIDERS
# -------------------------------------------------------------------
st.sidebar.header("🔧 What-If Scenario")
def get_default_assumption(df: pd.DataFrame, key: str, fallback: float):
    if df is None or df.empty:
        return fallback
    try:
        last_assumptions = eval(df["assumptions"].iloc[-1])
        return last_assumptions.get(key, fallback)
    except Exception:
        return fallback

if "inflation_adj" not in st.session_state:
    st.session_state.inflation_adj = get_default_assumption(gold_df, "inflation", 2.5)
if "usd_adj" not in st.session_state:
    st.session_state.usd_adj = get_default_assumption(gold_df, "usd_strength", 0.0)
if "oil_adj" not in st.session_state:
    st.session_state.oil_adj = get_default_assumption(gold_df, "energy_prices", 0.0)
if "vix_adj" not in st.session_state:
    st.session_state.vix_adj = get_default_assumption(gold_df, "tail_risk_event", 20.0)

def reset_sliders():
    st.session_state.inflation_adj = get_default_assumption(gold_df, "inflation", 2.5)
    st.session_state.usd_adj = get_default_assumption(gold_df, "usd_strength", 0.0)
    st.session_state.oil_adj = get_default_assumption(gold_df, "energy_prices", 0.0)
    st.session_state.vix_adj = get_default_assumption(gold_df, "tail_risk_event", 20.0)

inflation_adj = st.sidebar.slider("Inflation 💹 (%)", 0.0, 10.0, st.session_state.inflation_adj, 0.1, key="inflation_adj")
usd_adj = st.sidebar.slider("USD Strength 💵 (%)", -10.0, 10.0, st.session_state.usd_adj, 0.1, key="usd_adj")
oil_adj = st.sidebar.slider("Oil Price 🛢️ (%)", -50.0, 50.0, st.session_state.oil_adj, 0.1, key="oil_adj")
vix_adj = st.sidebar.slider("VIX / Volatility 🚨", 0.0, 100.0, st.session_state.vix_adj, 1.0, key="vix_adj")
st.sidebar.button("Reset to Predicted Values", on_click=reset_sliders)

def apply_what_if(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    adj = 1 + st.session_state.inflation_adj * 0.01 - st.session_state.usd_adj * 0.01 + st.session_state.oil_adj * 0.01 - st.session_state.vix_adj * 0.005
    df_copy = df.copy()
    if "predicted_price" in df_copy.columns:
        try:
            df_copy["predicted_price"] = df_copy["predicted_price"].astype(float) * adj
            df_copy["target_price"] = df_copy["predicted_price"]
        except Exception:
            pass
    return df_copy

gold_df_adj = apply_what_if(gold_df)
btc_df_adj = apply_what_if(btc_df)


def generate_summary(asset_df: pd.DataFrame, asset_name: str) -> str:
    if asset_df is None or asset_df.empty:
        return f"No data for {asset_name}"
    last_row = asset_df.iloc[-1]
    return f"**{asset_name} Market Summary:** Signal: {last_row.get('signal', 'N/A')} | Trend: {last_row.get('trend', 'N/A')} | Target Price: {last_row.get('target_price', 'N/A')}"


# -------------------------------------------------------------------
# NOTE: CANDLESTICK PATTERN DETECTION & SYNTHESIS
# -------------------------------------------------------------------
# All candlestick pattern detection and synthetic-prediction logic has been moved to
# candlestick_predictions.py. This file will call render_candlestick_dashboard(...) when
# the user chooses the "Candlestick Predictions" menu. This keeps app.py stable and short
# of domain-specific candlestick code.


# -------------------------------------------------------------------
# MAIN MENU: add "Candlestick Predictions"
# -------------------------------------------------------------------
menu = st.sidebar.radio(
    "📊 Choose Dashboard",
    ["Gold & Bitcoin", "AI Forecast", "Candlestick Predictions", "Jobs", "Real Estate Bot",  "Formation Word & Excel", "📅 Task Planner"]
)


# small header helper
def render_header(title: str, subtitle: Optional[str] = None):
    st.title(title)
    if subtitle:
        st.markdown(f"_{subtitle}_")


# -------------------------------------------------------------------
# GOLD & BITCOIN DASHBOARD (original behavior preserved)
# -------------------------------------------------------------------
if menu == "Gold & Bitcoin":
    render_header("🌸 Gold & Bitcoin Market Dashboard (Pastel Theme)", "Combined view of predictions, actuals and AI forecasts")
    col1, col2 = st.columns(2)

    for col, df, name, actual_col in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"], ["gold_actual", "bitcoin_actual"]):
        with col:
            st.subheader(name)
            if df is not None and not df.empty:
                last_signal = df.get("signal", pd.Series(["Hold"])).iloc[-1]
                st.markdown(alert_badge(last_signal, name), unsafe_allow_html=True)
                last_trend = df.get("trend", pd.Series(["Neutral ⚖️"])).iloc[-1]
                st.markdown(f"**Market Trend:** {last_trend}")
                st.markdown(generate_summary(df, name))
                target_price_card(df["target_price"].iloc[-1] if "target_price" in df.columns else "N/A", name, df["target_horizon"].iloc[-1] if "target_horizon" in df.columns else "Days")
                try:
                    explanation_card(df, name)
                    assumptions_card(df, name)
                except Exception:
                    st.info("Error showing explanation or assumptions.")

                n_steps = 7
                try:
                    df_ai = predict_next_n(asset_name=name, n_steps=n_steps)
                except Exception:
                    df_ai = pd.DataFrame()

                theme = ASSET_THEMES.get(name, ASSET_THEMES["Gold"])
                fig = go.Figure()

                if "timestamp" in df.columns and "actual" in df.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers", name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                    except Exception:
                        pass

                if "timestamp" in df.columns and "predicted_price" in df.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers", name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                    except Exception:
                        pass

                if not df_ai.empty and "timestamp" in df_ai.columns and "predicted_price" in df_ai.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"], mode="lines+markers", name="AI Forecast", line=dict(color=theme["chart_ai"], dash="dot")))
                    except Exception:
                        pass

                fig.update_layout(title=f"{name} Prices: Actual + Predicted + AI Forecast", xaxis_title="Date", yaxis_title="Price", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")


# -------------------------------------------------------------------
# AI FORECAST DASHBOARD (candlestick detection moved out)
# -------------------------------------------------------------------
elif menu == "AI Forecast":
    render_header("🤖 AI Forecast Dashboard", "AI-predicted prices and historical comparisons")

    n_steps = st.sidebar.number_input("Forecast next days", min_value=1, max_value=30, value=7)
    col_left, col_right = st.columns(2)

    # GOLD (left)
    with col_left:
        asset = "Gold"
        actual_col = "gold_actual"
        st.subheader(asset)
        st.markdown("**Historical AI Predictions vs Actual**")
        df_hist_actual = pd.DataFrame()
        if actual_col in df_actual.columns:
            df_hist_actual = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}).dropna()
        df_hist_pred = pd.DataFrame()
        if "asset" in df_ai_pred_log.columns:
            df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == asset][["timestamp", "predicted_price"]].dropna()

        if not df_hist_actual.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_hist_actual["timestamp"], y=df_hist_actual["actual"], mode="lines+markers", name="Actual Price", line=dict(width=2)))
            fig_cmp.add_trace(go.Scatter(x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"], mode="lines+markers", name="Predicted Price", line=dict(dash="dot")))
            fig_cmp.update_layout(title=f"{asset} – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white", height=350)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info(f"No historical data available for {asset} or no AI predictions logged.")

        st.markdown("**Future AI Forecast**")
        try:
            df_ai_future = predict_next_n(asset_name=asset, n_steps=n_steps)
        except Exception:
            df_ai_future = pd.DataFrame()

        if not df_ai_future.empty:
            fig = go.Figure()
            if not df_hist_actual.empty:
                fig.add_trace(go.Scatter(x=df_hist_actual["timestamp"], y=df_hist_actual["actual"], mode="lines+markers", name="Actual", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=df_ai_future["timestamp"], y=df_ai_future["predicted_price"], mode="lines+markers", name="AI Predicted", line=dict(dash="dash")))
            fig.update_layout(title=f"{asset} AI Forecast vs Actual", xaxis_title="Date", yaxis_title="Price", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No AI forecast available for {asset}.")

    # BITCOIN (right) - Note: candlestick-based predictions & detection have been removed from this menu.
    with col_right:
        asset = "Bitcoin"
        st.subheader(asset)

        # Prepare close time-series for Bitcoin
        df_close = pd.DataFrame()
        if "bitcoin_close" in df_actual.columns:
            df_close = df_actual[["timestamp", "bitcoin_close"]].rename(columns={"bitcoin_close": "actual"}).dropna()
        elif "bitcoin_actual" in df_actual.columns:
            df_close = df_actual[["timestamp", "bitcoin_actual"]].rename(columns={"bitcoin_actual": "actual"}).dropna()

        st.markdown("**Price Series (Bitcoin Close) & AI Forecast**")
        df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == "Bitcoin"][["timestamp", "predicted_price"]].dropna() if "asset" in df_ai_pred_log.columns else pd.DataFrame()

        # Show combined line chart of actual closes and AI predicted points
        if not df_close.empty or not df_hist_pred.empty:
            fig_btc = go.Figure()
            if not df_close.empty:
                fig_btc.add_trace(go.Scatter(x=df_close["timestamp"], y=df_close["actual"], mode="lines+markers", name="Actual Close", line=dict(width=2)))
            if not df_hist_pred.empty:
                fig_btc.add_trace(go.Scatter(x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"], mode="lines+markers", name="AI Predictions", line=dict(dash="dot")))
            try:
                df_ai_future = predict_next_n(asset_name="Bitcoin", n_steps=n_steps)
            except Exception:
                df_ai_future = pd.DataFrame()
            if not df_ai_future.empty and "timestamp" in df_ai_future.columns and "predicted_price" in df_ai_future.columns:
                fig_btc.add_trace(go.Scatter(x=df_ai_future["timestamp"], y=df_ai_future["predicted_price"], mode="lines+markers", name="AI Forecast (future)", line=dict(dash="dash")))
            fig_btc.update_layout(title="Bitcoin – Close Price & AI Forecast", xaxis_title="Date", yaxis_title="Price", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=480)
            st.plotly_chart(fig_btc, use_container_width=True)
        else:
            st.info("No Bitcoin close-series or AI predictions available for plotting.")

        # Historical comparison small table
        st.markdown("**Historical AI Predictions vs Actual (table)**")
        if not df_close.empty and not df_hist_pred.empty:
            try:
                joined = pd.merge_asof(df_hist_pred.sort_values("timestamp"), df_close.sort_values("timestamp").rename(columns={"actual": "actual_price"}), on="timestamp", direction="backward", tolerance=pd.Timedelta("1D"))
            except Exception as e:
                # Force timestamps to datetime and retry
                df_hist_pred["timestamp"] = pd.to_datetime(df_hist_pred["timestamp"], errors="coerce")
                df_close["timestamp"] = pd.to_datetime(df_close["timestamp"], errors="coerce")
                joined = pd.merge(
                df_hist_pred.sort_values("timestamp"),
                df_close.sort_values("timestamp").rename(columns={"actual": "actual_price"}),
                how="left",
                on="timestamp"
                )
                st.dataframe(joined.head(50))
        else:
            st.info("Insufficient data to show historical table for Bitcoin.")


# -------------------------------------------------------------------
# CANDLESTICK PREDICTIONS MENU (delegates to candlestick_predictions.py)
# -------------------------------------------------------------------
elif menu == "Candlestick Predictions":
    # This function is implemented in candlestick_predictions.py
    # It contains all candlestick pattern detection, weekly aggregation,
    # synthetic candle generation, plotting and logging features.
    render_candlestick_dashboard(df_actual)
    render_daily_candlestick_dashboard(df_actual)    


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
if section == "📅 Task Planner":
    render_task_planner()

# -------------------------------------------------------------------
# FOOTER: diagnostics and downloads
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("### Diagnostics & Data Files")
colA, colB, colC = st.columns([1, 1, 2])

with colA:
    st.markdown("**Data file paths**")
    st.write(f"PREDICTION_FILE = `{PREDICTION_FILE}`")
    st.write(f"AI_PRED_FILE = `{AI_PRED_FILE}`")
    st.write(f"ACTUAL_FILE = `{ACTUAL_FILE}`")

with colB:
    if st.button("Reload CSVs (refresh)"):
        try:
            df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
            df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
            df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price", "method"])
            ensure_timestamp(df_pred)
            ensure_timestamp(df_actual)
            ensure_timestamp(df_ai_pred_log)
            st.success("CSVs reloaded.")
        except Exception as e:
            st.warning(f"Reload failed: {e}")

with colC:
    st.markdown("**Download AI predictions (last 90 days)**")
    try:
        if not df_ai_pred_log.empty:
            tmp = df_ai_pred_log.copy()
            try:
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                tmp = tmp[tmp["timestamp"] >= cutoff]
            except Exception:
                pass
            csv_bytes = tmp.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (AI predictions 90d)", data=csv_bytes, file_name="ai_predictions_recent.csv", mime="text/csv")
        else:
            st.info("No AI predictions logged yet.")
    except Exception:
        st.info("No AI predictions available for download.")


# -------------------------------------------------------------------
# END NOTES
# -------------------------------------------------------------------
st.markdown("""
---
**Notes & Disclaimers**
- Candlestick-based predictions are rule-based heuristics for demonstration and educational purposes only. They are **not** financial advice.
- Synthetic predicted candles are simplistic: open = prior close, close = open * (1 + drift), high/low are small offsets. You can replace with any model in candlestick_predictions.py.
- Logging writes to `data/ai_predictions_log.csv`. Ensure the Streamlit process can write to `data/`.
- If external modules (`jobs_app`, `ai_predictor`, `real_estate_bot`, `candlestick_predictions`) are missing, the app will show placeholders but the rest of the UI will function.
""")
