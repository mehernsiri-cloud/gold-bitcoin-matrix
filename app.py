# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
from jobs_app import jobs_dashboard  # import Jobs dashboard
from datetime import timedelta

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide")

# ------------------------------
# DATA FILES
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# LOAD DATA SAFELY
# ------------------------------
def load_csv_safe(path, default_cols):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=True)
    else:
        df = pd.DataFrame(columns=default_cols)
    return df

df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual"])

# ------------------------------
# LOAD WEIGHTS / ASSUMPTIONS
# ------------------------------
if os.path.exists(WEIGHT_FILE):
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
else:
    weights = {"gold": {}, "bitcoin": {}}

# ------------------------------
# EMOJI MAPPING
# ------------------------------
INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

# ------------------------------
# THEME PER ASSET
# ------------------------------
ASSET_THEMES = {
    "Gold": {
        "buy": "#FFD700", "sell": "#B8860B", "hold": "#808080",
        "target_bg": "#FFF8DC", "target_text": "black",
        "assumption_pos": "#FFD700", "assumption_neg": "#B8860B",
        "chart_actual": "gold", "chart_pred": "orange"
    },
    "Bitcoin": {
        "buy": "#1E90FF", "sell": "#FF4500", "hold": "#6c757d",
        "target_bg": "#1E90FF", "target_text": "white",
        "assumption_pos": "#1E90FF", "assumption_neg": "#FF4500",
        "chart_actual": "blue", "chart_pred": "green"
    }
}

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_actual_pred(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    if actual_col in df_actual.columns:
        asset_pred = pd.merge_asof(
            asset_pred.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        asset_pred["actual"] = None

    asset_pred["predicted_price"] = pd.to_numeric(asset_pred["predicted_price"], errors='coerce')
    asset_pred["actual"] = pd.to_numeric(asset_pred["actual"], errors='coerce')

    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if row["predicted_price"] > row["actual"] else ("Sell" if row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    asset_pred["trend"] = ""
    if len(asset_pred) >= 3:
        last3 = asset_pred["predicted_price"].tail(3)
        if last3.is_monotonic_increasing:
            asset_pred["trend"] = "Bullish 📈"
        elif last3.is_monotonic_decreasing:
            asset_pred["trend"] = "Bearish 📉"
        else:
            asset_pred["trend"] = "Neutral ⚖️"

    # store assumptions
    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))

    # set target horizon dynamically
    horizon = "Days"
    if "volatility" in asset_pred.columns and not asset_pred["volatility"].isna().all():
        avg_vol = asset_pred["volatility"].mean()
        if avg_vol < 0.02:
            horizon = "Years"
        elif avg_vol < 0.05:
            horizon = "Months"
        else:
            horizon = "Days"
    asset_pred["target_horizon"] = horizon

    # target price
    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"] == "Buy" else (row["predicted_price"] if row["signal"] == "Sell" else row["actual"]),
        axis=1
    )

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# UTILS WITH THEMES
# ------------------------------
def alert_badge(signal, asset_name):
    theme = ASSET_THEMES[asset_name]
    if signal == "Buy":
        color = theme["buy"]
        text = "BUY"
    elif signal == "Sell":
        color = theme["sell"]
        text = "SELL"
    else:
        color = theme["hold"]
        text = "HOLD"
    return f'<div style="background-color:{color};color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">{text}</div>'

def target_price_card(price, asset_name, horizon):
    theme = ASSET_THEMES[asset_name]
    st.markdown(f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:8px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price} <br>⏳ Horizon: {horizon}
        </div>
        """, unsafe_allow_html=True)

def explanation_card(asset_df, asset_name):
    if asset_df.empty:
        return
    assumptions_str = asset_df["assumptions"].iloc[-1]
    try:
        assumptions = eval(assumptions_str) if assumptions_str else {}
    except:
        assumptions = {}
    if not assumptions:
        return
    strongest = max(assumptions.items(), key=lambda x: abs(x[1]))
    indicator, impact = strongest
    direction = "upward 📈" if impact > 0 else "downward 📉"
    st.markdown(f"""
    <div style='background-color:#f8f9fa;padding:10px;border-radius:8px;margin-bottom:10px'>
    🔍 **Forecast for {asset_name}:**  
    The outlook suggests a **{direction} trend** mainly driven by **{indicator} {INDICATOR_ICONS.get(indicator,"")}**.
    </div>
    """, unsafe_allow_html=True)

def assumptions_card(asset_df, asset_name):
    theme = ASSET_THEMES[asset_name]
    if asset_df.empty:
        st.info(f"No assumptions available for {asset_name}")
        return
    assumptions_str = asset_df["assumptions"].iloc[-1]
    target_horizon = asset_df["target_horizon"].iloc[-1]
    try:
        assumptions = eval(assumptions_str) if assumptions_str else {}
    except:
        assumptions = {}
    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return
    indicators = list(assumptions.keys())
    values = [assumptions[k] for k in indicators]
    icons = [INDICATOR_ICONS.get(k, "❔") for k in indicators]
    colors = [theme["assumption_pos"] if v > 0 else theme["assumption_neg"] if v < 0 else theme["hold"] for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(
            x=[f"{icon} {ind}"], y=[val],
            marker_color=color, text=[f"{val:.2f}"], textposition='auto'
        ))
    fig.update_layout(title=f"{asset_name} Assumptions & Target ({target_horizon})",
                      yaxis_title="Weight / Impact")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# WHAT-IF SLIDERS
# ------------------------------
st.sidebar.header("🔧 What-If Scenario")
def get_default_assumption(df, key):
    if df.empty:
        return 0.0
    try:
        last_assumptions = eval(df["assumptions"].iloc[-1])
        return last_assumptions.get(key, 0.0)
    except:
        return 0.0

inflation_default = get_default_assumption(gold_df, "inflation") or 2.5
usd_default = get_default_assumption(gold_df, "usd_strength") or 0.0
oil_default = get_default_assumption(gold_df, "energy_prices") or 0.0
vix_default = get_default_assumption(gold_df, "tail_risk_event") or 20.0

inflation_adj = st.sidebar.slider("Inflation 💹 (%)", 0.0, 10.0, inflation_default, 0.1)
usd_adj = st.sidebar.slider("USD Strength 💵 (%)", -10.0, 10.0, usd_default, 0.1)
oil_adj = st.sidebar.slider("Oil Price 🛢️ (%)", -50.0, 50.0, oil_default, 0.1)
vix_adj = st.sidebar.slider("VIX / Volatility 🚨", 0.0, 100.0, vix_default, 1.0)

if st.sidebar.button("Reset to Predicted Values"):
    inflation_adj, usd_adj, oil_adj, vix_adj = inflation_default, usd_default, oil_default, vix_default

def apply_what_if(df):
    if df.empty:
        return df
    adj = 1 + inflation_adj * 0.01 - usd_adj * 0.01 + oil_adj * 0.01 - vix_adj * 0.005
    df = df.copy()
    df["predicted_price"] = df["predicted_price"] * adj
    df["target_price"] = df["predicted_price"]
    return df

gold_df_adj = apply_what_if(gold_df)
btc_df_adj = apply_what_if(btc_df)

def generate_summary(asset_df, asset_name):
    if asset_df.empty:
        return f"No data for {asset_name}"
    last_row = asset_df.iloc[-1]
    return f"**{asset_name} Market Summary:** Signal: {last_row['signal']} | Trend: {last_row['trend']} | Target Price: {last_row['target_price']}"

# ------------------------------
# MAIN MENU
# ------------------------------
menu = st.sidebar.radio("📊 Choose Dashboard", ["Gold & Bitcoin", "Jobs"])

if menu == "Gold & Bitcoin":
    st.title("📊 Gold & Bitcoin Market Dashboard")
    col1, col2 = st.columns(2)

    for col, df, name in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"]):
        with col:
            st.subheader(name)
            if not df.empty:
                last_signal = df["signal"].iloc[-1]
                st.markdown(alert_badge(last_signal, name), unsafe_allow_html=True)
                last_trend = df["trend"].iloc[-1] if df["trend"].iloc[-1] else "Neutral ⚖️"
                st.markdown(f"**Market Trend:** {last_trend}")
                st.markdown(generate_summary(df, name))
                target_price_card(df["target_price"].iloc[-1], name, df["target_horizon"].iloc[-1])
                explanation_card(df, name)

                display_df = df[["timestamp", "actual", "predicted_price", "volatility", "risk", "signal"]].tail(2)
                st.dataframe(display_df)

                assumptions_card(df, name)

                # Chart with asset-specific colors
                theme = ASSET_THEMES[name]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers",
                                         name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers",
                                         name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")
elif menu == "Jobs":
    jobs_dashboard()
