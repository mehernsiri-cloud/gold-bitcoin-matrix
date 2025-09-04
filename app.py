# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
from datetime import timedelta
from jobs_app import jobs_dashboard
from ai_predictor import predict_next_n

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
# LOAD DATA
# ------------------------------
def load_csv_safe(path, default_cols):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=True)
    else:
        df = pd.DataFrame(columns=default_cols)
    return df

df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual"])

if os.path.exists(WEIGHT_FILE):
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
else:
    weights = {"gold": {}, "bitcoin": {}}

# ------------------------------
# EMOJI ICONS
# ------------------------------
INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

# ------------------------------
# THEMES
# ------------------------------
ASSET_THEMES = {
    "Gold": {
        "buy": "#FFF9C4", "sell": "#FFE0B2", "hold": "#E0E0E0",
        "target_bg": "#FFFDE7", "target_text": "black",
        "assumption_pos": "#FFD54F", "assumption_neg": "#FFAB91",
        "chart_actual": "#FBC02D", "chart_pred": "#FFCC80"
    },
    "Bitcoin": {
        "buy": "#BBDEFB", "sell": "#FFCDD2", "hold": "#CFD8DC",
        "target_bg": "#E3F2FD", "target_text": "black",
        "assumption_pos": "#64B5F6", "assumption_neg": "#EF9A9A",
        "chart_actual": "#42A5F5", "chart_pred": "#81D4FA"
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

    asset_pred["predicted_price"] = pd.to_numeric(asset_pred["predicted_price"], errors="coerce")
    asset_pred["actual"] = pd.to_numeric(asset_pred["actual"], errors="coerce")

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

    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))

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

    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"] == "Buy" else (row["predicted_price"] if row["signal"] == "Sell" else row["actual"]),
        axis=1
    )

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# UTILS
# ------------------------------
def alert_badge(signal, asset_name):
    theme = ASSET_THEMES[asset_name]
    color = theme["buy"] if signal == "Buy" else theme["sell"] if signal == "Sell" else theme["hold"]
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'

def target_price_card(price, asset_name, horizon):
    theme = ASSET_THEMES[asset_name]
    st.markdown(f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
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
    <div style='background-color:#FAFAFA;padding:12px;border-radius:10px;margin-bottom:10px'>
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
                      yaxis_title="Weight / Impact",
                      plot_bgcolor="#FAFAFA",
                      paper_bgcolor="#FAFAFA")
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
menu = st.sidebar.radio("📊 Choose Dashboard", ["Gold & Bitcoin", "AI Forecast", "Jobs"])

if menu == "Gold & Bitcoin":
    st.title("🌸 Gold & Bitcoin Market Dashboard (Pastel Theme)")
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

                theme = ASSET_THEMES[name]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers",
                                         name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers",
                                         name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                fig.update_layout(plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")

elif menu == "AI Forecast":
    st.title("🤖 AI Forecast Dashboard")
    st.markdown("This dashboard shows **AI-predicted prices** based on historical data.")
    n_steps = st.sidebar.number_input("Forecast next days", min_value=1, max_value=30, value=7)
    
    for asset, col in [("Gold", "gold_actual"), ("Bitcoin", "bitcoin_actual")]:
        st.subheader(asset)
        df_ai = predict_next_n(df_actual, df_pred, asset, n_steps)
        if not df_ai.empty:
            st.dataframe(df_ai)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"],
                                     mode="lines+markers", name="AI Predicted",
                                     line=dict(color="#FF6F61", dash="dash")))
            fig.update_layout(plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No AI prediction available for {asset}.")

elif menu == "Jobs":
    jobs_dashboard()
