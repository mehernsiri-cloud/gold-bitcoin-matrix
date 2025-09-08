# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
from datetime import timedelta
from jobs_app import jobs_dashboard
from ai_predictor import predict_next_n, compare_predictions_vs_actuals

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
        "chart_actual": "#FBC02D", "chart_pred": "#FFCC80", "chart_ai": "#FF6F61"
    },
    "Bitcoin": {
        "buy": "#BBDEFB", "sell": "#FFCDD2", "hold": "#CFD8DC",
        "target_bg": "#E3F2FD", "target_text": "black",
        "assumption_pos": "#64B5F6", "assumption_neg": "#EF9A9A",
        "chart_actual": "#42A5F5", "chart_pred": "#81D4FA", "chart_ai": "#FF6F61"
    }
}

# ------------------------------
# HELPER FUNCTIONS
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
    st.subheader(f"📖 Explanation for {asset_name}")

    if "assumptions" in asset_df.columns and not asset_df.empty:
        try:
            assumptions_str = asset_df["assumptions"].iloc[-1]
            assumptions = eval(assumptions_str) if isinstance(assumptions_str, str) else {}
        except Exception:
            assumptions = {}
    else:
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

def assumptions_card(asset_df, asset_name):
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
    except:
        assumptions = {}

    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return

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
    theme = ASSET_THEMES[asset_name]
    colors = [theme["assumption_pos"] if v > 0 else theme["assumption_neg"] if v < 0 else theme["hold"] for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(x=[f"{icon} {ind}"], y=[val], marker_color=color, text=[f"{val:.2f}"], textposition='auto'))

    fig.update_layout(
        title=f"{asset_name} Assumptions & Target ({target_horizon})",
        yaxis_title="Weight / Impact",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA"
    )
    st.plotly_chart(fig, use_container_width=True)

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
# WHAT-IF SLIDERS WITH SESSION STATE
# ------------------------------
st.sidebar.header("🔧 What-If Scenario")

def get_default_assumption(df, key, fallback):
    if df.empty:
        return fallback
    try:
        last_assumptions = eval(df["assumptions"].iloc[-1])
        return last_assumptions.get(key, fallback)
    except:
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

def apply_what_if(df):
    if df.empty:
        return df
    adj = 1 + st.session_state.inflation_adj * 0.01 - st.session_state.usd_adj * 0.01 + st.session_state.oil_adj * 0.01 - st.session_state.vix_adj * 0.005
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

    for col, df, name, actual_col in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"], ["gold_actual", "bitcoin_actual"]):
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
                assumptions_card(df, name)

                n_steps = 7
                try:
                    df_ai = predict_next_n(asset_name=name, n_steps=n_steps)
                except TypeError:
                    df_ai = pd.DataFrame()

                theme = ASSET_THEMES[name]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers",
                                         name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers",
                                         name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                if not df_ai.empty:
                    fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"], mode="lines+markers",
                                             name="AI Forecast", line=dict(color=theme["chart_ai"], dash="dot")))
                fig.update_layout(title=f"{name} Prices: Actual + Predicted + AI Forecast",
                                  xaxis_title="Date", yaxis_title="Price",
                                  plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")

elif menu == "AI Forecast":
    st.title("🤖 AI Forecast Dashboard")
    st.markdown("This dashboard shows **AI-predicted prices** based on historical data.")
    n_steps = st.sidebar.number_input("Forecast next days", min_value=1, max_value=30, value=7)

    # Two-column layout for Gold (left) and Bitcoin (right)
    col1, col2 = st.columns(2)

    for col, asset, actual_col in zip([col1, col2], ["Gold", "Bitcoin"], ["gold_actual", "bitcoin_actual"]):
        with col:
            st.subheader(asset)

            # --- Historical AI predictions vs Actual ---
            st.markdown("**Historical AI Predictions vs Actual**")
            try:
                df_cmp = compare_predictions_vs_actuals(asset)
                if not df_cmp.empty:
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatter(x=df_cmp["timestamp"], y=df_cmp["actual_price"],
                                                 mode="lines+markers", name="Actual Price",
                                                 line=dict(color="green")))
                    fig_cmp.add_trace(go.Scatter(x=df_cmp["timestamp"], y=df_cmp["predicted_price"],
                                                 mode="lines+markers", name="Predicted Price",
                                                 line=dict(color="orange", dash="dot")))
                    fig_cmp.update_layout(title=f"{asset} – Historical AI Predictions vs Actual",
                                          xaxis_title="Date", yaxis_title="Price",
                                          template="plotly_white")
                    st.plotly_chart(fig_cmp, use_container_width=True)
                else:
                    st.info(f"No historical prediction data for {asset}.")
            except Exception as e:
                st.warning(f"Could not load historical comparison for {asset}: {e}")

            # --- Future AI Forecast ---
            st.markdown("**Future AI Forecast**")
            try:
                df_ai = predict_next_n(asset_name=asset, n_steps=n_steps)
            except TypeError:
                df_ai = pd.DataFrame()
            if not df_ai.empty:
                df_hist = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["actual"],
                                         mode="lines+markers", name="Actual",
                                         line=dict(color="#42A5F5", width=2)))
                fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"],
                                         mode="lines+markers", name="AI Predicted",
                                         line=dict(color="#FF6F61", dash="dash")))
                fig.update_layout(title=f"{asset} AI Forecast vs Actual",
                                  xaxis_title="Date", yaxis_title="Price",
                                  plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No AI prediction available for {asset}.")

elif menu == "Jobs":
    jobs_dashboard()
