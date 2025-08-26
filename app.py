# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
import numpy as np
from datetime import datetime, timedelta

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold, Bitcoin & Dubai Dashboard", layout="wide")

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
    "inflation": "💹",
    "real_rates": "🏦",
    "bond_yields": "📈",
    "energy_prices": "🛢️",
    "usd_strength": "💵",
    "liquidity": "💧",
    "equity_flows": "📊",
    "regulation": "🏛️",
    "adoption": "🤝",
    "currency_instability": "⚖️",
    "recession_probability": "📉",
    "tail_risk_event": "🚨",
    "geopolitics": "🌍"
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

    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))
    asset_pred["target_horizon"] = "Days"

    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"]=="Buy" else (row["predicted_price"] if row["signal"]=="Sell" else row["actual"]),
        axis=1
    )

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# UTILS
# ------------------------------
def color_signal(val):
    if val == "Buy":
        color = "#1f77b4"
    elif val == "Sell":
        color = "#ff7f0e"
    else:
        color = "gray"
    return f'color: {color}; font-weight:bold; text-align:center'

def alert_badge(signal):
    if signal == "Buy":
        return f'<div style="background-color:#1f77b4;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">BUY</div>'
    elif signal == "Sell":
        return f'<div style="background-color:#ff7f0e;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">SELL</div>'
    else:
        return f'<div style="background-color:gray;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">HOLD</div>'

def target_price_card(price, asset_name):
    st.markdown(f"""
        <div style='background-color:#ffd700;color:black;padding:12px;font-size:22px;text-align:center;border-radius:8px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price}
        </div>
        """, unsafe_allow_html=True)

def assumptions_card(asset_df, asset_name):
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
    icons = [INDICATOR_ICONS.get(k,"❔") for k in indicators]
    colors = ["#1f77b4" if v>0 else "#ff7f0e" if v<0 else "gray" for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(
            x=[f"{icon} {ind}"],
            y=[val],
            marker_color=color,
            text=[f"{val:.2f}"],
            textposition='auto'
        ))
    fig.update_layout(title=f"{asset_name} Assumptions & Target ({target_horizon})",
                      yaxis_title="Weight / Impact")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# SYNTHETIC DUBAI DATA
# ------------------------------
def generate_synthetic_dubai_data():
    areas = ["Dubai Marina", "Downtown Dubai", "Business Bay", "Jumeirah Village Circle", "Deira"]
    types = ["Studio", "1BR", "2BR", "3BR"]
    rows = []
    today = datetime.utcnow().date()
    for a in areas:
        for t in types:
            avg_price = np.round(np.random.uniform(200000, 2000000), 2)
            tx = int(np.random.uniform(5, 200))
            rows.append([a, t, avg_price, tx, today])
    df = pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])
    return df

def generate_synthetic_population():
    years = list(range(2010, datetime.utcnow().year + 1))
    pop = [2000000 + (i - 2010) * 80000 for i in years]
    return pd.DataFrame({"Year": years, "Population": pop})

# ------------------------------
# DUBAI DASHBOARD
# ------------------------------
def dubai_dashboard():
    st.title("🏙️ Dubai Real Estate Dashboard")
    submenu = st.selectbox("Choose Category", ["Market Overview", "Prices", "Sales", "Market Health", "Population"])

    df_market = generate_synthetic_dubai_data()
    df_pop = generate_synthetic_population()

    if submenu == "Market Overview":
        st.subheader("📊 Market Overview")
        avg_price = int(df_market["avg_price"].mean())
        total_tx = int(df_market["transactions"].sum())
        areas_covered = df_market["area"].nunique()
        last_date = df_market["date"].max()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Price", f"{avg_price:,}")
        col2.metric("Transactions (sum)", f"{total_tx:,}")
        col3.metric("Areas Covered", areas_covered)
        col4.metric("Last Update", str(last_date))

    elif submenu == "Prices":
        st.subheader("📈 Property Prices")
        st.dataframe(df_market)
    elif submenu == "Sales":
        st.subheader("💰 Sales & Transactions")
        sales_agg = df_market.groupby("area", as_index=False).transactions.sum().sort_values("transactions", ascending=False)
        st.dataframe(sales_agg)
    elif submenu == "Market Health":
        st.subheader("🩺 Market Health")
        # Simple synthetic score
        avg_price = df_market["avg_price"].mean()
        total_tx = df_market["transactions"].sum()
        score = max(0, min(100, 50 + (avg_price/2000000)*20 - np.log1p(total_tx)))
        st.metric("Market Health Score (0 worst → 100 best)", f"{int(score)}")
    elif submenu == "Population":
        st.subheader("👥 Population")
        st.dataframe(df_pop)

# ------------------------------
# MAIN MENU
# ------------------------------
menu = st.sidebar.radio("📊 Choose Dashboard", ["Gold & Bitcoin", "Dubai Real Estate"])

if menu == "Gold & Bitcoin":
    st.title("📊 Gold & Bitcoin Market Dashboard")
    col1, col2 = st.columns(2)
    for col, df, name in zip([col1, col2], [gold_df, btc_df], ["Gold", "Bitcoin"]):
        with col:
            st.subheader(name)
            if not df.empty:
                last_signal = df["signal"].iloc[-1]
                st.markdown(alert_badge(last_signal), unsafe_allow_html=True)
                last_trend = df["trend"].iloc[-1] if df["trend"].iloc[-1] else "Neutral ⚖️"
                st.markdown(f"**Market Trend:** {last_trend}")
                target_price_card(df["target_price"].iloc[-1], name)
                assumptions_card(df, name)
            else:
                st.info(f"No {name} data available yet.")
elif menu == "Dubai Real Estate":
    dubai_dashboard()
