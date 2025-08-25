# app.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import ast

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Predictions", layout="wide")

# ------------------------------
# DATA FILES
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# LOAD PREDICTIONS
# ------------------------------
if os.path.exists(PREDICTION_FILE):
    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
else:
    df_pred = pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "volatility", "risk", "assumptions", "target_horizon"])

# ------------------------------
# LOAD ACTUALS
# ------------------------------
if os.path.exists(ACTUAL_FILE):
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])
else:
    df_actual = pd.DataFrame(columns=["timestamp", "gold_actual", "bitcoin_actual"])

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_data(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if not df_actual.empty and actual_col in df_actual.columns:
        # Align by timestamp length (last rows)
        asset_pred = asset_pred.tail(len(df_actual))
        asset_pred["actual"] = df_actual[actual_col].tail(len(asset_pred)).values
    else:
        asset_pred["actual"] = None
    return asset_pred

gold_df = merge_data("Gold", "gold_actual")
btc_df = merge_data("Bitcoin", "bitcoin_actual")

# ------------------------------
# BUY/SELL ALERT FUNCTION
# ------------------------------
def buy_sell_alert(asset_df):
    if asset_df.empty:
        return "N/A", "gray"
    last_row = asset_df.iloc[-1]
    predicted = last_row["predicted_price"]
    actual = last_row["actual"]
    if actual is None:
        return "N/A", "gray"
    # Simple strategy: predicted > actual → Buy, predicted < actual → Sell
    if predicted > actual:
        return "Buy", "green"
    elif predicted < actual:
        return "Sell", "red"
    else:
        return "Hold", "yellow"

# ------------------------------
# ASSUMPTIONS / TARGET PORTLET
# ------------------------------
def display_assumptions(asset_name, asset_df):
    assumptions = asset_df["assumptions"].iloc[-1] if not asset_df.empty else {}
    target_horizon = asset_df["target_horizon"].iloc[-1] if not asset_df.empty else "N/A"

    if isinstance(assumptions, str):
        try:
            assumptions = ast.literal_eval(assumptions)
        except:
            assumptions = {}

    if assumptions:
        indicators = list(assumptions.keys())
        values = [assumptions[k] for k in indicators]
        colors = ["green" if v > 0 else "red" if v < 0 else "yellow" for v in values]

        fig = go.Figure([go.Bar(
            x=indicators,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='auto'
        )])
        fig.update_layout(title=f"{asset_name} Assumptions & Target ({target_horizon})", yaxis_title="Weight / Impact")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No assumptions available for {asset_name}")

# ------------------------------
# DASHBOARD LAYOUT
# ------------------------------
st.title("Gold & Bitcoin Predictions Dashboard")

col1, col2 = st.columns(2)

# ------------------------------
# GOLD SECTION
# ------------------------------
with col1:
    st.subheader("Gold")
    action, color = buy_sell_alert(gold_df)
    st.markdown(f"<h1 style='color:{color}'>{action}</h1>", unsafe_allow_html=True)
    if not gold_df.empty:
        st.dataframe(gold_df.tail(2)[["timestamp","predicted_price","actual","volatility","risk"]])
        fig_gold = px.line(
            gold_df,
            x="timestamp",
            y=["predicted_price","actual"],
            labels={"value": "Price", "timestamp": "Timestamp"},
            title="Gold: Predicted vs Actual"
        )
        st.plotly_chart(fig_gold, use_container_width=True)
    else:
        st.info("No Gold data available yet.")
    display_assumptions("Gold", gold_df)

# ------------------------------
# BITCOIN SECTION
# ------------------------------
with col2:
    st.subheader("Bitcoin")
    action, color = buy_sell_alert(btc_df)
    st.markdown(f"<h1 style='color:{color}'>{action}</h1>", unsafe_allow_html=True)
    if not btc_df.empty:
        st.dataframe(btc_df.tail(2)[["timestamp","predicted_price","actual","volatility","risk"]])
        fig_btc = px.line(
            btc_df,
            x="timestamp",
            y=["predicted_price","actual"],
            labels={"value": "Price", "timestamp": "Timestamp"},
            title="Bitcoin: Predicted vs Actual"
        )
        st.plotly_chart(fig_btc, use_container_width=True)
    else:
        st.info("No Bitcoin data available yet.")
    display_assumptions("Bitcoin", btc_df)
