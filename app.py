import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# -----------------------------
# Load Data Functions
# -----------------------------
@st.cache_data
def load_predictions():
    if os.path.exists(PREDICTION_FILE) and os.path.getsize(PREDICTION_FILE) > 0:
        df = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
        return df
    return pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])

@st.cache_data
def load_actuals():
    if os.path.exists(ACTUAL_FILE) and os.path.getsize(ACTUAL_FILE) > 0:
        df = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])
        return df
    return pd.DataFrame(columns=["timestamp","date","gold_actual","bitcoin_actual"])

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Gold & Bitcoin Predictions", layout="wide")
st.title("Gold & Bitcoin Predictions Dashboard")

pred_df = load_predictions()
act_df = load_actuals()

if pred_df.empty or act_df.empty:
    st.info("No prediction or actual data available yet.")
else:
    # Get latest timestamp
    latest_timestamp = pred_df["timestamp"].max()
    latest_pred = pred_df[pred_df["timestamp"] == latest_timestamp].set_index("asset")
    latest_actual = act_df[act_df["timestamp"] == act_df["timestamp"].max()].iloc[0]

    # -----------------------------
    # Columns layout: Gold | Bitcoin
    # -----------------------------
    col1, col2 = st.columns(2)

    # Gold Portlet
    with col1:
        st.subheader("Gold")
        gold_pred = latest_pred.loc["Gold", "predicted_price"]
        gold_actual = latest_actual["gold_actual"]

        st.metric(label="Predicted Price (USD)", value=f"${gold_pred:,.2f}")
        st.metric(label="Actual Price (USD)", value=f"${gold_actual:,.2f}")
        st.metric(label="Risk", value=latest_pred.loc["Gold", "risk"])

        # Gold Chart
        gold_chart_df = pred_df[pred_df["asset"] == "Gold"].copy()
        gold_chart_df["actual"] = act_df["gold_actual"].iloc[-len(gold_chart_df):].values
        st.line_chart(
            gold_chart_df.set_index("timestamp")[["predicted_price","actual"]],
            use_container_width=True
        )

    # Bitcoin Portlet
    with col2:
        st.subheader("Bitcoin")
        btc_pred = latest_pred.loc["Bitcoin", "predicted_price"]
        btc_actual = latest_actual["bitcoin_actual"]

        st.metric(label="Predicted Price (USD)", value=f"${btc_pred:,.2f}")
        st.metric(label="Actual Price (USD)", value=f"${btc_actual:,.2f}")
        st.metric(label="Risk", value=latest_pred.loc["Bitcoin", "risk"])

        # Bitcoin Chart
        btc_chart_df = pred_df[pred_df["asset"] == "Bitcoin"].copy()
        btc_chart_df["actual"] = act_df["bitcoin_actual"].iloc[-len(btc_chart_df):].values
        st.line_chart(
            btc_chart_df.set_index("timestamp")[["predicted_price","actual"]],
            use_container_width=True
        )

    st.caption(f"Last updated: {latest_timestamp}")
