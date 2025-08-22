import streamlit as st
import pandas as pd
import os

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

st.set_page_config(page_title="Gold & Bitcoin Predictions", layout="wide")
st.title("Gold & Bitcoin Predictions Dashboard")

def load_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame()

pred_df = load_csv(PREDICTION_FILE)
act_df = load_csv(ACTUAL_FILE)

if pred_df.empty or act_df.empty:
    st.info("No data available yet.")
else:
    latest_ts = pred_df["timestamp"].max()
    latest_pred = pred_df[pred_df["timestamp"]==latest_ts].set_index("asset")
    latest_act = act_df.iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gold")
        st.metric("Predicted Price (USD)", f"${latest_pred.loc['Gold','predicted_price']:,.2f}")
        st.metric("Actual Price (USD)", f"${latest_act['gold_actual']:,.2f}")
        st.metric("Risk", latest_pred.loc['Gold','risk'])

        gold_df = pred_df[pred_df["asset"]=="Gold"].copy()
        gold_df["actual"] = act_df["gold_actual"].iloc[-len(gold_df):].values
        st.line_chart(gold_df.set_index("timestamp")[["predicted_price","actual"]])

    with col2:
        st.subheader("Bitcoin")
        st.metric("Predicted Price (USD)", f"${latest_pred.loc['Bitcoin','predicted_price']:,.2f}")
        st.metric("Actual Price (USD)", f"${latest_act['bitcoin_actual']:,.2f}")
        st.metric("Risk", latest_pred.loc['Bitcoin','risk'])

        btc_df = pred_df[pred_df["asset"]=="Bitcoin"].copy()
        btc_df["actual"] = act_df["bitcoin_actual"].iloc[-len(btc_df):].values
        st.line_chart(btc_df.set_index("timestamp")[["predicted_price","actual"]])

    st.caption(f"Last updated: {latest_ts}")
