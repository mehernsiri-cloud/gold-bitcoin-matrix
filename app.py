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
    # Align predictions and actuals on hourly timestamp
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    act_df["timestamp"] = pd.to_datetime(act_df["timestamp"])
    merged = pd.merge_asof(
        pred_df.sort_values("timestamp"),
        act_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    col1, col2 = st.columns(2)

    # Gold
    with col1:
        gold_df = merged[merged["asset"]=="Gold"].copy()
        st.subheader("Gold")
        if not gold_df.empty:
            st.metric("Predicted Price (USD)", f"${gold_df['predicted_price'].iloc[-1]:,.2f}")
            st.metric("Actual Price (USD)", f"${gold_df['gold_actual'].iloc[-1]:,.2f}")
            st.metric("Risk", gold_df['risk'].iloc[-1])
            st.line_chart(gold_df.set_index("timestamp")[["predicted_price","gold_actual"]])

    # Bitcoin
    with col2:
        btc_df = merged[merged["asset"]=="Bitcoin"].copy()
        st.subheader("Bitcoin")
        if not btc_df.empty:
            st.metric("Predicted Price (USD)", f"${btc_df['predicted_price'].iloc[-1]:,.2f}")
            st.metric("Actual Price (USD)", f"${btc_df['bitcoin_actual'].iloc[-1]:,.2f}")
            st.metric("Risk", btc_df['risk'].iloc[-1])
            st.line_chart(btc_df.set_index("timestamp")[["predicted_price","bitcoin_actual"]])

    st.caption(f"Last updated: {merged['timestamp'].max()}")
