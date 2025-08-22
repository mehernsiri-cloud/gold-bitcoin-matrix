import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_DIR = "data"
PRED_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACT_FILE = os.path.join(DATA_DIR, "actual_data.csv")

st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide")
st.title("Gold and Bitcoin Predictions vs Actuals")

# ------------------------------
# Load CSVs safely
# ------------------------------
if os.path.exists(PRED_FILE):
    pred_df = pd.read_csv(PRED_FILE, parse_dates=["timestamp"])
else:
    pred_df = pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])

if os.path.exists(ACT_FILE):
    act_df = pd.read_csv(ACT_FILE, parse_dates=["timestamp","date"])
else:
    act_df = pd.DataFrame()

# ------------------------------
# Layout
# ------------------------------
gold_df = pred_df[pred_df["asset"]=="Gold"]
bitcoin_df = pred_df[pred_df["asset"]=="Bitcoin"]

if not act_df.empty:
    gold_df = gold_df.merge(act_df[["timestamp","gold_actual"]], on="timestamp", how="left")
    bitcoin_df = bitcoin_df.merge(act_df[["timestamp","bitcoin_actual"]], on="timestamp", how="left")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gold")
    if not gold_df.empty:
        st.dataframe(gold_df[["timestamp","predicted_price","gold_actual","volatility","risk"]])
        fig, ax = plt.subplots()
        ax.plot(gold_df["timestamp"], gold_df["predicted_price"], label="Predicted", marker='o')
        if "gold_actual" in gold_df:
            ax.plot(gold_df["timestamp"], gold_df["gold_actual"], label="Actual", marker='x')
        ax.set_ylabel("Price")
        ax.set_xlabel("Timestamp")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No Gold data available.")

with col2:
    st.subheader("Bitcoin")
    if not bitcoin_df.empty:
        st.dataframe(bitcoin_df[["timestamp","predicted_price","bitcoin_actual","volatility","risk"]])
        fig, ax = plt.subplots()
        ax.plot(bitcoin_df["timestamp"], bitcoin_df["predicted_price"], label="Predicted", marker='o')
        if "bitcoin_actual" in bitcoin_df:
            ax.plot(bitcoin_df["timestamp"], bitcoin_df["bitcoin_actual"], label="Actual", marker='x')
        ax.set_ylabel("Price")
        ax.set_xlabel("Timestamp")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No Bitcoin data available.")
