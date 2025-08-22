import streamlit as st
import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
PRED_FILE = os.path.join(DATA_DIR, "predictions_log.csv")

st.set_page_config(page_title="Predictions Dashboard", layout="wide")
st.title("📊 Predictions vs Actuals Dashboard")

# --- Load Data ---
@st.cache_data
def load_csv(file):
    if os.path.exists(file):
        return pd.read_csv(file)
    return pd.DataFrame()

actual_df = load_csv(ACTUAL_FILE)
pred_df = load_csv(PRED_FILE)

# --- Latest Actuals ---
if not actual_df.empty:
    latest_actual = actual_df.sort_values("date").iloc[-1]
    st.subheader("Latest Market Actuals")
    st.table(pd.DataFrame([latest_actual]))
else:
    st.warning("No actual data available yet. Run `fetch_data.py` or wait for the workflow.")

# --- Latest Predictions + Compare ---
if not pred_df.empty and not actual_df.empty:
    latest_pred = pred_df.sort_values("timestamp").iloc[-1].copy()
    # Match prediction date with actuals
    latest_pred["date"] = pd.to_datetime(latest_pred["timestamp"]).date()
    latest_actual_date = actual_df["date"].iloc[-1]

    st.subheader("Latest Predictions vs Actuals")
    merged = pd.DataFrame([{
        "date": latest_actual_date,
        "gold_predicted": latest_pred.get("gold_pred", None),
        "gold_actual": latest_actual["gold_actual"],
        "bitcoin_predicted": latest_pred.get("bitcoin_pred", None),
        "bitcoin_actual": latest_actual["bitcoin_actual"],
        "france_studio_pred": latest_pred.get("france_studio_pred", None),
        "france_studio_actual": latest_actual["france_studio_price"],
        "france_2bed_pred": latest_pred.get("france_2bed_pred", None),
        "france_2bed_actual": latest_actual["france_2bed_price"],
        "dubai_studio_pred": latest_pred.get("dubai_studio_pred", None),
        "dubai_studio_actual": latest_actual["dubai_studio_price"],
        "dubai_2bed_pred": latest_pred.get("dubai_2bed_pred", None),
        "dubai_2bed_actual": latest_actual["dubai_2bed_price"],
    }])
    st.dataframe(merged, use_container_width=True)
else:
    st.info("Waiting for both predictions and actuals to display comparison.")

# --- History Charts ---
if not actual_df.empty:
    st.subheader("📈 Historical Actual Prices")
    st.line_chart(actual_df.set_index("date")[[
        "gold_actual",
        "bitcoin_actual",
        "france_studio_price",
        "france_2bed_price",
        "dubai_studio_price",
        "dubai_2bed_price"
    ]])
