import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Gold & Bitcoin Investment Dashboard", layout="wide")
st.title("💰 Gold & Bitcoin Investment Dashboard")

DATA_DIR = "data"
PRED_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# --- Load Data ---
pred_df = pd.read_csv(PRED_FILE) if os.path.exists(PRED_FILE) else pd.DataFrame()
actual_df = pd.read_csv(ACTUAL_FILE) if os.path.exists(ACTUAL_FILE) else pd.DataFrame()

# --- Latest Actuals ---
if not actual_df.empty:
    latest_actual = actual_df.iloc[-1]
    st.subheader("📈 Latest Actual Prices")
    st.write({
        "Gold": latest_actual["gold_actual"],
        "Bitcoin": latest_actual["bitcoin_actual"]
    })
else:
    st.warning("No actual data available yet.")

# --- Latest Predictions vs Actuals ---
if not pred_df.empty and not actual_df.empty:
    latest_preds = pred_df.sort_values("timestamp").groupby("asset").tail(1)
    table_rows = []
    for _, row in latest_preds.iterrows():
        asset_name = row["asset"]
        actual_price = latest_actual.get(f"{asset_name.lower()}_actual", None)
        # Investment signal
        signal = "Hold"
        if row["predicted_price"] > actual_price * 1.01:
            signal = "Buy"
        elif row["predicted_price"] < actual_price * 0.99:
            signal = "Sell"
        table_rows.append({
            "Asset": asset_name,
            "Predicted Price": row["predicted_price"],
            "Actual Price": actual_price,
            "Volatility": row["volatility"],
            "Risk": row["risk"],
            "Signal": signal
        })
    st.subheader("📊 Latest Predictions vs Actuals")
    st.dataframe(pd.DataFrame(table_rows))

# --- Historical Charts ---
if not actual_df.empty:
    st.subheader("📉 Historical Prices")
    hist = actual_df[["date", "gold_actual", "bitcoin_actual"]].set_index("date")
    st.line_chart(hist)

if not pred_df.empty:
    st.subheader("📈 Predicted Prices")
    pred_plot = pred_df.pivot(index="timestamp", columns="asset", values="predicted_price")
    st.line_chart(pred_plot)
