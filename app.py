import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Predictions vs Actuals Dashboard", layout="wide")
st.title("📊 Predictions vs Actuals Dashboard")

# --- Paths ---
DATA_DIR = "data"
PRED_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# --- Load CSVs ---
def load_csv(file):
    if os.path.exists(file):
        return pd.read_csv(file)
    return pd.DataFrame()

pred_df = load_csv(PRED_FILE)
actual_df = load_csv(ACTUAL_FILE)

# --- Latest Actuals ---
if not actual_df.empty:
    latest_actual = actual_df.iloc[-1]
    st.subheader("Latest Market Actuals")
    st.table(latest_actual)
else:
    st.warning("No actual data available yet. Run fetch_data.py or wait for workflow.")

# --- Latest Predictions vs Actuals ---
if not pred_df.empty and not actual_df.empty:
    latest_preds = pred_df.sort_values("timestamp").groupby("asset").tail(1)

    table_rows = []
    asset_map = {
        "Gold": "gold_actual",
        "Bitcoin": "bitcoin_actual",
        "Real_Estate_France": "france_2bed_actual",
        "Real_Estate_Dubai": "dubai_2bed_actual"
    }

    for _, row in latest_preds.iterrows():
        asset_name = row["asset"]
        actual_val = latest_actual.get(asset_map.get(asset_name, ""), "N/A")
        table_rows.append({
            "Asset": asset_name,
            "Predicted Price": row["predicted_price"],
            "Actual Price": actual_val,
            "Volatility": row["volatility"],
            "Risk": row["risk"]
        })

    st.subheader("Latest Predictions vs Actuals")
    st.dataframe(pd.DataFrame(table_rows))

# --- Historical Charts ---
if not actual_df.empty:
    st.subheader("📈 Historical Actual Prices")
    chart_cols = [col for col in actual_df.columns if col != "date"]
    st.line_chart(actual_df.set_index("date")[chart_cols])

if not pred_df.empty:
    st.subheader("📉 Historical Predictions")
    chart_cols = ["predicted_price"]
    st.line_chart(pred_df.pivot(index="timestamp", columns="asset", values="predicted_price"))
