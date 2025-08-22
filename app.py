import streamlit as st
import pandas as pd
import os
from fetch_data import save_actual_data, DATA_DIR

PRED_CSV = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_CSV = os.path.join(DATA_DIR, "actual_data.csv")

st.set_page_config(page_title="Gold & Bitcoin Predictions", layout="wide")
st.title("Gold & Bitcoin Dashboard")

# ------------------------------
# Load data
# ------------------------------
save_actual_data()  # refresh actual prices

def load_csv(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return pd.read_csv(file_path)
    return pd.DataFrame()

pred_df = load_csv(PRED_CSV)
actual_df = load_csv(ACTUAL_CSV)

# ------------------------------
# Latest Predictions Table
# ------------------------------
if not pred_df.empty and not actual_df.empty:
    latest_pred_time = pred_df["timestamp"].max()
    latest_pred = pred_df[pred_df["timestamp"] == latest_pred_time]

    latest_actual = actual_df.sort_values("date").iloc[-1]

    # Build table with actuals
    table_data = []
    for idx, row in latest_pred.iterrows():
        asset = row['asset']
        actual = latest_actual.get(f"{asset.lower()}_actual", None)
        signal = "Hold"
        if actual and row["predicted_price"]:
            if row["predicted_price"] > actual*1.01:
                signal = "Buy"
            elif row["predicted_price"] < actual*0.99:
                signal = "Sell"
        table_data.append({
            "Asset": asset,
            "Predicted Price": row["predicted_price"],
            "Actual Price": actual,
            "Volatility": row["volatility"],
            "Risk": row["risk"],
            "Signal": signal
        })

    st.subheader("Latest Predictions vs Actuals")
    st.dataframe(pd.DataFrame(table_data))
else:
    st.info("No prediction or actual data available yet.")

# ------------------------------
# Charts
# ------------------------------
if not actual_df.empty:
    st.subheader("📊 Actual Prices Over Time")
    actual_plot = actual_df.set_index("date")[["gold_actual","bitcoin_actual"]]
    st.line_chart(actual_plot)

if not pred_df.empty:
    st.subheader("📈 Predicted Prices Over Time")
    pred_plot = pred_df.pivot(index="timestamp", columns="asset", values="predicted_price")
    st.line_chart(pred_plot)
