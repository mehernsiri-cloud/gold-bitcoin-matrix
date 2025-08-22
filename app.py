import pandas as pd
import os
import streamlit as st
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_FILE = os.path.join(BASE_DIR, "data", "predictions_log.csv")
ACTUALS_FILE = os.path.join(BASE_DIR, "data", "actual_data.csv")

def load_latest_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        return pd.DataFrame()

    preds = pd.read_csv(PREDICTIONS_FILE)
    if preds.empty:
        return pd.DataFrame()

    # Get most recent timestamp per asset
    latest_time = preds["timestamp"].max()
    latest_preds = preds[preds["timestamp"] == latest_time]

    return latest_preds

def load_actuals():
    if not os.path.exists(ACTUALS_FILE):
        return pd.DataFrame()

    actuals = pd.read_csv(ACTUALS_FILE)
    if actuals.empty:
        return pd.DataFrame()

    # Take today’s row (latest date)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    todays_actuals = actuals[actuals["date"] == today]

    return todays_actuals

def build_predictions_table():
    preds = load_latest_predictions()
    actuals = load_actuals()

    if preds.empty or actuals.empty:
        return pd.DataFrame()

    # Map actuals into tidy format
    actuals_tidy = pd.melt(
        actuals,
        id_vars=["date"],
        var_name="asset",
        value_name="actual_price"
    )

    # Normalize asset names to match predictions
    actuals_tidy["asset"] = actuals_tidy["asset"].str.replace("_actual", "", regex=False)
    actuals_tidy["asset"] = actuals_tidy["asset"].str.replace("_", " ").str.title()

    preds["asset_norm"] = preds["asset"].str.replace("_", " ").str.title()

    # Merge predictions with actuals
    merged = preds.merge(
        actuals_tidy,
        left_on="asset_norm",
        right_on="asset",
        how="left"
    )

    # Final clean-up
    merged = merged[["asset_x", "predicted_price", "actual_price", "volatility", "risk"]]
    merged.columns = ["Asset", "Predicted Price", "Actual Price", "Volatility", "Risk"]

    return merged

# Streamlit usage
st.subheader("📊 Latest Predictions with Actuals")
table = build_predictions_table()
if table.empty:
    st.info("No prediction data available yet.")
else:
    st.dataframe(table, use_container_width=True)
