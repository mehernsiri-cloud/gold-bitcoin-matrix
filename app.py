import pandas as pd
import streamlit as st
from datetime import datetime

# Load actuals
actual_data = pd.read_csv("data/actual_data.csv")
latest_actual = actual_data.iloc[-1] if not actual_data.empty else None

# Load predictions
predictions = pd.read_csv("data/predictions_log.csv")

# Merge latest actuals with predictions
if latest_actual is not None:
    today_actuals = {
        "Gold": latest_actual["gold_actual"],
        "Bitcoin": latest_actual["bitcoin_actual"],
        "France Studio": latest_actual["france_studio_actual"],
        "France 2-Bed": latest_actual["france_2bed_actual"],
        "France 3-Bed": latest_actual["france_3bed_actual"],
        "Dubai Studio": latest_actual["dubai_studio_actual"],
        "Dubai 2-Bed": latest_actual["dubai_2bed_actual"],
        "Dubai 3-Bed": latest_actual["dubai_3bed_actual"],
    }

    latest_preds = predictions[predictions["timestamp"] == predictions["timestamp"].max()]

    # Build table with actuals included
    table_data = []
    for _, row in latest_preds.iterrows():
        asset_name = row["asset"]
        table_data.append({
            "Asset": asset_name,
            "Predicted Price": row["predicted_price"],
            "Actual Price": today_actuals.get(asset_name.replace("_", " "), None),
            "Volatility": row["volatility"],
            "Risk": row["risk"],
        })

    st.subheader("📊 Latest Predictions with Actuals")
    st.dataframe(pd.DataFrame(table_data))
