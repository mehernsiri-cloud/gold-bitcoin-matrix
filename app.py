# app.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Predictions", layout="wide")

# ------------------------------
# DATA FILES
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# LOAD PREDICTIONS
# ------------------------------
if os.path.exists(PREDICTION_FILE):
    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
else:
    df_pred = pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "volatility", "risk"])

# ------------------------------
# LOAD ACTUALS
# ------------------------------
if os.path.exists(ACTUAL_FILE):
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])
else:
    df_actual = pd.DataFrame(columns=["timestamp", "gold_actual", "bitcoin_actual"])

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_data(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if not df_actual.empty and actual_col in df_actual.columns:
        # Align by timestamp if possible, else fill NaN
        asset_pred["actual"] = df_actual[actual_col].reindex(asset_pred.index, fill_value=None)
    else:
        asset_pred["actual"] = None
    return asset_pred

gold_df = merge_data("Gold", "gold_actual")
btc_df = merge_data("Bitcoin", "bitcoin_actual")

# ------------------------------
# LAYOUT
# ------------------------------
st.title("Gold & Bitcoin Predictions Dashboard")

col1, col2 = st.columns(2)

# Gold Section
with col1:
    st.subheader("Gold")
    if not gold_df.empty:
        st.dataframe(gold_df[["timestamp","predicted_price","actual","volatility","risk"]])
        fig_gold = px.line(
            gold_df,
            x="timestamp",
            y=["predicted_price","actual"],
            labels={"value": "Price", "timestamp": "Timestamp"},
            title="Gold: Predicted vs Actual"
        )
        st.plotly_chart(fig_gold, use_container_width=True)
    else:
        st.info("No Gold data available yet.")

# Bitcoin Section
with col2:
    st.subheader("Bitcoin")
    if not btc_df.empty:
        st.dataframe(btc_df[["timestamp","predicted_price","actual","volatility","risk"]])
        fig_btc = px.line(
            btc_df,
            x="timestamp",
            y=["predicted_price","actual"],
            labels={"value": "Price", "timestamp": "Timestamp"},
            title="Bitcoin: Predicted vs Actual"
        )
        st.plotly_chart(fig_btc, use_container_width=True)
    else:
        st.info("No Bitcoin data available yet.")

# ------------------------------
# END
# ------------------------------
