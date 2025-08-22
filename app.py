import streamlit as st
import pandas as pd
import os
import yaml
from fetch_data import save_actual_data

st.set_page_config(layout="wide", page_title="Gold / Bitcoin / Real Estate Predictions")
st.title("📊 Predictions Dashboard")

# Paths
PRED_CSV="data/predictions_log.csv"
ACTUAL_CSV="data/actual_data.csv"
WEIGHT_FILE="weight.yaml"

# Load CSVs
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"] if "predictions" in path else ["date"])
    return pd.DataFrame()

predictions=load_csv(PRED_CSV)
actuals=load_csv(ACTUAL_CSV)

# Load weights
with open(WEIGHT_FILE,"r") as f:
    weights=yaml.safe_load(f)

sections=["Gold","Bitcoin","Real_Estate_France","Real_Estate_Dubai"]

def merge_data(pred_df, actual_df, asset):
    pred_asset=pred_df[pred_df["asset"]==asset].copy()
    col_name=asset.lower()+"_actual"
    actual_subset=actual_df[["date",col_name]].rename(columns={col_name:"actual_price"})
    return pd.merge(pred_asset,actual_subset,left_on="timestamp",right_on="date",how="left")

for asset in sections:
    st.header(asset)
    merged=merge_data(predictions,actuals,asset)
    if merged.empty:
        st.info(f"No data for {asset}")
        continue
    st.subheader("Latest predictions")
    latest_date=merged["timestamp"].max()
    st.dataframe(merged.tail(5)[["timestamp","predicted_price","actual_price","volatility","risk"]])
    st.subheader("Price Trend")
    st.line_chart(merged.set_index("timestamp")[["predicted_price","actual_price"]])
    if asset in ["Gold","Bitcoin"]:
        st.subheader("Weighted Prediction Factors")
        st.write(weights[asset.lower()])

st.sidebar.header("Data Refresh")
if st.sidebar.button("Fetch Latest Actuals"):
    save_actual_data()
    st.sidebar.success("Actuals updated!")
