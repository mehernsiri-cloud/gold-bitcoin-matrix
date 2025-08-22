# app.py
import streamlit as st
import pandas as pd
import os
from urllib.error import URLError

# ------------------------------
# Config
# ------------------------------
LOCAL_CSV = "predictions_log.csv"
GITHUB_CSV_URL = ""  # Optional: add your raw GitHub URL here if you want to load remotely

# ------------------------------
# Load predictions function
# ------------------------------
@st.cache_data
def load_predictions():
    """
    Load predictions CSV either locally or from GitHub URL.
    Returns a DataFrame with 'timestamp', 'asset', 'predicted_price', 'volatility', 'risk'
    """
    # Try local CSV first
    if os.path.exists(LOCAL_CSV):
        try:
            df = pd.read_csv(LOCAL_CSV, parse_dates=["timestamp"])
            return df
        except Exception as e:
            st.warning(f"Error reading local CSV: {e}")

    # Fallback: try GitHub URL
    if GITHUB_CSV_URL:
        try:
            df = pd.read_csv(GITHUB_CSV_URL, parse_dates=["timestamp"])
            return df
        except URLError as e:
            st.error(f"Cannot fetch CSV from URL: {e}")
        except Exception as e:
            st.error(f"Error reading CSV from URL: {e}")

    # If all fails, return empty DataFrame
    st.warning("No predictions CSV available. Showing empty table.")
    columns = ["timestamp", "asset", "predicted_price", "volatility", "risk"]
    return pd.DataFrame(columns=columns)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Gold/Bitcoin/Real Estate Predictions", layout="wide")
st.title("Predictions Dashboard")

df = load_predictions()

if df.empty:
    st.info("No prediction data available yet.")
else:
    # Show latest predictions only
    latest_timestamp = df["timestamp"].max()
    st.subheader(f"Latest predictions: {latest_timestamp}")
    latest_df = df[df["timestamp"] == latest_timestamp]
    st.dataframe(latest_df)

    # Optional: chart over time
    st.subheader("Prediction Trends Over Time")
    for asset in df["asset"].unique():
        asset_df = df[df["asset"] == asset]
        st.line_chart(asset_df.set_index("timestamp")["predicted_price"], use_container_width=True)
