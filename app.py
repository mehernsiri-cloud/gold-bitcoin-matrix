# app.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide")

# ------------------------------
# DATA FILES
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# LOAD DATA
# ------------------------------
def load_csv_safe(path, default_cols):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=True)
    else:
        df = pd.DataFrame(columns=default_cols)
    return df

df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual"])

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_actual_pred(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    if actual_col in df_actual.columns:
        asset_pred = pd.merge_asof(
            asset_pred.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        asset_pred["actual"] = None

    # Ensure numeric
    asset_pred["predicted_price"] = pd.to_numeric(asset_pred["predicted_price"], errors='coerce')
    asset_pred["actual"] = pd.to_numeric(asset_pred["actual"], errors='coerce')

    # Buy/Sell Signal
    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if row["predicted_price"] > row["actual"] else ("Sell" if row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    # Trend: compare last 3 predicted prices
    asset_pred["trend"] = ""
    if len(asset_pred) >= 3:
        last3 = asset_pred["predicted_price"].tail(3)
        if last3.is_monotonic_increasing:
            asset_pred["trend"] = "Bullish 📈"
        elif last3.is_monotonic_decreasing:
            asset_pred["trend"] = "Bearish 📉"
        else:
            asset_pred["trend"] = "Neutral ⚖️"

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# LAYOUT
# ------------------------------
st.title("📊 Gold & Bitcoin Market Dashboard")

col1, col2 = st.columns(2)

# Function to color Buy/Sell signals in the table
def color_signal(val):
    if val == "Buy":
        color = "green"
    elif val == "Sell":
        color = "red"
    else:
        color = "gray"
    return f'color: {color}; font-weight:bold; text-align:center'

# Function to create a visual alert badge
def alert_badge(signal):
    if signal == "Buy":
        return f'<div style="background-color:green;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">BUY</div>'
    elif signal == "Sell":
        return f'<div style="background-color:red;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">SELL</div>'
    else:
        return f'<div style="background-color:gray;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">HOLD</div>'

# ------------------------------
# GOLD SECTION
# ------------------------------
with col1:
    st.subheader("Gold")
    if not gold_df.empty:
        last_signal = gold_df["signal"].iloc[-1]
        st.markdown(alert_badge(last_signal), unsafe_allow_html=True)

        last_trend = gold_df["trend"].iloc[-1] if gold_df["trend"].iloc[-1] != "" else "Neutral ⚖️"
        st.markdown(f"**Market Trend:** {last_trend}")

        # Show only last 2 rows
        display_gold = gold_df[["timestamp", "actual", "predicted_price", "volatility", "risk", "signal"]].tail(2)
        st.dataframe(display_gold.style.applymap(color_signal, subset=["signal"]))

        # Chart
        fig_gold = px.line(
            gold_df,
            x="timestamp",
            y=["actual", "predicted_price"],
            labels={"value": "Gold Price", "timestamp": "Timestamp"},
            title="Gold: Actual vs Predicted"
        )
        st.plotly_chart(fig_gold, use_container_width=True)
    else:
        st.info("No Gold data available yet.")

# ------------------------------
# BITCOIN SECTION
# ------------------------------
with col2:
    st.subheader("Bitcoin")
    if not btc_df.empty:
        last_signal = btc_df["signal"].iloc[-1]
        st.markdown(alert_badge(last_signal), unsafe_allow_html=True)

        last_trend = btc_df["trend"].iloc[-1] if btc_df["trend"].iloc[-1] != "" else "Neutral ⚖️"
        st.markdown(f"**Market Trend:** {last_trend}")

        # Show only last 2 rows
        display_btc = btc_df[["timestamp", "actual", "predicted_price", "volatility", "risk", "signal"]].tail(2)
        st.dataframe(display_btc.style.applymap(color_signal, subset=["signal"]))

        # Chart
        fig_btc = px.line(
            btc_df,
            x="timestamp",
            y=["actual", "predicted_price"],
            labels={"value": "Bitcoin Price", "timestamp": "Timestamp"},
            title="Bitcoin: Actual vs Predicted"
        )
        st.plotly_chart(fig_btc, use_container_width=True)
    else:
        st.info("No Bitcoin data available yet.")
