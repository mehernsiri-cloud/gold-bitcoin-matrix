import streamlit as st
import pandas as pd
import os

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# Ensure CSV files exist
# ------------------------------
os.makedirs(DATA_DIR, exist_ok=True)

for f, columns in [(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "actual_price"]),
                   (ACTUAL_FILE, ["timestamp", "date", "gold_actual", "bitcoin_actual",
                                  "inflation", "real_rates","usd_strength","liquidity",
                                  "equity_flows","bond_yields","regulation","adoption",
                                  "currency_instability","recession_probability",
                                  "tail_risk_event","geopolitics","energy_prices"])]:
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=columns).to_csv(f, index=False)

# ------------------------------
# Load CSVs
# ------------------------------
@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTION_FILE)

@st.cache_data
def load_actuals():
    return pd.read_csv(ACTUAL_FILE)

df = load_predictions()
act_df = load_actuals()

# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Predictions", layout="wide")
st.title("Gold & Bitcoin Predictions Dashboard")

if df.empty:
    st.info("No prediction data yet. Run fetch_data.py and predict_headless.py first.")
else:
    # Separate Gold and Bitcoin
    gold_df = df[df["asset"] == "Gold"].copy()
    bitcoin_df = df[df["asset"] == "Bitcoin"].copy()

    # Merge actual prices safely
    if not act_df.empty:
        gold_df["actual_price"] = act_df["gold_actual"].iloc[-len(gold_df):].values
        bitcoin_df["actual_price"] = act_df["bitcoin_actual"].iloc[-len(bitcoin_df):].values

    # Layout columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gold")
        st.dataframe(gold_df[["timestamp", "predicted_price", "actual_price"]])
        st.line_chart(
            gold_df.set_index("timestamp")[["predicted_price", "actual_price"]],
            use_container_width=True
        )

    with col2:
        st.subheader("Bitcoin")
        st.dataframe(bitcoin_df[["timestamp", "predicted_price", "actual_price"]])
        st.line_chart(
            bitcoin_df.set_index("timestamp")[["predicted_price", "actual_price"]],
            use_container_width=True
        )
