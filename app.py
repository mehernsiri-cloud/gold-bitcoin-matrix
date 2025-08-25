# app.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import yaml

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
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# LOAD DATA SAFELY
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
# LOAD WEIGHTS / ASSUMPTIONS
# ------------------------------
if os.path.exists(WEIGHT_FILE):
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
else:
    weights = {"gold": {}, "bitcoin": {}}

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_actual_pred(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"].str.lower() == asset_name.lower()].copy()
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

    # Numeric conversion
    asset_pred["predicted_price"] = pd.to_numeric(asset_pred["predicted_price"], errors='coerce')
    asset_pred["actual"] = pd.to_numeric(asset_pred["actual"], errors='coerce')

    # Buy/Sell Signal
    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if row["predicted_price"] > row["actual"] else ("Sell" if row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    # Trend: last 3 predicted prices
    asset_pred["trend"] = ""
    if len(asset_pred) >= 3:
        last3 = asset_pred["predicted_price"].tail(3)
        if last3.is_monotonic_increasing:
            asset_pred["trend"] = "Bullish 📈"
        elif last3.is_monotonic_decreasing:
            asset_pred["trend"] = "Bearish 📉"
        else:
            asset_pred["trend"] = "Neutral ⚖️"

    # Add assumptions and target horizon
    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))
    asset_pred["target_horizon"] = "Days"

    # Target buy/sell price: realistic strategy
    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"]=="Buy" else (row["predicted_price"] if row["signal"]=="Sell" else row["actual"]),
        axis=1
    )

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# LAYOUT
# ------------------------------
st.title("📊 Gold & Bitcoin Market Dashboard")
col1, col2 = st.columns(2)

# ------------------------------
# FUNCTIONS
# ------------------------------
def color_signal(val):
    if val == "Buy":
        color = "#1f77b4"  # blue
    elif val == "Sell":
        color = "#ff7f0e"  # orange
    else:
        color = "gray"
    return f'color: {color}; font-weight:bold; text-align:center'

def alert_badge(signal):
    if signal == "Buy":
        return f'<div style="background-color:#1f77b4;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">BUY</div>'
    elif signal == "Sell":
        return f'<div style="background-color:#ff7f0e;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">SELL</div>'
    else:
        return f'<div style="background-color:gray;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">HOLD</div>'

def target_price_card(price, asset_name):
    st.markdown(f"""
        <div style='background-color:#ffd700;color:black;padding:12px;font-size:22px;text-align:center;border-radius:8px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price}
        </div>
        """, unsafe_allow_html=True)

def assumptions_card(asset_df, asset_name, fig=None):
    if asset_df.empty:
        st.info(f"No assumptions available for {asset_name}")
        return
    assumptions_str = asset_df["assumptions"].iloc[-1]
    try:
        assumptions = eval(assumptions_str) if assumptions_str else {}
    except:
        assumptions = {}

    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return

    indicators = list(assumptions.keys())
    values = [assumptions[k] for k in indicators]
    colors = ["#1f77b4" if v>0 else "#ff7f0e" if v<0 else "gray" for v in values]

    # Bar chart for indicator weights
    fig_bar = go.Figure([go.Bar(
        x=indicators,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition='auto'
    )])
    fig_bar.update_layout(title=f"{asset_name} Assumptions & Target (Days)", yaxis_title="Weight / Impact")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Add emojis on the price chart if provided
    if fig is not None:
        for i, ind in enumerate(indicators):
            fig.add_annotation(
                x=asset_df["timestamp"].iloc[-1],
                y=asset_df["predicted_price"].iloc[-1],
                text="📈" if values[i]>0 else "📉" if values[i]<0 else "⚖️",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20*(i+1),
                font=dict(size=16)
            )

# ------------------------------
# GOLD SECTION
# ------------------------------
with col1:
    st.subheader("Gold")
    if not gold_df.empty:
        last_signal = gold_df["signal"].iloc[-1]
        st.markdown(alert_badge(last_signal), unsafe_allow_html=True)
        last_trend = gold_df["trend"].iloc[-1] if gold_df["trend"].iloc[-1] else "Neutral ⚖️"
        st.markdown(f"**Market Trend:** {last_trend}")

        target_price_card(gold_df["target_price"].iloc[-1], "Gold")

        display_gold = gold_df[["timestamp","actual","predicted_price","volatility","risk","signal"]].tail(2)
        st.dataframe(display_gold.style.applymap(color_signal, subset=["signal"]))

        # Gold chart
        fig_gold = px.line(
            gold_df,
            x="timestamp",
            y=["actual","predicted_price"],
            labels={"value":"Gold Price","timestamp":"Timestamp"},
            title="Gold: Actual vs Predicted"
        )
        assumptions_card(gold_df, "Gold", fig=fig_gold)
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
        last_trend = btc_df["trend"].iloc[-1] if btc_df["trend"].iloc[-1] else "Neutral ⚖️"
        st.markdown(f"**Market Trend:** {last_trend}")

        target_price_card(btc_df["target_price"].iloc[-1], "Bitcoin")

        display_btc = btc_df[["timestamp","actual","predicted_price","volatility","risk","signal"]].tail(2)
        st.dataframe(display_btc.style.applymap(color_signal, subset=["signal"]))

        # Bitcoin chart
        fig_btc = px.line(
            btc_df,
            x="timestamp",
            y=["actual","predicted_price"],
            labels={"value":"Bitcoin Price","timestamp":"Timestamp"},
            title="Bitcoin: Actual vs Predicted"
        )
        assumptions_card(btc_df, "Bitcoin", fig=fig_btc)
        st.plotly_chart(fig_btc, use_container_width=True)
    else:
        st.info("No Bitcoin data available yet.")
