# candlestick_predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ------------------------------
# Pattern Detection
# ------------------------------
def detect_candle_patterns(df_ohlc):
    """Detect basic candlestick patterns from last 7 days of data."""
    patterns = []
    if df_ohlc is None or df_ohlc.shape[0] < 3:
        return patterns
    recent = df_ohlc.tail(7).reset_index(drop=True)
    for i in range(1, len(recent)):
        c0 = recent.iloc[i]
        c1 = recent.iloc[i-1]
        body0 = abs(c0["close"] - c0["open"])
        range0 = c0["high"] - c0["low"] if c0["high"] - c0["low"] > 0 else 1
        upper_wick = c0["high"] - max(c0["open"], c0["close"])
        lower_wick = min(c0["open"], c0["close"]) - c0["low"]

        if body0 < 0.15 * range0:
            patterns.append("Doji")
        if lower_wick > 2 * body0 and upper_wick < body0 and c0["close"] > c0["open"]:
            patterns.append("Hammer (Bullish)")
        if upper_wick > 2 * body0 and lower_wick < body0 and c0["close"] < c0["open"]:
            patterns.append("Shooting Star (Bearish)")
        if (c1["close"] < c1["open"]) and (c0["close"] > c0["open"]) and (c0["close"] > c1["open"]) and (c0["open"] < c1["close"]):
            patterns.append("Bullish Engulfing")
        if (c1["close"] > c1["open"]) and (c0["close"] < c0["open"]) and (c0["open"] > c1["close"]) and (c0["close"] < c1["open"]):
            patterns.append("Bearish Engulfing")
    return patterns

def pattern_to_signal(patterns):
    """Convert detected patterns to overall weekly signal."""
    if not patterns:
        return "Neutral"
    bulls = sum(1 for p in patterns if "Bullish" in p or "Hammer" in p)
    bears = sum(1 for p in patterns if "Bearish" in p or "Shooting" in p)
    if bulls > bears:
        return "Bullish"
    if bears > bulls:
        return "Bearish"
    return "Neutral"

# ------------------------------
# Synthetic Prediction Generator
# ------------------------------
def generate_next_week_predictions(last_close, signal, start_date):
    """Generate synthetic OHLC candles for the next 7 days based on signal."""
    preds = []
    for i in range(1, 8):
        date = start_date + timedelta(days=i)
        if signal == "Bullish":
            close = last_close * (1 + 0.01 * i)
        elif signal == "Bearish":
            close = last_close * (1 - 0.01 * i)
        else:
            close = last_close
        open_ = last_close
        high = max(open_, close) * 1.005
        low = min(open_, close) * 0.995
        preds.append({"timestamp": date, "open": open_, "high": high, "low": low, "close": close})
        last_close = close
    return pd.DataFrame(preds)

# ------------------------------
# Chart Plotting
# ------------------------------
def plot_candlestick_with_prediction(df_actual, df_pred):
    """Plot last 7 days actual OHLC + next 7 days predicted OHLC."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_actual["timestamp"], open=df_actual["open"], high=df_actual["high"],
        low=df_actual["low"], close=df_actual["close"],
        name="Actual (Last Week)", increasing_line_color="green", decreasing_line_color="red"
    ))
    fig.add_trace(go.Candlestick(
        x=df_pred["timestamp"], open=df_pred["open"], high=df_pred["high"],
        low=df_pred["low"], close=df_pred["close"],
        name="Predicted (Next Week)", increasing_line_color="blue", decreasing_line_color="orange"
    ))
    fig.update_layout(
        title="Bitcoin Candlestick Prediction (Last 7 days vs Next 7 days)",
        xaxis_title="Date", yaxis_title="Price", template="plotly_white"
    )
    return fig

# ------------------------------
# Logging Predictions
# ------------------------------
def log_predictions(df_pred, ai_pred_file):
    """Append candlestick predictions into ai_predictions_log.csv"""
    df_to_log = df_pred[["timestamp", "close"]].copy()
    df_to_log["asset"] = "Bitcoin-Candle"
    df_to_log.rename(columns={"close": "predicted_price"}, inplace=True)
    try:
        existing = pd.read_csv(ai_pred_file, parse_dates=["timestamp"])
    except Exception:
        existing = pd.DataFrame(columns=["timestamp", "asset", "predicted_price"])
    df_final = pd.concat([existing, df_to_log], ignore_index=True).drop_duplicates(subset=["timestamp", "asset"], keep="last")
    df_final.to_csv(ai_pred_file, index=False)

# ------------------------------
# Dashboard
# ------------------------------
def candlestick_dashboard(df_actual, ai_pred_file):
    st.header("📊 Candlestick Predictions – Bitcoin")
    st.write("This module analyzes last 7 days of Bitcoin OHLC data, detects candlestick patterns, and projects the next 7 days with synthetic candlesticks.")

    if df_actual.empty:
        st.warning("No actual data available.")
        return

    # Prepare OHLC for last 7 days
    df_ohlc = df_actual[["timestamp", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]].copy()
    df_ohlc.rename(columns={"bitcoin_open": "open", "bitcoin_high": "high", "bitcoin_low": "low", "bitcoin_close": "close"}, inplace=True)
    df_last_week = df_ohlc.tail(7)

    if df_last_week.empty:
        st.warning("Not enough OHLC data for candlestick predictions.")
        return

    # Detect patterns & signal
    patterns = detect_candle_patterns(df_last_week)
    signal = pattern_to_signal(patterns)
    st.subheader("Detected Patterns (Last Week)")
    st.write(", ".join(patterns) if patterns else "No strong patterns found")
    st.info(f"Overall Weekly Signal: **{signal}**")

    # Generate next week predictions
    last_close = df_last_week["close"].iloc[-1]
    start_date = df_last_week["timestamp"].iloc[-1]
    df_pred = generate_next_week_predictions(last_close, signal, start_date)

    # Plot
    fig = plot_candlestick_with_prediction(df_last_week, df_pred)
    st.plotly_chart(fig, use_container_width=True)

    # Log predictions
    log_predictions(df_pred, ai_pred_file)
    st.success("Predictions logged successfully into ai_predictions_log.csv")
