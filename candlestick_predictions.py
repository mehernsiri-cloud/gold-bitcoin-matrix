# candlestick_predictions.py
# Candlestick logic and UI rendering
# Updated for: last month classical patterns + last week short-term patterns
# Author: ChatGPT
# Date: 2025-10-03
# ------------------------------------------------------------------------------

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple, Dict

DATA_DIR = "data"
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")

# -------------------------------------------------------------------
# SHORT-TERM 3-CANDLE PATTERNS (Last week)
# -------------------------------------------------------------------
def detect_patterns_in_3_candles(window: pd.DataFrame) -> List[str]:
    patterns: List[str] = []
    if window.shape[0] != 3:
        return patterns

    c0, c1 = window.iloc[2], window.iloc[1]
    open0, high0, low0, close0 = map(float, [c0.open, c0.high, c0.low, c0.close])
    body0 = abs(close0 - open0)
    rng0 = max(high0 - low0, 1e-9)
    upper_wick = high0 - max(open0, close0)
    lower_wick = min(open0, close0) - low0

    # Doji
    if body0 < 0.15 * rng0:
        patterns.append("Doji (indecision)")

    # Hammer / Shooting Star
    if body0 < rng0 * 0.3:
        if lower_wick > 2 * body0:
            patterns.append("Hammer (bullish)")
        if upper_wick > 2 * body0:
            patterns.append("Shooting Star (bearish)")

    # Engulfing
    if close0 > open0 and c1.close < c1.open and close0 > c1.open and open0 < c1.close:
        patterns.append("Bullish Engulfing")
    if close0 < open0 and c1.close > c1.open and close0 < c1.open and open0 > c1.close:
        patterns.append("Bearish Engulfing")

    return patterns

def detect_candle_patterns_on_series(df_ohlc: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    results = []
    if df_ohlc.shape[0] < 3:
        return results
    df_sorted = df_ohlc.sort_values("timestamp").reset_index(drop=True)
    for i in range(2, len(df_sorted)):
        window = df_sorted.iloc[i-2:i+1].copy()
        ts = window.iloc[2].timestamp
        patterns = detect_patterns_in_3_candles(window)
        for p in patterns:
            results.append((ts, p))
    return results

# -------------------------------------------------------------------
# CLASSICAL MULTI-CANDLE PATTERNS (Last month)
# -------------------------------------------------------------------
# (Copy all previous functions: detect_head_shoulders, detect_double_triple_top_bottom, detect_triangle_patterns, etc.)

def detect_classical_patterns(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    results = []
    results += detect_head_shoulders(df)
    results += detect_double_triple_top_bottom(df)
    results += detect_triangle_patterns(df)
    results += detect_cup_handle(df)
    results += detect_flags_pennants(df)
    results += detect_rounding_patterns(df)
    results += detect_wedges(df)
    return results

# -------------------------------------------------------------------
# WEEKLY AGGREGATION
# -------------------------------------------------------------------
def aggregate_weekly_patterns(patterns: List[Tuple[pd.Timestamp, str]]) -> Dict[str, int]:
    agg = {}
    for _, pat in patterns:
        agg[pat] = agg.get(pat, 0) + 1
    return agg

def decide_weekly_signal(weekly_patterns: Dict[str, int]) -> str:
    bull_patterns = ["Bullish", "Hammer", "Double Bottom", "Triple Bottom",
                     "Cup & Handle", "Rounding Bottom", "Falling Wedge", "Ascending Triangle"]
    bear_patterns = ["Bearish", "Shooting Star", "Double Top", "Triple Top",
                     "Head & Shoulders", "Rounding Top", "Rising Wedge", "Descending Triangle"]
    bull = sum(v for k, v in weekly_patterns.items() if any(bp in k for bp in bull_patterns))
    bear = sum(v for k, v in weekly_patterns.items() if any(bp in k for bp in bear_patterns))
    if bull > bear:
        return "Bullish 📈"
    elif bear > bull:
        return "Bearish 📉"
    else:
        return "Neutral ⚖️"

# -------------------------------------------------------------------
# SYNTHETIC NEXT-WEEK CANDLES
# -------------------------------------------------------------------
def synthesize_predicted_candles(last_week: pd.DataFrame, signal: str) -> pd.DataFrame:
    if last_week.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    last_close = last_week.iloc[-1].close
    candles = []
    base_date = last_week.iloc[-1].timestamp
    drift = 0.01 if "Bullish" in signal else -0.01 if "Bearish" in signal else 0.0

    for i in range(1, 6):
        date = base_date + timedelta(days=i)
        open_p = last_close * (1 + drift * 0.2)
        close_p = open_p * (1 + drift * 0.5)
        high_p = max(open_p, close_p) * (1 + 0.005)
        low_p = min(open_p, close_p) * (1 - 0.005)
        candles.append((date, open_p, high_p, low_p, close_p))
        last_close = close_p
    return pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close"])

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
def log_weekly_candlestick_predictions(df_pred: pd.DataFrame):
    if df_pred.empty:
        return
    rows = []
    for _, r in df_pred.iterrows():
        rows.append({
            "timestamp": r.timestamp,
            "asset": "Bitcoin",
            "predicted_price": r.close,
            "method": "Candlestick Synthetic"
        })
    df_new = pd.DataFrame(rows)
    if os.path.exists(AI_PRED_FILE):
        df_old = pd.read_csv(AI_PRED_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(AI_PRED_FILE, index=False)

# -------------------------------------------------------------------
# UI RENDERING
# -------------------------------------------------------------------
def render_candlestick_dashboard(df_actual: pd.DataFrame):
    st.title("🕯️ Candlestick Predictions")

    if df_actual.empty:
        st.error("No actual Bitcoin OHLC data available.")
        return

    required_cols = ["bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close", "timestamp"]
    missing = [c for c in required_cols if c not in df_actual.columns]
    if missing:
        st.error(f"Missing columns in actual data: {missing}")
        return

    df_ohlc = df_actual.rename(columns={
        "bitcoin_open": "open",
        "bitcoin_high": "high",
        "bitcoin_low": "low",
        "bitcoin_close": "close"
    }).dropna(subset=["open", "high", "low", "close"])

    # Ensure timestamp is datetime
    df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"])

    # --- Short-term patterns: last week only
    one_week_ago = df_ohlc["timestamp"].max() - timedelta(days=7)
    df_last_week = df_ohlc[df_ohlc["timestamp"] >= one_week_ago]
    short_patterns = detect_candle_patterns_on_series(df_last_week)

    # --- Classical patterns: last month
    one_month_ago = df_ohlc["timestamp"].max() - timedelta(days=30)
    df_last_month = df_ohlc[df_ohlc["timestamp"] >= one_month_ago]
    classical_patterns = detect_classical_patterns(df_last_month)

    all_patterns = short_patterns + classical_patterns
    weekly_patterns = aggregate_weekly_patterns(all_patterns)
    signal = decide_weekly_signal(weekly_patterns)
    st.subheader(f"Weekly Signal: {signal}")

    # --- Last week prediction
    df_predicted = synthesize_predicted_candles(df_last_week.tail(5), signal)
    if not df_predicted.empty:
        log_weekly_candlestick_predictions(df_predicted)

    # --- Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_ohlc["timestamp"], open=df_ohlc["open"], high=df_ohlc["high"],
        low=df_ohlc["low"], close=df_ohlc["close"], name="Actual"
    ))
    if not df_predicted.empty:
        fig.add_trace(go.Candlestick(
            x=df_predicted["timestamp"], open=df_predicted["open"], high=df_predicted["high"],
            low=df_predicted["low"], close=df_predicted["close"], name="Predicted",
            increasing_line_color="blue", decreasing_line_color="red"
        ))
    fig.update_layout(title="Bitcoin Candlestick Predictions", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Weekly Pattern Counts")
    st.json(weekly_patterns)
