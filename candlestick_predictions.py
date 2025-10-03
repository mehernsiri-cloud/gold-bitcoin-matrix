# candlestick_predictions.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple, Dict

DATA_DIR = "data"
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")

# -------------------------------
# SHORT-TERM 3-CANDLE PATTERNS
# -------------------------------
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
    if body0 < 0.15 * rng0:
        patterns.append("Doji (indecision)")
    if body0 < rng0 * 0.3:
        if lower_wick > 2 * body0:
            patterns.append("Hammer (bullish)")
        if upper_wick > 2 * body0:
            patterns.append("Shooting Star (bearish)")
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

# -------------------------------
# CLASSICAL MULTI-CANDLE PATTERNS
# -------------------------------
def detect_head_shoulders(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(3, len(prices)-3):
        left = prices[i-3:i]
        middle = prices[i-1:i+2]
        right = prices[i+1:i+4]
        if max(middle) > max(left) and max(middle) > max(right):
            patterns.append((ts_list[i], "Head & Shoulders (bearish)"))
    return patterns

def detect_double_triple_top_bottom(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(2, len(prices)-2):
        if prices[i-1] < prices[i] > prices[i+1] and abs(prices[i] - prices[i-2])/prices[i] < 0.02:
            patterns.append((ts_list[i], "Double Top (bearish)"))
        if prices[i-1] > prices[i] < prices[i+1] and abs(prices[i] - prices[i-2])/prices[i] < 0.02:
            patterns.append((ts_list[i], "Double Bottom (bullish)"))
        if i >= 3 and prices[i-2] < prices[i-1] > prices[i] and abs(prices[i-2]-prices[i])/prices[i] < 0.02:
            patterns.append((ts_list[i], "Triple Top (bearish)"))
        if i >= 3 and prices[i-2] > prices[i-1] < prices[i] and abs(prices[i-2]-prices[i])/prices[i] < 0.02:
            patterns.append((ts_list[i], "Triple Bottom (bullish)"))
    return patterns

def detect_triangle_patterns(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(3, len(prices)-2):
        window = prices[i-3:i+2]
        if all(x <= y for x, y in zip(window, window[1:])):
            patterns.append((ts_list[i], "Ascending Triangle (bullish)"))
        elif all(x >= y for x, y in zip(window, window[1:])):
            patterns.append((ts_list[i], "Descending Triangle (bearish)"))
        else:
            patterns.append((ts_list[i], "Symmetrical Triangle (neutral)"))
    return patterns

def detect_cup_handle(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(5, len(prices)-5):
        left = prices[i-5:i]
        right = prices[i+1:i+6]
        if min(left) < prices[i] and min(right) < prices[i]:
            patterns.append((ts_list[i], "Cup & Handle (bullish)"))
    return patterns

def detect_flags_pennants(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(3, len(prices)-2):
        window = prices[i-3:i+2]
        if max(window) - min(window) < 0.02 * prices[i]:
            patterns.append((ts_list[i], "Flag/Pennant (neutral)"))
    return patterns

def detect_rounding_patterns(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(2, len(prices)-2):
        if prices[i-1] > prices[i] < prices[i+1]:
            patterns.append((ts_list[i], "Rounding Bottom (bullish)"))
        if prices[i-1] < prices[i] > prices[i+1]:
            patterns.append((ts_list[i], "Rounding Top (bearish)"))
    return patterns

def detect_wedges(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, str]]:
    patterns = []
    prices = df["close"].values
    ts_list = df["timestamp"].values
    for i in range(3, len(prices)-2):
        window = prices[i-3:i+2]
        if all(x < y for x, y in zip(window, window[1:])):
            patterns.append((ts_list[i], "Rising Wedge (bearish)"))
        if all(x > y for x, y in zip(window, window[1:])):
            patterns.append((ts_list[i], "Falling Wedge (bullish)"))
    return patterns

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

# -------------------------------
# WEEKLY AGGREGATION
# -------------------------------
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

# -------------------------------
# SYNTHETIC NEXT-WEEK CANDLES
# -------------------------------
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

# -------------------------------
# LOGGING
# -------------------------------
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

# -------------------------------
# UI RENDERING
# -------------------------------
def render_candlestick_dashboard(df_actual: pd.DataFrame):
    st.title("🕯️ Candlestick Predictions")
    if df_actual.empty:
        st.error("No actual Bitcoin OHLC data available.")
        return
    required_cols = ["bitcoin_open","bitcoin_high","bitcoin_low","bitcoin_close","timestamp"]
    missing = [c for c in required_cols if c not in df_actual.columns]
    if missing:
        st.error(f"Missing columns in actual data: {missing}")
        return
    df_ohlc = df_actual.rename(columns={
        "bitcoin_open":"open","bitcoin_high":"high","bitcoin_low":"low","bitcoin_close":"close"
    }).dropna(subset=["open","high","low","close"])
    df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"])

    df_last_week = df_ohlc[df_ohlc["timestamp"] >= df_ohlc["timestamp"].max()-timedelta(days=7)]
    short_patterns = detect_candle_patterns_on_series(df_last_week)
    df_last_month = df_ohlc[df_ohlc["timestamp"] >= df_ohlc["timestamp"].max()-timedelta(days=30)]
    classical_patterns = detect_classical_patterns(df_last_month)

    all_patterns = short_patterns + classical_patterns
    weekly_patterns = aggregate_weekly_patterns(all_patterns)
    signal = decide_weekly_signal(weekly_patterns)
    st.subheader(f"Weekly Signal: {signal}")

    df_predicted = synthesize_predicted_candles(df_last_week.tail(5), signal)
    if not df_predicted.empty:
        log_weekly_candlestick_predictions(df_predicted)

    # --- Candlestick chart ---
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

    # --- Stacked Pattern Contribution Chart ---
    bull = {k:v for k,v in weekly_patterns.items() if "bullish" in k or "Bottom" in k or "Cup & Handle" in k or "Ascending" in k or "Falling Wedge" in k}
    bear = {k:v for k,v in weekly_patterns.items() if "bearish" in k or "Top" in k or "Head & Shoulders" in k or "Descending" in k or "Rising Wedge" in k}
    neutral = {k:v for k,v in weekly_patterns.items() if "neutral" in k or "Doji" in k or "Flag/Pennant" in k}

    def color_top_3(d):
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    colors = {}
    for i, (k, _) in enumerate(sorted_items):
        if i == 0:
            colors[k] = "#a8e6cf"  # pastel green
        elif i == 1:
            colors[k] = "#dcedff"  # pastel blue
        elif i == 2:
            colors[k] = "#ffd3e0"  # pastel pink
        else:
            colors[k] = "#f0f0f0"  # light pastel gray
    return colors

    fig2 = go.Figure()
    bull_colors = color_top_3(bull)
    for pattern, count in bull.items():
        fig2.add_trace(go.Bar(y=["Bullish 📈"], x=[count], name=pattern, orientation='h', marker_color=bull_colors[pattern]))
    bear_colors = color_top_3(bear)
    for pattern, count in bear.items():
        fig2.add_trace(go.Bar(y=["Bearish 📉"], x=[count], name=pattern, orientation='h', marker_color=bear_colors[pattern]))
    neutral_colors = color_top_3(neutral)
    for pattern, count in neutral.items():
        fig2.add_trace(go.Bar(y=["Neutral ⚖️"], x=[count], name=pattern, orientation='h', marker_color=neutral_colors[pattern]))

    fig2.update_layout(
        barmode='stack',
        title="Weekly Pattern Contributions by Type",
        xaxis_title="Count",
        yaxis_title="Signal Type",
        legend_title="Patterns",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
