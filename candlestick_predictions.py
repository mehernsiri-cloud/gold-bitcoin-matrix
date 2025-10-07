# candlestick_predictions.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple, Dict
import numpy as np

DATA_DIR = "data"
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")

# ===============================================================
# 🔍 1. SHORT-TERM (3-CANDLE) PATTERNS
# ===============================================================
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


# ===============================================================
# 🔷 2. CLASSICAL MULTI-CANDLE PATTERNS
# ===============================================================
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
    for f in [detect_head_shoulders, detect_double_triple_top_bottom, detect_triangle_patterns,
              detect_cup_handle, detect_flags_pennants, detect_rounding_patterns, detect_wedges]:
        try:
            results += f(df)
        except Exception:
            continue
    return results


# (for brevity you can keep your existing definitions — identical to your current version)
# ⬆️ no change, they remain as in your current 500-line file


# ===============================================================
# 📈 3. PATTERN AGGREGATION & WEIGHTING
# ===============================================================
def aggregate_weekly_patterns(patterns: List[Tuple[pd.Timestamp, str]]) -> Dict[str, int]:
    agg = {}
    for _, pat in patterns:
        agg[pat] = agg.get(pat, 0) + 1
    return agg


def decide_weekly_signal(weekly_patterns: Dict[str, int]) -> str:
    if not weekly_patterns:
        return "Neutral ⚖️"

    bull_patterns = ["Bullish", "Hammer", "Bottom", "Cup & Handle", "Falling Wedge", "Ascending Triangle"]
    bear_patterns = ["Bearish", "Shooting Star", "Top", "Head & Shoulders", "Rising Wedge", "Descending Triangle"]

    # Weighted scoring
    weights = {
        "Bullish Engulfing": 1.2, "Bearish Engulfing": 1.2,
        "Head & Shoulders (bearish)": 1.5, "Cup & Handle (bullish)": 1.5,
        "Double Bottom (bullish)": 1.3, "Double Top (bearish)": 1.3,
        "Hammer (bullish)": 1.1, "Shooting Star (bearish)": 1.1
    }

    bull = sum(v * weights.get(k, 1) for k, v in weekly_patterns.items() if any(bp in k for bp in bull_patterns))
    bear = sum(v * weights.get(k, 1) for k, v in weekly_patterns.items() if any(bp in k for bp in bear_patterns))

    if bull > bear:
        return "Bullish 📈"
    elif bear > bull:
        return "Bearish 📉"
    else:
        return "Neutral ⚖️"


# ===============================================================
# 🧮 4. VOLATILITY-ADJUSTED SYNTHETIC CANDLE CREATION
# ===============================================================
def synthesize_predicted_candles(df_ohlc: pd.DataFrame, signal: str) -> pd.DataFrame:
    if df_ohlc.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    last_close = df_ohlc.iloc[-1].close
    base_date = df_ohlc.iloc[-1].timestamp
    drift_factor = 0.0
    if "Bullish" in signal:
        drift_factor = 0.01
    elif "Bearish" in signal:
        drift_factor = -0.01

    # Realized volatility (using last 20 candles)
    recent_vol = df_ohlc["close"].pct_change().std() or 0.005
    adj_drift = drift_factor * (1 + 5 * recent_vol)

    candles = []
    for i in range(1, 6):
        date = base_date + timedelta(days=i)
        daily_move = np.random.normal(adj_drift, recent_vol)
        open_p = last_close * (1 + daily_move * 0.3)
        close_p = open_p * (1 + daily_move)
        high_p = max(open_p, close_p) * (1 + abs(daily_move) * 0.5)
        low_p = min(open_p, close_p) * (1 - abs(daily_move) * 0.5)
        candles.append((date, open_p, high_p, low_p, close_p))
        last_close = close_p

    return pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close"])


# ===============================================================
# 🧾 5. LOGGING (unchanged)
# ===============================================================
def log_weekly_candlestick_predictions(df_pred: pd.DataFrame):
    if df_pred.empty:
        return
    rows = [{
        "timestamp": r.timestamp,
        "asset": "Bitcoin",
        "predicted_price": r.close,
        "method": "Candlestick Synthetic"
    } for _, r in df_pred.iterrows()]

    df_new = pd.DataFrame(rows)
    if os.path.exists(AI_PRED_FILE):
        df_old = pd.read_csv(AI_PRED_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(AI_PRED_FILE, index=False)


# ===============================================================
# 📊 6. UNIFIED PLOT UTILITY
# ===============================================================
def plot_candlestick(df_actual, df_pred=None, title="Candlestick Chart", height=600):
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Candlestick(
        x=df_actual["timestamp"],
        open=df_actual["open"], high=df_actual["high"],
        low=df_actual["low"], close=df_actual["close"],
        name="Actual",
        increasing_line_color="green",
        decreasing_line_color="red",
        increasing_fillcolor="rgba(0,255,0,0.3)",
        decreasing_fillcolor="rgba(255,0,0,0.3)"
    ))

    # Predicted
    if df_pred is not None and not df_pred.empty:
        fig.add_trace(go.Candlestick(
            x=df_pred["timestamp"],
            open=df_pred["open"], high=df_pred["high"],
            low=df_pred["low"], close=df_pred["close"],
            name="Predicted",
            increasing_line_color="blue",
            decreasing_line_color="orange",
            increasing_fillcolor="rgba(0,0,255,0.25)",
            decreasing_fillcolor="rgba(255,165,0,0.25)"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=height,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.02, x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# ===============================================================
# 🖥️ 7. MAIN DASHBOARD RENDERING
# ===============================================================
import plotly.express as px

import matplotlib.pyplot as plt

# --- Sentiment data ---
bullish = weekly_patterns.get("Bullish", 0)
neutral = weekly_patterns.get("Neutral", 0)
bearish = weekly_patterns.get("Bearish", 0)

# Pastel colors for each sentiment
colors = {
    "Bullish": "#A8E6CF",  # pastel green
    "Neutral": "#FFD3B6",  # pastel peach
    "Bearish": "#FFAAA5",  # pastel red
}

# Data setup
categories = ["Bullish 🟢", "Neutral ⚪", "Bearish 🔴"]
values = [bullish, neutral, bearish]
bar_colors = [colors["Bullish"], colors["Neutral"], colors["Bearish"]]

# Create figure
fig, ax = plt.subplots(figsize=(6, 2.8))

# Plot each sentiment as its own horizontal bar
y_positions = range(len(categories))
bars = ax.barh(y_positions, values, color=bar_colors, edgecolor="white")

# Add count labels inside each bar
for i, bar in enumerate(bars):
    width = bar.get_width()
    if width > 0:
        ax.text(
            width / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black"
        )

# Clean style
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)
ax.set_xticks([])
ax.set_xlabel("")
ax.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1)
ax.set_title("📊 Weekly Pattern Sentiment Breakdown", fontsize=12, pad=10)

# Remove borders for a minimalist look
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

st.pyplot(fig)






    # Prediction
    df_pred = synthesize_predicted_candles(df_ohlc, signal)

    # Volatility metrics
    with st.expander("📊 Market Metrics", expanded=False):
        vol = df_ohlc["close"].pct_change().std() * 100
        avg_move = df_ohlc["close"].pct_change().mean() * 100
        trend_strength = avg_move / (vol + 1e-6)
        st.metric("Volatility (%)", f"{vol:.2f}")
        st.metric("Avg Daily Move (%)", f"{avg_move:.2f}")
        st.metric("Trend Strength", f"{trend_strength:.2f}")

    # Charts
    st.subheader("Hourly Candlestick with 5-Day Forecast")
    plot_candlestick(df_ohlc, df_pred, title="Hourly Candlestick + 5-Day Forecast")

    # Weekly aggregation
    st.subheader("Weekly Aggregated Candlestick View")
    try:
        df_weekly = df_ohlc.resample("W", on="timestamp").agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna().reset_index()
        plot_candlestick(df_weekly, title="Aggregated Weekly Candlestick", height=500)
    except Exception as e:
        st.warning(f"Could not create weekly chart: {e}")


# ===============================================================
# 📅 8. DAILY CANDLESTICK DASHBOARD
# ===============================================================
def render_daily_candlestick_dashboard(df_actual: pd.DataFrame):
    st.header("📅 Daily Candlestick Dashboard (7-Day Projection)")

    if df_actual.empty:
        st.warning("No data available.")
        return

    df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"], errors="coerce")
    df_actual = df_actual.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Build daily OHLC
    df_daily = (
        df_actual.set_index("timestamp")
        .resample("D")
        .agg({
            "bitcoin_open": "first",
            "bitcoin_high": "max",
            "bitcoin_low": "min",
            "bitcoin_close": "last"
        })
        .dropna()
        .reset_index()
        .rename(columns={
            "bitcoin_open": "open",
            "bitcoin_high": "high",
            "bitcoin_low": "low",
            "bitcoin_close": "close"
        })
    )

    if df_daily.empty:
        st.warning("No valid daily data.")
        return

    # Simple forecast
    last_close = df_daily["close"].iloc[-1]
    avg_change = df_daily["close"].pct_change().mean()
    vol = df_daily["close"].pct_change().std() or 0.01

    future = []
    for i in range(1, 8):
        last_close *= (1 + avg_change)
        open_ = last_close * (1 - vol / 2)
        close = last_close * (1 + vol / 2)
        high = max(open_, close) * (1 + vol)
        low = min(open_, close) * (1 - vol)
        future.append({
            "timestamp": df_daily["timestamp"].iloc[-1] + timedelta(days=i),
            "open": open_, "high": high, "low": low, "close": close
        })
    df_pred = pd.DataFrame(future)

    plot_candlestick(df_daily, df_pred, title="Daily Candlestick + 7-Day Forecast")
