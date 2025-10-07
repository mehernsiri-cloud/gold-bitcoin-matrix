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
import plotly.graph_objects as go

def classify_pattern_sentiment(pattern_name: str) -> str:
    """Classify pattern type into Bullish / Bearish / Neutral."""
    name = pattern_name.lower()
    if any(k in name for k in ["bullish", "hammer", "engulfing", "morning", "piercing", "kicker", "rising", "ascending", "bottom"]):
        return "Bullish"
    elif any(k in name for k in ["bearish", "shooting", "dark", "hanging", "evening", "tweezer", "falling", "top"]):
        return "Bearish"
    else:
        return "Neutral"


def render_candlestick_dashboard(df_actual: pd.DataFrame):
    st.header("📊 Bitcoin Candlestick Dashboard (Enhanced)")

    if df_actual is None or df_actual.empty:
        st.error("No data available to plot the candlestick chart.")
        return

    # Column mapping
    col_map = {
        "timestamp": "timestamp",
        "open": "bitcoin_open",
        "high": "bitcoin_high",
        "low": "bitcoin_low",
        "close": "bitcoin_close"
    }

    missing = [v for v in col_map.values() if v not in df_actual.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    df_ohlc = df_actual[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})
    df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], errors="coerce")
    df_ohlc.dropna(inplace=True)

    # Pattern detection
    with st.expander("📈 Detected Patterns and Signals", expanded=True):
        short = detect_candle_patterns_on_series(df_ohlc)
        classical = detect_classical_patterns(df_ohlc)
        all_patterns = short + classical
        weekly_patterns = aggregate_weekly_patterns(all_patterns)
        signal = decide_weekly_signal(weekly_patterns)

        st.metric("Overall Weekly Signal", signal)

        # --- 📊 Unified Sentiment Overview (Stacked Horizontal Bar) ---
        st.subheader("📊 Pattern Sentiment Overview")

        if weekly_patterns:
            # Convert to DataFrame
            pattern_df = pd.DataFrame(list(weekly_patterns.items()), columns=["Pattern", "Count"])
            pattern_df["Sentiment"] = pattern_df["Pattern"].apply(classify_pattern_sentiment)

            sentiments = ["Bullish", "Neutral", "Bearish"]
            colors = {"Bullish": "#2ecc71", "Neutral": "#95a5a6", "Bearish": "#e74c3c"}

            fig = go.Figure()

            # Add one stacked bar per sentiment
            for sentiment in sentiments:
                subset = pattern_df[pattern_df["Sentiment"] == sentiment]
                if subset.empty:
                    continue

                for _, row in subset.iterrows():
                    fig.add_trace(go.Bar(
                        y=[sentiment],
                        x=[row["Count"]],
                        orientation="h",
                        name=row["Pattern"],
                        text=f"{row['Pattern']} ({row['Count']})",
                        textposition="inside",
                        insidetextanchor="middle",
                        hovertemplate="%{text}<extra></extra>",
                        marker=dict(color=colors[sentiment], line=dict(width=0.5, color="white"))
                    ))

            fig.update_layout(
                barmode="stack",
                template="plotly_white",
                height=350,
                showlegend=False,
                xaxis_title="Count",
                yaxis_title="Sentiment Category",
                margin=dict(l=50, r=50, t=30, b=40),
                title="Detected Pattern Composition by Sentiment"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No patterns detected for this period.")

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

    if df_actual is None or df_actual.empty:
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

    # --- Forecast using Holt-Winters + Monte Carlo ensemble ---
    last_close = df_daily["close"].iloc[-1]
    returns = df_daily["close"].pct_change().dropna()
    recent_return = returns.tail(7).mean()
    recent_vol = returns.tail(20).std() or 0.01

    # 1) Holt-Winters forecast
    try:
        hw_model = ExponentialSmoothing(df_daily['close'], trend='add', seasonal=None, initialization_method="estimated")
        hw_fit = hw_model.fit()
        hw_pred = hw_fit.forecast(7).values
    except:
        hw_pred = np.full(7, last_close * (1 + recent_return))

    # 2) Monte-Carlo simulation
    N = 500
    mc_paths = np.zeros((N, 8))
    mc_paths[:, 0] = last_close
    for t in range(1, 8):
        z = np.random.normal(recent_return, recent_vol, size=N)
        mc_paths[:, t] = mc_paths[:, t-1] * (1 + z)
    mc_median = np.median(mc_paths, axis=0)[1:]
    mc_p10 = np.percentile(mc_paths, 10, axis=0)[1:]
    mc_p90 = np.percentile(mc_paths, 90, axis=0)[1:]

    # 3) Linear baseline (optional)
    linear = last_close * np.cumprod(1 + np.full(7, recent_return))

    # Ensemble (weights)
    w = np.array([0.5, 0.3, 0.2])
    ensemble = w[0]*hw_pred + w[1]*mc_median + w[2]*linear

    # Build predicted DataFrame
    future_dates = [df_daily["timestamp"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
    df_pred = pd.DataFrame({
        "timestamp": future_dates,
        "close": ensemble,
        "low": mc_p10,
        "high": mc_p90
    })

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_daily["timestamp"], df_daily["close"], label="Historical Close", color="#1f77b4")
    ax.plot(df_pred["timestamp"], df_pred["close"], label="Forecast (Ensemble)", color="#ff7f0e")
    ax.fill_between(df_pred["timestamp"], df_pred["low"], df_pred["high"], color="#ffbb78", alpha=0.3, label="Uncertainty (10-90%)")

    ax.set_title("Bitcoin Daily Candlestick + 7-Day Forecast", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
