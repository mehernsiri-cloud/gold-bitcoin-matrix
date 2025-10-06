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
    for f in [detect_head_shoulders, detect_double_triple_top_bottom, detect_triangle_patterns,
              detect_cup_handle, detect_flags_pennants, detect_rounding_patterns, detect_wedges]:
        try:
            results += f(df)
        except Exception:
            continue
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
    if df_actual is None or df_actual.empty:
        st.error("Error: No data available to plot the candlestick chart.")
        return

    # Map columns to OHLC
    required_columns_map = {
        "timestamp": "timestamp",
        "open": "bitcoin_open",
        "high": "bitcoin_high",
        "low": "bitcoin_low",
        "close": "bitcoin_close"
    }

    missing_cols = [v for v in required_columns_map.values() if v not in df_actual.columns]
    if missing_cols:
        st.error(f"Error: Missing required columns: {missing_cols}")
        return

    df_ohlc = df_actual[[v for v in required_columns_map.values()]].rename(
        columns={v: k for k, v in required_columns_map.items()}
    )

    # Convert types safely
    df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors="coerce")
    df_ohlc.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    if df_ohlc.empty:
        st.error("No valid OHLC data to plot.")
        return

    # --- Compute weekly patterns and decide signal ---
    try:
        short_term = detect_candle_patterns_on_series(df_ohlc)
        classical = detect_classical_patterns(df_ohlc)
        all_patterns = short_term + classical
        weekly_patterns = aggregate_weekly_patterns(all_patterns)
        weekly_signal = decide_weekly_signal(weekly_patterns)
    except Exception:
        weekly_patterns = {}
        weekly_signal = "Neutral ⚖️"

    # --- Generate predicted candles ---
    df_pred = synthesize_predicted_candles(df_ohlc, weekly_signal)

    # --- Plot candlestick chart ---
    try:
        fig = go.Figure()

        # Actual candles
        fig.add_trace(go.Candlestick(
            x=df_ohlc["timestamp"],
            open=df_ohlc["open"],
            high=df_ohlc["high"],
            low=df_ohlc["low"],
            close=df_ohlc["close"],
            name="Actual",
            increasing_line_color='green',
            decreasing_line_color='red',
            increasing_fillcolor='rgba(0,255,0,0.3)',
            decreasing_fillcolor='rgba(255,0,0,0.3)'
        ))

        # Predicted candles
        if not df_pred.empty:
            fig.add_trace(go.Candlestick(
                x=df_pred["timestamp"],
                open=df_pred["open"],
                high=df_pred["high"],
                low=df_pred["low"],
                close=df_pred["close"],
                name="Predicted (Next 5 Days)",
                increasing_line_color="blue",
                decreasing_line_color="orange",
                increasing_fillcolor="rgba(0,0,255,0.3)",
                decreasing_fillcolor="rgba(255,165,0,0.3)"
            ))

        fig.update_layout(title="Bitcoin Candlestick Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering candlestick chart: {e}")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta


def render_daily_candlestick_dashboard(df_actual: pd.DataFrame):
    """
    Render a daily candlestick chart using hourly actual_data.csv,
    and predict the next 7 days using simple trend continuation logic.
    """

    st.subheader("📅 Daily Candlestick Chart with 7-Day Projection")

    try:
        # -----------------------------
        # Validate input
        # -----------------------------
        if df_actual.empty:
            st.warning("No data available for candlestick generation.")
            return

        if "timestamp" not in df_actual.columns:
            st.error("Missing 'timestamp' column in dataset.")
            return

        # Ensure datetime format and sort
        df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"], errors="coerce")
        df_actual = df_actual.dropna(subset=["timestamp"]).sort_values("timestamp")

        # -----------------------------
        # Build daily OHLC
        # -----------------------------
        required_cols = ["bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]
        if not all(col in df_actual.columns for col in required_cols):
            st.error("Missing Bitcoin OHLC columns in actual_data.csv")
            return

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
        ).reset_index()

        if df_daily.empty:
            st.warning("No valid daily OHLC data could be computed.")
            return

        # -----------------------------
        # Simple 7-day projection logic
        # -----------------------------
        last_close = df_daily["bitcoin_close"].iloc[-1]
        recent_trend = df_daily["bitcoin_close"].iloc[-7:].pct_change().mean()  # avg daily change %

        future_dates = [df_daily["timestamp"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
        predicted_rows = []

        for d in future_dates:
            last_close *= (1 + recent_trend if not pd.isna(recent_trend) else 1)
            volatility = df_daily["bitcoin_close"].pct_change().std() or 0.01

            open_ = last_close * (1 - volatility / 2)
            close = last_close * (1 + volatility / 2)
            high = max(open_, close) * (1 + volatility)
            low = min(open_, close) * (1 - volatility)

            predicted_rows.append({
                "timestamp": d,
                "open": open_,
                "high": high,
                "low": low,
                "close": close
            })

        df_pred = pd.DataFrame(predicted_rows)

        # -----------------------------
        # Plot chart
        # -----------------------------
        fig = go.Figure()

        # Actual candles
        fig.add_trace(go.Candlestick(
            x=df_daily["timestamp"],
            open=df_daily["bitcoin_open"],
            high=df_daily["bitcoin_high"],
            low=df_daily["bitcoin_low"],
            close=df_daily["bitcoin_close"],
            name="Actual (Daily)",
            increasing_line_color="green",
            decreasing_line_color="red",
            increasing_fillcolor="rgba(0,255,0,0.3)",
            decreasing_fillcolor="rgba(255,0,0,0.3)"
        ))

        # Predicted candles
        fig.add_trace(go.Candlestick(
            x=df_pred["timestamp"],
            open=df_pred["open"],
            high=df_pred["high"],
            low=df_pred["low"],
            close=df_pred["close"],
            name="Predicted (Next 7 Days)",
            increasing_line_color="blue",
            decreasing_line_color="orange",
            increasing_fillcolor="rgba(0,0,255,0.3)",
            decreasing_fillcolor="rgba(255,165,0,0.3)"
        ))

        fig.update_layout(
            title="Bitcoin Daily Candlestick with 7-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering daily candlestick chart: {e}")


    
    # --- Weekly Pattern Contributions ---
    st.write("### Weekly Pattern Contributions")
    bull_patterns = ["Bullish","Bottom","Cup & Handle","Ascending","Falling Wedge"]
    bear_patterns = ["Bearish","Top","Head & Shoulders","Descending","Rising Wedge"]
    neutral_patterns = ["Neutral","Doji","Flag/Pennant"]

    bull = {k:v for k,v in weekly_patterns.items() if any(bp in k for bp in bull_patterns)}
    bear = {k:v for k,v in weekly_patterns.items() if any(bp in k for bp in bear_patterns)}
    neutral = {k:v for k,v in weekly_patterns.items() if any(bp in k for bp in neutral_patterns)}

    def color_top_3(d: Dict[str,int]) -> Dict[str,str]:
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        colors = {}
        for i, (k, _) in enumerate(sorted_items):
            if i == 0: colors[k] = "#a8e6cf"
            elif i == 1: colors[k] = "#dcedff"
            elif i == 2: colors[k] = "#ffd3e0"
            else: colors[k] = "#f0f0f0"
        return colors

    fig2 = go.Figure()
    for patterns_dict, y_label in [(bull, "Bullish 📈"), (bear, "Bearish 📉"), (neutral, "Neutral ⚖️")]:
        colors = color_top_3(patterns_dict)
        for pattern, count in patterns_dict.items():
            fig2.add_trace(go.Bar(y=[y_label], x=[count], name=pattern,
                                  orientation='h', marker_color=colors[pattern]))
    fig2.update_layout(barmode='stack', title="Weekly Pattern Contributions by Type",
                       xaxis_title="Count", yaxis_title="Signal Type", legend_title="Patterns", height=500)
    st.plotly_chart(fig2, use_container_width=True)
    # --- Weekly Candlestick Aggregation Chart ---
    st.write("### Weekly Candlestick View (Aggregated from Hourly Data)")

    try:
        df_weekly = df_ohlc.resample('W', on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()

        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Candlestick(
            x=df_weekly["timestamp"],
            open=df_weekly["open"],
            high=df_weekly["high"],
            low=df_weekly["low"],
            close=df_weekly["close"],
            name="Weekly Candlestick",
            increasing_line_color='green',
            decreasing_line_color='red',
            increasing_fillcolor='rgba(0,255,0,0.3)',
            decreasing_fillcolor='rgba(255,0,0,0.3)'
        ))

        fig_weekly.update_layout(
            title="Aggregated Weekly Candlestick (From Hourly Data)",
            xaxis_title="Week",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=500
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating weekly candlestick chart: {e}")
