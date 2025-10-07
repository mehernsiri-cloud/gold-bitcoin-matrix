# candlestick_predictions.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


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
    """
    Render a robust daily candlestick dashboard (from hourly df_actual) and
    show a 7-day probabilistic forecast. This function is defensive: it will
    try to find OHLC columns under several common names and fall back to a
    single price column if needed.
    """
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st
    from datetime import timedelta

    st.header("📅 Daily Candlestick Dashboard (7-Day Projection)")

    # Basic validation
    if df_actual is None or df_actual.empty:
        st.warning("No data available.")
        return

    # Work on a copy
    df = df_actual.copy()

    # --- Helper to find a best-fit column name from candidates ---
    def find_col(cols, candidates):
        cols_low = {c: c.lower() for c in cols}
        # 1) exact match on candidate
        for cand in candidates:
            for orig, low in cols_low.items():
                if low == cand:
                    return orig
        # 2) candidate substring in column name
        for cand in candidates:
            for orig, low in cols_low.items():
                if cand in low:
                    return orig
        return None

    cols = list(df.columns)

    # timestamp detection
    ts_col = find_col(cols, ["timestamp", "time", "date", "datetime"])
    if ts_col is None:
        st.error("Missing a timestamp column (looked for timestamp/time/date/datetime). Columns: " + ", ".join(cols))
        return
    # normalize timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        st.error("All timestamp values are invalid after parsing.")
        return
    df = df.sort_values(ts_col)

    # OHLC detection candidates
    open_col = find_col(cols, ["bitcoin_open", "open", "open_price", "btc_open"])
    high_col = find_col(cols, ["bitcoin_high", "high", "high_price", "btc_high"])
    low_col = find_col(cols, ["bitcoin_low", "low", "low_price", "btc_low"])
    close_col = find_col(cols, ["bitcoin_close", "close", "close_price", "bitcoin_actual", "bitcoin_price", "price", "last", "value"])

    # If no close/price found -> fail gracefully
    if close_col is None:
        st.error("Missing price/close information. Expected one of: bitcoin_close, bitcoin_actual, price, close, last. Columns: " + ", ".join(cols))
        return

    # Convert numeric columns to numeric dtype (coerce non-numeric -> NaN)
    for c in set([open_col, high_col, low_col, close_col]) - {None}:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If open/high/low missing, we'll fall back to using close_col for them
    src_open = open_col if open_col is not None else close_col
    src_high = high_col if high_col is not None else close_col
    src_low = low_col if low_col is not None else close_col
    src_close = close_col

    # Build daily OHLC by resampling on timestamp
    df_indexed = df.set_index(ts_col)
    try:
        daily_open = df_indexed[src_open].resample("D").first()
        daily_high = df_indexed[src_high].resample("D").max()
        daily_low = df_indexed[src_low].resample("D").min()
        daily_close = df_indexed[src_close].resample("D").last()
    except Exception as e:
        st.error(f"Error resampling data to daily OHLC: {e}")
        return

    df_daily = pd.concat([daily_open, daily_high, daily_low, daily_close], axis=1)
    df_daily.columns = ["open", "high", "low", "close"]
    df_daily = df_daily.dropna(subset=["close"]).reset_index()

    if df_daily.empty:
        st.warning("No valid daily OHLC rows after resampling. Check your input data.")
        st.write("Sample of input columns:", cols[:20])
        return

    # Make sure numeric
    for c in ["open", "high", "low", "close"]:
        df_daily[c] = pd.to_numeric(df_daily[c], errors="coerce")

    # Indicators: SMA and rolling volatility for bands
    df_daily["SMA20"] = df_daily["close"].rolling(window=20, min_periods=1).mean()
    df_daily["ret"] = df_daily["close"].pct_change()
    df_daily["vol20"] = df_daily["ret"].rolling(window=20, min_periods=1).std().fillna(0)
    df_daily["upper"] = df_daily["SMA20"] + 2 * df_daily["vol20"] * df_daily["SMA20"]
    df_daily["lower"] = df_daily["SMA20"] - 2 * df_daily["vol20"] * df_daily["SMA20"]

    # --- Forecast using Monte-Carlo (geometric-ish) ---
    last_close = float(df_daily["close"].iloc[-1])
    returns = df_daily["ret"].dropna()
    mu = float(returns.tail(7).mean()) if not returns.empty else 0.0
    sigma = float(returns.tail(20).std()) if not returns.empty else 0.01
    sigma = max(sigma, 1e-4)

    days = 7
    N = 500
    rng = np.random.default_rng()
    # simulate N paths
    z = rng.normal(loc=0.0, scale=1.0, size=(N, days))
    # use daily drift mu and volatility sigma
    paths = np.zeros((N, days + 1), dtype=float)
    paths[:, 0] = last_close
    for t in range(days):
        # multiplicative step: S_{t+1} = S_t * (1 + mu + sigma * z)
        paths[:, t + 1] = paths[:, t] * (1.0 + mu + sigma * z[:, t])

    mc_median = np.median(paths, axis=0)[1:]  # length = days
    mc_p10 = np.percentile(paths, 10, axis=0)[1:]
    mc_p90 = np.percentile(paths, 90, axis=0)[1:]

    # Build predicted DataFrame of OHLC-like candles
    prev_close = last_close
    pred_rows = []
    pred_dates = [df_daily[ts_col].iloc[-1] + timedelta(days=i) for i in range(1, days + 1)]
    for i, dt in enumerate(pred_dates):
        close_p = float(mc_median[i])
        open_p = float(prev_close)  # open = previous close (common assumption)
        # small wiggle to define high/low
        wiggle = max(0.0025, sigma * 0.5)
        high_p = max(open_p, close_p) * (1.0 + wiggle)
        low_p = min(open_p, close_p) * (1.0 - wiggle)
        pred_rows.append({"timestamp": dt, "open": open_p, "high": high_p, "low": low_p, "close": close_p})
        prev_close = close_p

    df_pred = pd.DataFrame(pred_rows)

    # --- Plot using Plotly (actual + predicted) ---
    fig = go.Figure()

    # hover text for actuals (include % change)
    pct = df_daily["close"].pct_change().fillna(0)
    hover_actual = [
        f"Date: {d:%Y-%m-%d}<br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}<br>Change: {p:.2%}"
        for d, o, h, l, c, p in zip(df_daily["timestamp"], df_daily["open"], df_daily["high"], df_daily["low"], df_daily["close"], pct)
    ]
    fig.add_trace(go.Candlestick(
        x=df_daily["timestamp"],
        open=df_daily["open"],
        high=df_daily["high"],
        low=df_daily["low"],
        close=df_daily["close"],
        name="Actual",
        increasing_line_color="#00CC96",
        decreasing_line_color="#EF553B",
        hovertext=hover_actual,
        hoverinfo="text"
    ))

    # SMA line
    fig.add_trace(go.Scatter(
        x=df_daily["timestamp"], y=df_daily["SMA20"],
        mode="lines", name="SMA 20", line=dict(color="#636EFA", width=2), hoverinfo="skip"
    ))

    # Volatility bands (fill between)
    fig.add_trace(go.Scatter(
        x=df_daily["timestamp"], y=df_daily["upper"],
        line=dict(color="rgba(200,200,200,0.2)", width=1),
        name="Upper Band", hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_daily["timestamp"], y=df_daily["lower"],
        fill='tonexty',
        line=dict(color="rgba(200,200,200,0.2)", width=1),
        name="Lower Band", hoverinfo="skip", showlegend=False
    ))

    # Add a subtle forecast zone rectangle
    if not df_pred.empty:
        x0 = df_pred["timestamp"].iloc[0]
        x1 = df_pred["timestamp"].iloc[-1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(150,150,150,0.06)", line_width=0, layer="below")

    # Predicted candles: draw individually so we can color each one correctly
    pastel_green = "#B2F7EF"
    pastel_red = "#FFB6B9"
    for _, r in df_pred.iterrows():
        color = pastel_green if r["close"] >= r["open"] else pastel_red
        hover_pred = (f"Predicted<br>Date: {r['timestamp']:%Y-%m-%d}<br>"
                      f"Open: {r['open']:.2f}<br>High: {r['high']:.2f}<br>Low: {r['low']:.2f}<br>Close: {r['close']:.2f}")
        fig.add_trace(go.Candlestick(
            x=[r["timestamp"]],
            open=[r["open"]],
            high=[r["high"]],
            low=[r["low"]],
            close=[r["close"]],
            name="Predicted",
            increasing_line_color=color,
            decreasing_line_color=color,
            opacity=0.8,
            hovertext=hover_pred,
            hoverinfo="text",
            showlegend=False
        ))

    # Optional: show MC band median & percentiles as line + shaded area
    df_band = pd.DataFrame({
        "timestamp": pred_dates,
        "p10": mc_p10,
        "p50": mc_median,
        "p90": mc_p90
    })
    # median line
    fig.add_trace(go.Scatter(
        x=df_band["timestamp"], y=df_band["p50"],
        mode="lines", name="Forecast median", line=dict(color="#FF7F0E", width=1.8)
    ))
    # shaded 10-90 band
    fig.add_trace(go.Scatter(
        x=list(df_band["timestamp"]) + list(df_band["timestamp"][::-1]),
        y=list(df_band["p90"]) + list(df_band["p10"][::-1]),
        fill='toself',
        fillcolor='rgba(255,127,14,0.12)',
        line=dict(color='rgba(255,127,14,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Forecast 10-90%"
    ))

    fig.update_layout(
        title="Bitcoin Daily Candlestick + 7-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

