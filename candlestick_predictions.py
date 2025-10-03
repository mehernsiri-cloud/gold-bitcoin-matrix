# candlestick_predictions.py
# All candlestick logic and UI rendering (corrected, robust)
#
# Provides:
#  - Pattern detection (Doji, Hammer, Engulfing, etc.)
#  - Weekly aggregation of patterns
#  - Weekly trading signal (Bullish/Bearish/Neutral)
#  - Synthetic prediction of next week candles
#  - Logging into data/ai_predictions_log.csv
#
# Author: ChatGPT (for user)
# Date: 2025-10-03 (corrected)
# -----------------------------------------------------------------------------

import os
from datetime import timedelta
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure data directory exists
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")


# -------------------------------------------------------------------
# PATTERN DETECTION
# -------------------------------------------------------------------
def detect_patterns_in_3_candles(window: pd.DataFrame) -> List[str]:
    """
    Detect a few simple single/multi-candle patterns based on the last candle
    of a 3-candle window. Returns list of pattern names (may be empty).
    """
    patterns: List[str] = []
    if window is None or window.shape[0] != 3:
        return patterns

    try:
        c0 = window.iloc[2]  # latest
        c1 = window.iloc[1]
        c2 = window.iloc[0]

        open0 = float(c0["open"])
        high0 = float(c0["high"])
        low0 = float(c0["low"])
        close0 = float(c0["close"])

        body0 = abs(close0 - open0)
        rng0 = max(high0 - low0, 1e-9)
        upper_wick = high0 - max(open0, close0)
        lower_wick = min(open0, close0) - low0

        # Doji (small body)
        if body0 < 0.15 * rng0:
            patterns.append("Doji (indecision)")

        # Hammer (bullish) / Shooting Star (bearish) heuristics
        if body0 < 0.3 * rng0:
            if lower_wick > 2 * body0 and close0 > open0:
                patterns.append("Hammer (bullish)")
            if upper_wick > 2 * body0 and close0 < open0:
                patterns.append("Shooting Star (bearish)")

        # Bullish Engulfing
        try:
            open1 = float(c1["open"])
            close1 = float(c1["close"])
            if (close1 < open1) and (close0 > open0) and (close0 > open1) and (open0 < close1):
                patterns.append("Bullish Engulfing")
            # Bearish Engulfing
            if (close1 > open1) and (close0 < open0) and (close0 < open1) and (open0 > close1):
                patterns.append("Bearish Engulfing")
        except Exception:
            # If prior candle parsing fails, skip engulfing checks
            pass

    except Exception:
        # Any unexpected parsing error: return whatever we've collected
        return patterns

    # dedupe preserving order
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def detect_candle_patterns_on_series(df_ohlc: pd.DataFrame) -> List[Tuple[pd.Timestamp, List[str]]]:
    """
    Apply 3-candle sliding window over an OHLC series and collect detected patterns.
    Returns list of tuples (timestamp_of_last_candle, [patterns_for_that_candle]).
    """
    results: List[Tuple[pd.Timestamp, List[str]]] = []
    if df_ohlc is None or df_ohlc.shape[0] < 3:
        return results

    df = df_ohlc.sort_values("timestamp").reset_index(drop=True)
    for i in range(2, len(df)):
        window = df.iloc[i - 2 : i + 1].copy()
        ts = window.iloc[2]["timestamp"]
        pats = detect_patterns_in_3_candles(window)
        results.append((ts, pats))
    return results


# -------------------------------------------------------------------
# WEEKLY AGGREGATION & SIGNAL
# -------------------------------------------------------------------
def aggregate_weekly_patterns(detections: List[Tuple[pd.Timestamp, List[str]]]) -> Dict[str, int]:
    """
    Turn detections into aggregated counts by pattern name.
    """
    agg: Dict[str, int] = {}
    for _, pats in detections:
        for p in pats:
            agg[p] = agg.get(p, 0) + 1
    return agg


def decide_weekly_signal(weekly_patterns: Dict[str, int]) -> str:
    """
    Convert aggregated pattern counts into a final weekly signal.
    Conservative rule:
      - if bullish_count > bearish_count => Bullish
      - if bearish_count > bullish_count => Bearish
      - else => Neutral
    We consider patterns containing 'Bull'/'Hammer' bullish, and 'Bear'/'Shooting' bearish.
    """
    bull = sum(v for k, v in weekly_patterns.items() if ("bull" in k.lower()) or ("hammer" in k.lower()))
    bear = sum(v for k, v in weekly_patterns.items() if ("bear" in k.lower()) or ("shooting" in k.lower()))
    if bull > bear:
        return "Bullish"
    elif bear > bull:
        return "Bearish"
    else:
        return "Neutral"


# -------------------------------------------------------------------
# SYNTHETIC NEXT-WEEK CANDLES
# -------------------------------------------------------------------
def synthesize_predicted_candles(start_date: pd.Timestamp, start_open: float, drift_per_day: float, n_days: int = 7) -> pd.DataFrame:
    """
    Deterministic synthetic candle generator.
    - start_date: last observed date (predictions begin at next day)
    - start_open: open price for first predicted day (often last close)
    - drift_per_day: fractional drift (e.g. 0.01 = +1% per day)
    - n_days: number of days to generate
    Returns DataFrame with columns: timestamp, open, high, low, close
    """
    rows = []
    prev_close = float(start_open)
    current_date = pd.to_datetime(start_date)
    for i in range(1, n_days + 1):
        pred_date = (current_date + pd.Timedelta(days=i)).normalize()
        open_price = prev_close
        close_price = open_price * (1.0 + drift_per_day)
        high_price = max(open_price, close_price) * 1.002
        low_price = min(open_price, close_price) * 0.998
        rows.append({
            "timestamp": pred_date,
            "open": round(open_price, 8),
            "high": round(high_price, 8),
            "low": round(low_price, 8),
            "close": round(close_price, 8)
        })
        prev_close = close_price
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# LOGGING (to data/ai_predictions_log.csv)
# -------------------------------------------------------------------
def log_weekly_candlestick_predictions(pred_df: pd.DataFrame, asset_name: str = "Bitcoin", method: str = "candlestick"):
    """
    Append predicted daily closes to the AI predictions CSV.
    Writes rows: timestamp (ISO), asset, predicted_price, method.
    Keeps it simple: no dedupe here (caller can handle reloading/cleanup).
    """
    if pred_df is None or pred_df.empty:
        return

    # prepare rows
    rows = []
    for _, r in pred_df.iterrows():
        ts = r["timestamp"]
        # ensure string ISO
        try:
            ts_iso = pd.to_datetime(ts).isoformat()
        except Exception:
            ts_iso = str(ts)
        rows.append({"timestamp": ts_iso, "asset": asset_name, "predicted_price": float(r["close"]), "method": method})

    df_new = pd.DataFrame(rows)

    # append to CSV safely
    if os.path.exists(AI_PRED_FILE):
        try:
            df_old = pd.read_csv(AI_PRED_FILE)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    try:
        df_all.to_csv(AI_PRED_FILE, index=False)
    except Exception:
        # last resort: try to create the file
        try:
            df_new.to_csv(AI_PRED_FILE, index=False)
        except Exception:
            st.warning(f"Unable to write predictions to {AI_PRED_FILE}")


# -------------------------------------------------------------------
# HELPERS: prepare OHLC from df_actual (robust)
# -------------------------------------------------------------------
def prepare_ohlc_from_actual(df_actual: pd.DataFrame) -> pd.DataFrame:
    """
    Build an OHLC DataFrame from df_actual with robust fallbacks:
      1) If bitcoin_open/high/low/close exist -> use them (rename to open/high/low/close).
      2) Else if bitcoin_actual exists -> synthesize OHLC per row using previous close as open.
      3) Else if single 'actual' column exists -> create uniform OHLC from it.
      4) Else return empty DataFrame.
    Ensures 'timestamp' is datetime and sorted.
    """
    if df_actual is None or df_actual.empty:
        return pd.DataFrame()

    df = df_actual.copy()

    # normalize timestamp column if present
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass

    # Case 1: explicit OHLC columns (various possible names)
    # prefer bitcoin_open/high/low/close
    if all(c in df.columns for c in ["bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]):
        df_ohlc = df[["timestamp", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]].rename(columns={
            "bitcoin_open": "open", "bitcoin_high": "high", "bitcoin_low": "low", "bitcoin_close": "close"
        }).dropna(subset=["open", "high", "low", "close"])
        return df_ohlc.sort_values("timestamp").reset_index(drop=True)

    # Case 2: single price series named bitcoin_actual
    if "bitcoin_actual" in df.columns:
        d = df[["timestamp", "bitcoin_actual"]].rename(columns={"bitcoin_actual": "close"}).copy()
        # open = previous close (shift)
        d = d.sort_values("timestamp").reset_index(drop=True)
        d["open"] = d["close"].shift(1).fillna(d["close"])
        d["high"] = d[["open", "close"]].max(axis=1) * 1.002
        d["low"] = d[["open", "close"]].min(axis=1) * 0.998
        return d[["timestamp", "open", "high", "low", "close"]]

    # Case 3: generic 'actual' column
    if "actual" in df.columns:
        d = df[["timestamp", "actual"]].rename(columns={"actual": "close"}).copy()
        d = d.sort_values("timestamp").reset_index(drop=True)
        d["open"] = d["close"].shift(1).fillna(d["close"])
        d["high"] = d[["open", "close"]].max(axis=1) * 1.002
        d["low"] = d[["open", "close"]].min(axis=1) * 0.998
        return d[["timestamp", "open", "high", "low", "close"]]

    # No usable columns
    return pd.DataFrame()


# -------------------------------------------------------------------
# UI RENDERING (main entrypoint used by app.py)
# -------------------------------------------------------------------
def render_candlestick_dashboard(df_actual: pd.DataFrame, df_ai_pred_log: pd.DataFrame = None):
    st.title("🕯️ Candlestick Predictions (Bitcoin)")

    # build OHLC robustly
    df_ohlc = prepare_ohlc_from_actual(df_actual)
    if df_ohlc is None or df_ohlc.empty:
        st.error("No OHLC or usable price series found in actual data (need bitcoin_open/high/low/close or bitcoin_actual or actual).")
        return

    # ensure timestamp is datetime
    try:
        df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], errors="coerce")
    except Exception:
        pass
    df_ohlc = df_ohlc.sort_values("timestamp").reset_index(drop=True)

    # UI: parameters (small set exposed)
    st.markdown("### Parameters")
    lookback_days = st.number_input("Lookback (days) for pattern detection", min_value=3, max_value=30, value=7, step=1)
    n_forecast_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=14, value=7, step=1)
    bull_drift_pct = st.number_input("Daily bullish drift (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
    bear_drift_pct = st.number_input("Daily bearish drift (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
    method_name = st.text_input("Logging method name (for ai_predictions_log)", value="candlestick")

    # select recent lookback candles by time or rows (prefer time-based)
    try:
        cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=lookback_days)
        df_recent = df_ohlc[df_ohlc["timestamp"] >= cutoff].copy()
        if df_recent.shape[0] < 3:
            df_recent = df_ohlc.tail(lookback_days).copy()
    except Exception:
        df_recent = df_ohlc.tail(lookback_days).copy()

    st.markdown(f"**Using {len(df_recent)} candles for pattern detection (lookback {lookback_days} days)**")

    if df_recent.shape[0] < 3:
        st.info("Not enough candles to run pattern detection (need at least 3).")
        return

    # show recent candlesticks
    fig_recent = go.Figure(data=[go.Candlestick(
        x=df_recent["timestamp"],
        open=df_recent["open"],
        high=df_recent["high"],
        low=df_recent["low"],
        close=df_recent["close"],
        name="Recent BTC"
    )])
    fig_recent.update_layout(title=f"Bitcoin: Last {len(df_recent)} Candles (for pattern detection)", xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig_recent, use_container_width=True)

    # detect patterns and aggregate
    detections = detect_candle_patterns_on_series(df_recent)
    weekly_patterns = aggregate_weekly_patterns(detections)
    st.write("#### Detected patterns (latest windows)")
    st.write({str(k): v for k, v in weekly_patterns.items()})

    # decide signal
    final_signal = decide_weekly_signal(weekly_patterns)
    if final_signal == "Bullish":
        st.success("Aggregated weekly signal: **Bullish** — projecting upward bias for next period 🚀")
    elif final_signal == "Bearish":
        st.error("Aggregated weekly signal: **Bearish** — projecting downward bias for next period 📉")
    else:
        st.info("Aggregated weekly signal: **Neutral** — projecting flat bias for next period ⚖️")

    # compute drift
    if final_signal == "Bullish":
        drift = bull_drift_pct
    elif final_signal == "Bearish":
        drift = -abs(bear_drift_pct)
    else:
        drift = 0.0

    # generate predicted candles
    last_row = df_recent.iloc[-1]
    last_close = float(last_row["close"])
    last_timestamp = last_row["timestamp"]
    st.markdown(f"**Last observed close:** {last_close:.8f} at {last_timestamp}")
    pred_candles = synthesize_predicted_candles(start_date=last_timestamp, start_open=last_close, drift_per_day=drift, n_days=n_forecast_days)

    # show combined chart (actual last period + predicted)
    fig_both = go.Figure()
    fig_both.add_trace(go.Candlestick(
        x=df_recent["timestamp"],
        open=df_recent["open"],
        high=df_recent["high"],
        low=df_recent["low"],
        close=df_recent["close"],
        name="Actual (recent)"
    ))
    if not pred_candles.empty:
        fig_both.add_trace(go.Candlestick(
            x=pred_candles["timestamp"],
            open=pred_candles["open"],
            high=pred_candles["high"],
            low=pred_candles["low"],
            close=pred_candles["close"],
            name="Predicted",
            increasing_line_color='cyan',
            decreasing_line_color='orange'
        ))
    fig_both.update_layout(title="Actual (recent) + Predicted Candlesticks", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig_both, use_container_width=True)

    # Logging
    st.markdown("#### Logging predicted daily closes to ai_predictions_log.csv")
    if st.button("Log predicted closes", key="log_candles_button"):
        try:
            log_weekly_candlestick_predictions(pred_candles, asset_name="Bitcoin", method=method_name)
            st.success(f"Logged {len(pred_candles)} predicted rows into {AI_PRED_FILE}.")
        except Exception as e:
            st.error(f"Failed to log predictions: {e}")

    # predicted table and download
    if not pred_candles.empty:
        st.markdown("#### Predicted candles (table)")
        try:
            display_df = pred_candles.copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.dataframe(display_df)
            csv_bytes = pred_candles.to_csv(index=False).encode("utf-8")
            st.download_button("Download predicted candles CSV", data=csv_bytes, file_name="predicted_candles.csv", mime="text/csv")
        except Exception:
            st.info("Unable to show predicted candles table.")

    st.markdown("### Weekly Pattern Counts")
    st.json(weekly_patterns)
