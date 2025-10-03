# app.py
# Rewritten full application file with separate "Candlestick Predictions" menu and
# next-week synthetic candlestick generation based on last 7 days of Bitcoin OHLC.
#
# Features:
#  - Gold & Bitcoin dashboards
#  - AI Forecast dashboard
#  - New: Candlestick Predictions menu (last week -> predicted next week)
#  - Jobs & Real Estate Bot placeholders preserved
#  - Logging of predicted daily values into data/ai_predictions_log.csv
#
# Note: This file intentionally verbose and long (>600 lines) to satisfy user request.
# Author: ChatGPT (generated for user)
# Date: 2025-10-03
# ------------------------------------------------------------------------------

import os
import yaml
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

# Preserve original external imports but provide safe fallbacks
try:
    from jobs_app import jobs_dashboard
except Exception:
    def jobs_dashboard():
        st.warning("jobs_app not found — Jobs dashboard unavailable.")

try:
    from ai_predictor import predict_next_n, compare_predictions_vs_actuals
except Exception:
    def predict_next_n(asset_name="Bitcoin", n_steps=7):
        # fallback: no AI predictions available
        return pd.DataFrame()

    def compare_predictions_vs_actuals(*args, **kwargs):
        return {}

try:
    from real_estate_bot import real_estate_dashboard
except Exception:
    def real_estate_dashboard():
        st.warning("real_estate_bot not found — Real Estate Bot unavailable.")


# -------------------------------------------------------------------
# APP CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------------------------------------------------
# FILE PATHS AND DATA DIRECTORIES
# -------------------------------------------------------------------
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass  # downstream writes will warn if they fail

PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# -------------------------------------------------------------------
# UTILITIES: FILE I/O (robust)
# -------------------------------------------------------------------
def load_csv_safe(path: str, default_cols: List[str], parse_ts: bool = True) -> pd.DataFrame:
    """
    Load a CSV safely. If the file doesn't exist, returns an empty DataFrame with default_cols.
    Attempts to parse 'timestamp' to datetime when parse_ts=True.
    """
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if parse_ts and "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
                except Exception:
                    # fallback: leave as-is
                    pass
            return df
        except Exception as e:
            st.warning(f"Failed to read {path}: {e}")
            return pd.DataFrame(columns=default_cols)
    else:
        return pd.DataFrame(columns=default_cols)


def save_df_to_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save a DataFrame to CSV, with defensive error handling.
    """
    try:
        df.to_csv(path, index=index)
    except Exception as e:
        st.warning(f"Failed to save CSV to {path}: {e}")


def append_prediction_to_log(path: str, row: Dict[str, Any], dedupe_on: Optional[List[str]] = None):
    """
    Append a single-row dict to CSV at 'path'. If file doesn't exist, create it.
    If dedupe_on is provided, drop duplicates keeping the last timestamp per key.
    """
    try:
        new_row = pd.DataFrame([row])
        if os.path.exists(path):
            existing = pd.read_csv(path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        # Attempt to parse timestamp column if present
        if "timestamp" in combined.columns:
            try:
                combined["timestamp"] = pd.to_datetime(combined["timestamp"], infer_datetime_format=True)
            except Exception:
                pass

        if dedupe_on:
            # sort by timestamp so keep last
            if "timestamp" in combined.columns:
                combined.sort_values("timestamp", inplace=True)
            combined = combined.drop_duplicates(subset=dedupe_on, keep="last")

        combined.to_csv(path, index=False)
    except Exception as e:
        st.warning(f"Could not append to log {path}: {e}")


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price", "method"])

# normalize timestamp columns where present
def ensure_timestamp(df: pd.DataFrame, col: str = "timestamp"):
    if df is None or df.empty:
        return
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        except Exception:
            pass

ensure_timestamp(df_pred)
ensure_timestamp(df_actual)
ensure_timestamp(df_ai_pred_log)

# Load weights (assumptions)
if os.path.exists(WEIGHT_FILE):
    try:
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f) or {}
    except Exception:
        weights = {"gold": {}, "bitcoin": {}}
else:
    weights = {"gold": {}, "bitcoin": {}}


# -------------------------------------------------------------------
# UI THEMES & ICONS
# -------------------------------------------------------------------
INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

ASSET_THEMES = {
    "Gold": {
        "buy": "#FFF9C4", "sell": "#FFE0B2", "hold": "#E0E0E0",
        "target_bg": "#FFFDE7", "target_text": "black",
        "assumption_pos": "#FFD54F", "assumption_neg": "#FFAB91",
        "chart_actual": "#FBC02D", "chart_pred": "#FFCC80", "chart_ai": "#FF6F61"
    },
    "Bitcoin": {
        "buy": "#BBDEFB", "sell": "#FFCDD2", "hold": "#CFD8DC",
        "target_bg": "#E3F2FD", "target_text": "black",
        "assumption_pos": "#64B5F6", "assumption_neg": "#EF9A9A",
        "chart_actual": "#42A5F5", "chart_pred": "#81D4FA", "chart_ai": "#FF6F61"
    }
}


# -------------------------------------------------------------------
# PRESENTATIONAL HELPERS
# -------------------------------------------------------------------
def alert_badge(signal: str, asset_name: str) -> str:
    theme = ASSET_THEMES.get(asset_name, ASSET_THEMES["Gold"])
    color = theme["buy"] if signal == "Buy" else theme["sell"] if signal == "Sell" else theme["hold"]
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'


def target_price_card(price: Any, asset_name: str, horizon: str):
    theme = ASSET_THEMES.get(asset_name, ASSET_THEMES["Gold"])
    st.markdown(f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price} <br>⏳ Horizon: {horizon}
        </div>
        """, unsafe_allow_html=True)


def pretty_datetime(dt: Optional[datetime]) -> str:
    if dt is None or pd.isna(dt):
        return "-"
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except Exception:
            return str(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------------------------
# ASSUMPTIONS & EXPLANATION CARDS
# -------------------------------------------------------------------
def explanation_card(asset_df: pd.DataFrame, asset_name: str):
    st.subheader(f"📖 Explanation for {asset_name}")
    assumptions = {}
    if "assumptions" in asset_df.columns and not asset_df.empty:
        try:
            assumptions_str = asset_df["assumptions"].iloc[-1]
            assumptions = eval(assumptions_str) if isinstance(assumptions_str, str) else assumptions_str or {}
        except Exception:
            assumptions = {}

    if assumptions:
        main_indicator = max(assumptions.items(), key=lambda x: abs(x[1]))
        indicator, value = main_indicator
        direction = "rising 📈" if value > 0 else "falling 📉" if value < 0 else "stable ⚖️"
        explanation_text = f"The latest forecast for **{asset_name}** was mainly driven by **{indicator.replace('_',' ').title()}**, which is {direction} ({value:.2f})."
    else:
        explanation_text = f"No clear driver identified for {asset_name}."

    st.markdown(f"""
        <div style="background-color:#FDF6EC; padding:12px; border-radius:12px; 
        box-shadow:0px 1px 3px rgba(0,0,0,0.1); font-size:16px; line-height:1.6;">
            {explanation_text}
        </div>
    """, unsafe_allow_html=True)


def assumptions_card(asset_df: pd.DataFrame, asset_name: str):
    st.subheader(f"📖 Assumptions for {asset_name}")

    if "assumptions" in asset_df.columns and not asset_df.empty:
        try:
            assumptions_str = asset_df["assumptions"].iloc[-1]
        except Exception:
            assumptions_str = "{}"
    else:
        assumptions_str = "{}"

    if "target_horizon" in asset_df.columns and not asset_df.empty:
        try:
            target_horizon = asset_df["target_horizon"].iloc[-1]
        except Exception:
            target_horizon = "Days"
    else:
        target_horizon = "Days"

    try:
        assumptions = eval(assumptions_str)
    except Exception:
        assumptions = {}

    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return

    # determine main factor
    max_factor = max(assumptions.items(), key=lambda x: abs(x[1]))
    factor_name, factor_value = max_factor
    if factor_value > 0:
        explanation = f"Predictions for **{asset_name}** are mainly supported by positive trends in **{factor_name.replace('_',' ')}**."
    elif factor_value < 0:
        explanation = f"Predictions for **{asset_name}** are mainly pressured by negative trends in **{factor_name.replace('_',' ')}**."
    else:
        explanation = f"Predictions for **{asset_name}** show no dominant influencing factor."

    st.markdown(f"""
        <div style="background-color:#FDF6EC; padding:12px; border-radius:12px; 
                    box-shadow:0px 1px 3px rgba(0,0,0,0.1);">
            {explanation}<br>
            <i>Forecast horizon: {target_horizon}</i>
        </div>
    """, unsafe_allow_html=True)

    indicators = list(assumptions.keys())
    values = [assumptions[k] for k in indicators]
    icons = [INDICATOR_ICONS.get(k, "❔") for k in indicators]
    theme = ASSET_THEMES.get(asset_name, ASSET_THEMES["Gold"])
    colors = [theme["assumption_pos"] if v > 0 else theme["assumption_neg"] if v < 0 else theme["hold"] for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(x=[f"{icon} {ind}"], y=[val], marker_color=color, text=[f"{val:.2f}"], textposition='auto'))

    fig.update_layout(
        title=f"{asset_name} Assumptions & Target ({target_horizon})",
        yaxis_title="Weight / Impact",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# -------------------------------------------------------------------
def merge_actual_pred(asset_name: str, actual_col: str) -> pd.DataFrame:
    """
    Merge predictions file with actuals file to generate signal, trend, assumptions and target horizon.
    """
    global df_pred, df_actual, weights

    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    if "timestamp" in asset_pred.columns and "timestamp" in df_actual.columns:
        try:
            asset_pred["timestamp"] = pd.to_datetime(asset_pred["timestamp"], infer_datetime_format=True)
            df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"], infer_datetime_format=True)
        except Exception:
            pass

    if actual_col in df_actual.columns:
        try:
            asset_pred = pd.merge_asof(
                asset_pred.sort_values("timestamp"),
                df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("1D")
            )
        except Exception:
            asset_pred = pd.merge_asof(
                asset_pred.sort_values("timestamp"),
                df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
                on="timestamp",
                direction="backward"
            )
    else:
        asset_pred["actual"] = None

    asset_pred["predicted_price"] = pd.to_numeric(asset_pred.get("predicted_price", pd.Series(dtype=float)), errors="coerce")
    asset_pred["actual"] = pd.to_numeric(asset_pred.get("actual", pd.Series(dtype=float)), errors="coerce")

    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if pd.notna(row["predicted_price"]) and pd.notna(row["actual"]) and row["predicted_price"] > row["actual"]
        else ("Sell" if pd.notna(row["predicted_price"]) and pd.notna(row["actual"]) and row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    # Trend: last 3 predicted points monotonicity
    asset_pred["trend"] = ""
    try:
        if len(asset_pred) >= 3:
            last3 = asset_pred["predicted_price"].tail(3)
            if last3.is_monotonic_increasing:
                asset_pred["trend"] = "Bullish 📈"
            elif last3.is_monotonic_decreasing:
                asset_pred["trend"] = "Bearish 📉"
            else:
                asset_pred["trend"] = "Neutral ⚖️"
    except Exception:
        asset_pred["trend"] = "Neutral ⚖️"

    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))

    # Horizon based on volatility
    horizon = "Days"
    if "volatility" in asset_pred.columns and not asset_pred["volatility"].isna().all():
        try:
            avg_vol = asset_pred["volatility"].mean()
            if avg_vol < 0.02:
                horizon = "Years"
            elif avg_vol < 0.05:
                horizon = "Months"
            else:
                horizon = "Days"
        except Exception:
            horizon = "Days"

    asset_pred["target_horizon"] = horizon
    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"] == "Buy" else (row["predicted_price"] if row["signal"] == "Sell" else row["actual"]),
        axis=1
    )
    return asset_pred


gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")


# -------------------------------------------------------------------
# WHAT-IF SLIDERS
# -------------------------------------------------------------------
st.sidebar.header("🔧 What-If Scenario")
def get_default_assumption(df: pd.DataFrame, key: str, fallback: float):
    if df is None or df.empty:
        return fallback
    try:
        last_assumptions = eval(df["assumptions"].iloc[-1])
        return last_assumptions.get(key, fallback)
    except Exception:
        return fallback

if "inflation_adj" not in st.session_state:
    st.session_state.inflation_adj = get_default_assumption(gold_df, "inflation", 2.5)
if "usd_adj" not in st.session_state:
    st.session_state.usd_adj = get_default_assumption(gold_df, "usd_strength", 0.0)
if "oil_adj" not in st.session_state:
    st.session_state.oil_adj = get_default_assumption(gold_df, "energy_prices", 0.0)
if "vix_adj" not in st.session_state:
    st.session_state.vix_adj = get_default_assumption(gold_df, "tail_risk_event", 20.0)

def reset_sliders():
    st.session_state.inflation_adj = get_default_assumption(gold_df, "inflation", 2.5)
    st.session_state.usd_adj = get_default_assumption(gold_df, "usd_strength", 0.0)
    st.session_state.oil_adj = get_default_assumption(gold_df, "energy_prices", 0.0)
    st.session_state.vix_adj = get_default_assumption(gold_df, "tail_risk_event", 20.0)

inflation_adj = st.sidebar.slider("Inflation 💹 (%)", 0.0, 10.0, st.session_state.inflation_adj, 0.1, key="inflation_adj")
usd_adj = st.sidebar.slider("USD Strength 💵 (%)", -10.0, 10.0, st.session_state.usd_adj, 0.1, key="usd_adj")
oil_adj = st.sidebar.slider("Oil Price 🛢️ (%)", -50.0, 50.0, st.session_state.oil_adj, 0.1, key="oil_adj")
vix_adj = st.sidebar.slider("VIX / Volatility 🚨", 0.0, 100.0, st.session_state.vix_adj, 1.0, key="vix_adj")
st.sidebar.button("Reset to Predicted Values", on_click=reset_sliders)

def apply_what_if(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    adj = 1 + st.session_state.inflation_adj * 0.01 - st.session_state.usd_adj * 0.01 + st.session_state.oil_adj * 0.01 - st.session_state.vix_adj * 0.005
    df_copy = df.copy()
    if "predicted_price" in df_copy.columns:
        try:
            df_copy["predicted_price"] = df_copy["predicted_price"].astype(float) * adj
            df_copy["target_price"] = df_copy["predicted_price"]
        except Exception:
            pass
    return df_copy

gold_df_adj = apply_what_if(gold_df)
btc_df_adj = apply_what_if(btc_df)


def generate_summary(asset_df: pd.DataFrame, asset_name: str) -> str:
    if asset_df is None or asset_df.empty:
        return f"No data for {asset_name}"
    last_row = asset_df.iloc[-1]
    return f"**{asset_name} Market Summary:** Signal: {last_row.get('signal', 'N/A')} | Trend: {last_row.get('trend', 'N/A')} | Target Price: {last_row.get('target_price', 'N/A')}"


# -------------------------------------------------------------------
# CANDLESTICK PATTERN DETECTION (single-candle and weekly aggregation)
# -------------------------------------------------------------------
def detect_candle_patterns_on_series(df_ohlc: pd.DataFrame) -> List[Tuple[pd.Timestamp, List[str]]]:
    """
    Detect patterns for each candle in df_ohlc (requires at least 3 candles to detect multi-candle patterns).
    Returns list of tuples: (timestamp_of_candle, [patterns_detected_for_that_candle])
    """
    results: List[Tuple[pd.Timestamp, List[str]]] = []
    if df_ohlc is None or df_ohlc.shape[0] < 3:
        return results

    df_sorted = df_ohlc.sort_values("timestamp").reset_index(drop=True)
    n = df_sorted.shape[0]
    for i in range(2, n):  # start from index 2 to have previous two candles
        window = df_sorted.iloc[i-2:i+1].copy().reset_index(drop=True)  # 3-candle window
        ts = window.loc[2, "timestamp"]
        patterns = detect_patterns_in_3_candles(window)
        results.append((ts, patterns))
    return results


def detect_patterns_in_3_candles(window: pd.DataFrame) -> List[str]:
    """
    Given 3-row window DataFrame with columns [timestamp, open, high, low, close],
    return detected patterns for the last candle considering the previous two candles.
    Simple rules: Doji, Hammer, Shooting Star, Bullish Engulfing, Bearish Engulfing, Piercing, Dark Cloud.
    """
    patterns: List[str] = []
    if window is None or window.shape[0] != 3:
        return patterns

    # latest is index 2, prior 1 and 0
    try:
        c0 = window.loc[2]
        c1 = window.loc[1]
        c2 = window.loc[0]

        c0_open = float(c0["open"])
        c0_high = float(c0["high"])
        c0_low = float(c0["low"])
        c0_close = float(c0["close"])
        body0 = abs(c0_close - c0_open)
        range0 = (c0_high - c0_low) if (c0_high - c0_low) > 0 else 1.0
        upper_wick = c0_high - max(c0_open, c0_close)
        lower_wick = min(c0_open, c0_close) - c0_low

        # Doji
        if body0 < 0.15 * range0:
            patterns.append("Doji (indecision)")

        # Hammer
        if lower_wick > 2 * body0 and upper_wick < body0 and c0_close > c0_open:
            patterns.append("Hammer (bullish reversal)")

        # Shooting Star
        if upper_wick > 2 * body0 and lower_wick < body0 and c0_close < c0_open:
            patterns.append("Shooting Star (bearish reversal)")

        # Bullish Engulfing
        if (float(c1["close"]) < float(c1["open"])) and (c0_close > c0_open) and (c0_close > float(c1["open"])) and (c0_open < float(c1["close"])):
            patterns.append("Bullish Engulfing")

        # Bearish Engulfing
        if (float(c1["close"]) > float(c1["open"])) and (c0_close < c0_open) and (c0_open > float(c1["close"])) and (c0_close < float(c1["open"])):
            patterns.append("Bearish Engulfing")

        # Piercing Line-ish (bullish)
        if (float(c1["close"]) < float(c1["open"])) and (c0_close > c0_open) and (c0_close > (float(c1["open"]) + float(c1["close"]))/2):
            patterns.append("Piercing Line-ish (bullish)")

        # Dark Cloud Cover-ish (bearish)
        if (float(c1["close"]) > float(c1["open"])) and (c0_close < c0_open) and (c0_close < (float(c1["open"]) + float(c1["close"]))/2):
            patterns.append("Dark Cloud Cover-ish (bearish)")

    except Exception:
        # on any error return patterns collected so far
        pass

    # dedupe while preserving order
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def aggregate_weekly_patterns(df_ohlc: pd.DataFrame, lookback_days: int = 7) -> Dict[str, int]:
    """
    Analyze the last `lookback_days` days of df_ohlc and return aggregated counts of bullish/bearish/neutral patterns.
    Returns dict: {'bullish': int, 'bearish': int, 'neutral': int, 'patterns_list': [..]}
    """
    result = {"bullish": 0, "bearish": 0, "neutral": 0, "patterns": []}
    if df_ohlc is None or df_ohlc.empty:
        return result

    # filter last lookback_days by timestamp
    try:
        df_sorted = df_ohlc.sort_values("timestamp")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent = df_sorted[df_sorted["timestamp"] >= cutoff]
        if recent.shape[0] < 3:
            # not enough data in the strict last N days; fallback to last N rows
            recent = df_sorted.tail(lookback_days)
    except Exception:
        recent = df_ohlc.tail(lookback_days)

    detections = detect_candle_patterns_on_series(recent)
    # detections: list of (timestamp, [patterns])
    for ts, pats in detections:
        for p in pats:
            lower = p.lower()
            result["patterns"].append(p)
            if any(k in lower for k in ["bull", "hammer", "piercing", "piercing line"]):
                result["bullish"] += 1
            elif any(k in lower for k in ["bear", "shooting", "dark cloud", "star"]):
                result["bearish"] += 1
            else:
                result["neutral"] += 1
    return result


def decide_weekly_signal(agg: Dict[str, int]) -> str:
    """
    Turn aggregated counts into a final weekly signal: Bullish, Bearish, Neutral.
    Conservative rule:
      - if bullish_count > bearish_count by >=1 -> Bullish
      - if bearish_count > bullish_count by >=1 -> Bearish
      - else Neutral
    """
    b = agg.get("bullish", 0)
    r = agg.get("bearish", 0)
    if b > r:
        return "Bullish"
    elif r > b:
        return "Bearish"
    else:
        return "Neutral"


# -------------------------------------------------------------------
# SYNTHETIC CANDLE GENERATION (for predicted days)
# -------------------------------------------------------------------
def synthesize_predicted_candles(start_date: pd.Timestamp, start_open: float, drift_per_day: float, n_days: int = 7) -> pd.DataFrame:
    """
    Generate n_days synthetic OHLC candles starting from start_date (next day).
    - start_open: the open for the first predicted day (we use last close)
    - drift_per_day: fractional drift to apply to close each day (e.g., 0.01 for +1%).
    Returns DataFrame with columns: timestamp, open, high, low, close
    Deterministic generation: high = max(open, close) * (1 + 0.002), low = min(open, close) * (1 - 0.002)
    """
    rows = []
    prev_close = start_open  # we feed last close as the base open for first predicted day
    current_date = pd.to_datetime(start_date)
    # Predictions will be made for next n_days: day1 = next calendar day after start_date
    for i in range(1, n_days + 1):
        pred_date = (current_date + pd.Timedelta(days=i)).normalize()
        open_price = prev_close
        close_price = open_price * (1.0 + drift_per_day)
        high_price = max(open_price, close_price) * 1.002  # small wiggle
        low_price = min(open_price, close_price) * 0.998
        # store
        rows.append({
            "timestamp": pred_date,
            "open": round(open_price, 8),
            "high": round(high_price, 8),
            "low": round(low_price, 8),
            "close": round(close_price, 8)
        })
        # next day's open is this day's close (simple assumption)
        prev_close = close_price
    df = pd.DataFrame(rows)
    return df


# -------------------------------------------------------------------
# LOGGING PREDICTIONS (write each predicted day's close to ai_predictions_log.csv)
# -------------------------------------------------------------------
def log_weekly_candlestick_predictions(pred_df: pd.DataFrame, asset_name: str = "Bitcoin", method: str = "candlestick"):
    """
    For each predicted day in pred_df (with 'timestamp' and 'close'), append an entry to ai_predictions_log.csv.
    Each row will contain: timestamp (ISO), asset, predicted_price (close), method
    Dedupe will keep last per (timestamp, asset, method).
    """
    if pred_df is None or pred_df.empty:
        return
    for _, row in pred_df.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts_iso = ts.isoformat()
        elif isinstance(ts, datetime):
            ts_iso = ts.isoformat()
        else:
            ts_iso = str(ts)
        r = {"timestamp": ts_iso, "asset": asset_name, "predicted_price": float(row["close"]), "method": method}
        append_prediction_to_log(AI_PRED_FILE, r, dedupe_on=["timestamp", "asset", "method"])


# -------------------------------------------------------------------
# MAIN MENU: add "Candlestick Predictions"
# -------------------------------------------------------------------
menu = st.sidebar.radio(
    "📊 Choose Dashboard",
    ["Gold & Bitcoin", "AI Forecast", "Candlestick Predictions", "Jobs", "Real Estate Bot"]
)


# small header helper
def render_header(title: str, subtitle: Optional[str] = None):
    st.title(title)
    if subtitle:
        st.markdown(f"_{subtitle}_")


# -------------------------------------------------------------------
# GOLD & BITCOIN DASHBOARD (original behavior preserved)
# -------------------------------------------------------------------
if menu == "Gold & Bitcoin":
    render_header("🌸 Gold & Bitcoin Market Dashboard (Pastel Theme)", "Combined view of predictions, actuals and AI forecasts")
    col1, col2 = st.columns(2)

    for col, df, name, actual_col in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"], ["gold_actual", "bitcoin_actual"]):
        with col:
            st.subheader(name)
            if df is not None and not df.empty:
                last_signal = df.get("signal", pd.Series(["Hold"])).iloc[-1]
                st.markdown(alert_badge(last_signal, name), unsafe_allow_html=True)
                last_trend = df.get("trend", pd.Series(["Neutral ⚖️"])).iloc[-1]
                st.markdown(f"**Market Trend:** {last_trend}")
                st.markdown(generate_summary(df, name))
                target_price_card(df["target_price"].iloc[-1] if "target_price" in df.columns else "N/A", name, df["target_horizon"].iloc[-1] if "target_horizon" in df.columns else "Days")
                try:
                    explanation_card(df, name)
                    assumptions_card(df, name)
                except Exception:
                    st.info("Error showing explanation or assumptions.")

                n_steps = 7
                try:
                    df_ai = predict_next_n(asset_name=name, n_steps=n_steps)
                except Exception:
                    df_ai = pd.DataFrame()

                theme = ASSET_THEMES.get(name, ASSET_THEMES["Gold"])
                fig = go.Figure()

                if "timestamp" in df.columns and "actual" in df.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers", name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                    except Exception:
                        pass

                if "timestamp" in df.columns and "predicted_price" in df.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers", name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                    except Exception:
                        pass

                if not df_ai.empty and "timestamp" in df_ai.columns and "predicted_price" in df_ai.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"], mode="lines+markers", name="AI Forecast", line=dict(color=theme["chart_ai"], dash="dot")))
                    except Exception:
                        pass

                fig.update_layout(title=f"{name} Prices: Actual + Predicted + AI Forecast", xaxis_title="Date", yaxis_title="Price", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")


# -------------------------------------------------------------------
# AI FORECAST DASHBOARD (unchanged semantics)
# -------------------------------------------------------------------
elif menu == "AI Forecast":
    render_header("🤖 AI Forecast Dashboard", "AI-predicted prices and candlestick-based pattern predictions (Bitcoin)")

    n_steps = st.sidebar.number_input("Forecast next days", min_value=1, max_value=30, value=7)
    col_left, col_right = st.columns(2)

    # GOLD (left)
    with col_left:
        asset = "Gold"
        actual_col = "gold_actual"
        st.subheader(asset)
        st.markdown("**Historical AI Predictions vs Actual**")
        df_hist_actual = pd.DataFrame()
        if actual_col in df_actual.columns:
            df_hist_actual = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}).dropna()
        df_hist_pred = pd.DataFrame()
        if "asset" in df_ai_pred_log.columns:
            df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == asset][["timestamp", "predicted_price"]].dropna()

        if not df_hist_actual.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_hist_actual["timestamp"], y=df_hist_actual["actual"], mode="lines+markers", name="Actual Price", line=dict(width=2)))
            fig_cmp.add_trace(go.Scatter(x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"], mode="lines+markers", name="Predicted Price", line=dict(dash="dot")))
            fig_cmp.update_layout(title=f"{asset} – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white", height=350)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info(f"No historical data available for {asset} or no AI predictions logged.")

        st.markdown("**Future AI Forecast**")
        try:
            df_ai_future = predict_next_n(asset_name=asset, n_steps=n_steps)
        except Exception:
            df_ai_future = pd.DataFrame()

        if not df_ai_future.empty:
            fig = go.Figure()
            if not df_hist_actual.empty:
                fig.add_trace(go.Scatter(x=df_hist_actual["timestamp"], y=df_hist_actual["actual"], mode="lines+markers", name="Actual", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=df_ai_future["timestamp"], y=df_ai_future["predicted_price"], mode="lines+markers", name="AI Predicted", line=dict(dash="dash")))
            fig.update_layout(title=f"{asset} AI Forecast vs Actual", xaxis_title="Date", yaxis_title="Price", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No AI forecast available for {asset}.")

    # BITCOIN (right)
    with col_right:
        asset = "Bitcoin"
        st.subheader(asset)

        # Prepare OHLC
        ohlc_cols_present = all(c in df_actual.columns for c in ["bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
        if ohlc_cols_present:
            df_ohlc = df_actual[["timestamp", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]].dropna().copy()
            df_ohlc = df_ohlc.rename(columns={"bitcoin_open": "open", "bitcoin_high": "high", "bitcoin_low": "low", "bitcoin_close": "close"})
            try:
                df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], infer_datetime_format=True)
            except Exception:
                pass
            df_ohlc = df_ohlc.sort_values("timestamp")
        else:
            if "bitcoin_actual" in df_actual.columns:
                d = df_actual[["timestamp", "bitcoin_actual"]].dropna().copy().sort_values("timestamp")
                d["open"] = d["bitcoin_actual"].shift(1).fillna(d["bitcoin_actual"])
                d["close"] = d["bitcoin_actual"]
                d["high"] = d[["open", "close"]].max(axis=1) * 1.002
                d["low"] = d[["open", "close"]].min(axis=1) * 0.998
                df_ohlc = d.rename(columns={"timestamp": "timestamp"})
            else:
                df_ohlc = pd.DataFrame()

        st.markdown("**Candlestick Chart (Bitcoin Real-Time Index)**")
        if not df_ohlc.empty:
            fig_candle = go.Figure(data=[go.Candlestick(x=df_ohlc["timestamp"], open=df_ohlc["open"], high=df_ohlc["high"], low=df_ohlc["low"], close=df_ohlc["close"], name="BTC")])
            fig_candle.update_layout(title="Bitcoin Candlesticks", xaxis_rangeslider_visible=False, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.info("OHLC data not available for Bitcoin; cannot render candlestick chart.")

        st.markdown("**Candlestick Pattern Detection & Rule-based Prediction**")
        patterns = detect_patterns_in_3_candles(df_ohlc.tail(3)) if not df_ohlc.empty and df_ohlc.shape[0] >= 3 else []
        if patterns:
            st.markdown(f"**Detected patterns (latest):** {', '.join(patterns)}")
            signal = "Bullish" if any("bull" in p.lower() or "hammer" in p.lower() for p in patterns) else ("Bearish" if any("bear" in p.lower() or "shooting" in p.lower() for p in patterns) else "Neutral")
            if signal == "Bullish":
                st.success("Pattern-based prediction: **Bullish** — short-term upward bias 🚀")
            elif signal == "Bearish":
                st.error("Pattern-based prediction: **Bearish** — short-term downward bias 📉")
            else:
                st.info("Pattern-based prediction: **Neutral** — no clear short-term signal ⚖️")
        else:
            st.info("No clear candlestick pattern detected (insufficient data or none matched).")

        st.markdown("**Historical AI Predictions vs Actual**")
        df_hist_actual_close = pd.DataFrame()
        if "bitcoin_close" in df_actual.columns:
            df_hist_actual_close = df_actual[["timestamp", "bitcoin_close"]].rename(columns={"bitcoin_close": "actual"}).dropna()
        elif "bitcoin_actual" in df_actual.columns:
            df_hist_actual_close = df_actual[["timestamp", "bitcoin_actual"]].rename(columns={"bitcoin_actual": "actual"}).dropna()
        df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == "Bitcoin"][["timestamp", "predicted_price"]].dropna() if "asset" in df_ai_pred_log.columns else pd.DataFrame()

        if not df_hist_actual_close.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_hist_actual_close["timestamp"], y=df_hist_actual_close["actual"], mode="lines+markers", name="Actual Price", line=dict(width=2)))
            fig_cmp.add_trace(go.Scatter(x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"], mode="lines+markers", name="Predicted Price", line=dict(dash="dot")))
            fig_cmp.update_layout(title="Bitcoin – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white", height=350)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("No historical comparison available (missing actual OHLC/close or AI predictions).")

        st.markdown("**Future AI Forecast (overlay)**")
        try:
            df_ai_future = predict_next_n(asset_name="Bitcoin", n_steps=n_steps)
        except Exception:
            df_ai_future = pd.DataFrame()

        if not df_ai_future.empty:
            fig_overlay = go.Figure()
            if not df_ohlc.empty:
                fig_overlay.add_trace(go.Candlestick(x=df_ohlc["timestamp"], open=df_ohlc["open"], high=df_ohlc["high"], low=df_ohlc["low"], close=df_ohlc["close"], name="BTC"))
                last_date = df_ohlc["timestamp"].max()
                try:
                    last_close = float(df_ohlc.loc[df_ohlc["timestamp"] == last_date, "close"].iloc[0])
                except Exception:
                    last_close = float(df_ai_future["predicted_price"].iloc[0])
            else:
                if not df_hist_actual_close.empty:
                    fig_overlay.add_trace(go.Scatter(x=df_hist_actual_close["timestamp"], y=df_hist_actual_close["actual"], mode="lines+markers", name="Actual", line=dict(width=2)))
                    last_date = df_hist_actual_close["timestamp"].max()
                    last_close = float(df_hist_actual_close["actual"].iloc[-1])
                else:
                    last_date = pd.Timestamp.now()
                    last_close = float(df_ai_future["predicted_price"].iloc[0])

            x_join = [last_date] + list(pd.to_datetime(df_ai_future["timestamp"]))
            y_join = [last_close] + list(df_ai_future["predicted_price"])
            fig_overlay.add_trace(go.Scatter(x=x_join, y=y_join, mode="lines+markers", name="AI Forecast", line=dict(color=ASSET_THEMES["Bitcoin"]["chart_ai"], dash="dot")))
            fig_overlay.update_layout(title="Bitcoin Candlesticks + AI Forecast Overlay", xaxis_rangeslider_visible=False, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=500)
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.info("No AI forecast available for Bitcoin.")


# -------------------------------------------------------------------
# CANDLESTICK PREDICTIONS MENU (NEW)
# -------------------------------------------------------------------
elif menu == "Candlestick Predictions":
    render_header("🕯️ Candlestick Predictions (Bitcoin)", "Analyze last 7 days of candles → predict the next 7 days as synthetic candlesticks")

    # Step 1: Prepare OHLC for Bitcoin (prefer explicit ohlc columns)
    ohlc_cols_present = all(c in df_actual.columns for c in ["bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
    if ohlc_cols_present:
        df_ohlc_all = df_actual[["timestamp", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]].dropna().copy()
        df_ohlc_all = df_ohlc_all.rename(columns={
            "bitcoin_open": "open", "bitcoin_high": "high", "bitcoin_low": "low", "bitcoin_close": "close"
        })
    else:
        # fallback to synthetic ohlc from bitcoin_actual if available
        if "bitcoin_actual" in df_actual.columns:
            tmp = df_actual[["timestamp", "bitcoin_actual"]].dropna().copy().sort_values("timestamp")
            tmp["open"] = tmp["bitcoin_actual"].shift(1).fillna(tmp["bitcoin_actual"])
            tmp["close"] = tmp["bitcoin_actual"]
            tmp["high"] = tmp[["open", "close"]].max(axis=1) * 1.002
            tmp["low"] = tmp[["open", "close"]].min(axis=1) * 0.998
            df_ohlc_all = tmp.rename(columns={"timestamp": "timestamp"})
        else:
            df_ohlc_all = pd.DataFrame()

    # Ensure timestamps are datetimes
    if not df_ohlc_all.empty:
        try:
            df_ohlc_all["timestamp"] = pd.to_datetime(df_ohlc_all["timestamp"], infer_datetime_format=True)
        except Exception:
            pass
        df_ohlc_all = df_ohlc_all.sort_values("timestamp").reset_index(drop=True)

    # UI: parameters
    st.markdown("#### Parameters")
    lookback_days = st.number_input("Lookback (days) for pattern detection (week length)", min_value=3, max_value=30, value=7, step=1)
    n_forecast_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=14, value=7, step=1)
    bull_drift_pct = st.number_input("Daily bullish drift (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
    bear_drift_pct = st.number_input("Daily bearish drift (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
    method_name = st.text_input("Logging method name (for ai_predictions_log)", value="candlestick")

    # Step 2: Extract last lookback_days candles
    if df_ohlc_all.empty:
        st.info("No OHLC data available for Bitcoin; cannot run candlestick predictions.")
    else:
        # define lookback period by time or by rows; prefer last lookback_days by timestamp
        try:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
            df_recent = df_ohlc_all[df_ohlc_all["timestamp"] >= cutoff].copy()
            if df_recent.shape[0] < 3:
                # fallback: last N rows
                df_recent = df_ohlc_all.tail(lookback_days).copy()
        except Exception:
            df_recent = df_ohlc_all.tail(lookback_days).copy()

        st.markdown(f"**Using {len(df_recent)} candles for pattern detection (last {lookback_days} days/rows)**")

        if df_recent.empty or df_recent.shape[0] < 3:
            st.info("Not enough recent candles to detect patterns (need at least 3).")
        else:
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

            # Step 3: detect weekly patterns (aggregate)
            agg = aggregate_weekly_patterns(df_recent, lookback_days=lookback_days)
            st.markdown("#### Aggregated pattern counts (last period)")
            st.write({
                "bullish_patterns": agg.get("bullish", 0),
                "bearish_patterns": agg.get("bearish", 0),
                "neutral_patterns": agg.get("neutral", 0),
                "patterns_found": agg.get("patterns", [])
            })

            final_signal = decide_weekly_signal(agg)
            if final_signal == "Bullish":
                st.success("Aggregated weekly signal: **Bullish** — projecting upward bias for next week 🚀")
            elif final_signal == "Bearish":
                st.error("Aggregated weekly signal: **Bearish** — projecting downward bias for next week 📉")
            else:
                st.info("Aggregated weekly signal: **Neutral** — projecting flat bias for next week ⚖️")

            # Step 4: Compute drift per day based on final_signal
            if final_signal == "Bullish":
                drift = bull_drift_pct
            elif final_signal == "Bearish":
                drift = -abs(bear_drift_pct)
            else:
                drift = 0.0

            # Step 5: Generate synthetic predicted candles for next n days
            last_row = df_recent.iloc[-1]
            last_close = float(last_row["close"])
            last_timestamp = last_row["timestamp"]
            st.markdown(f"**Last observed close:** {last_close:.8f} at {pretty_datetime(last_timestamp)}")
            pred_candles = synthesize_predicted_candles(start_date=last_timestamp, start_open=last_close, drift_per_day=drift, n_days=n_forecast_days)

            # show predicted candles as candlestick traces
            # For clarity, create predicted candles as separate trace with slightly different color/opacity
            fig_both = go.Figure()
            # actual last week candlesticks
            fig_both.add_trace(go.Candlestick(
                x=df_recent["timestamp"],
                open=df_recent["open"], high=df_recent["high"], low=df_recent["low"], close=df_recent["close"],
                name="Actual (last week)"
            ))
            # predicted candlesticks (synthetic) - add as another candlestick trace
            # We will set increasing/decreasing line colors to indicate direction
            fig_both.add_trace(go.Candlestick(
                x=pred_candles["timestamp"],
                open=pred_candles["open"],
                high=pred_candles["high"],
                low=pred_candles["low"],
                close=pred_candles["close"],
                name="Predicted (next week)",
                increasing_line_color='cyan',
                decreasing_line_color='orange',
                increasing_fillcolor='rgba(0,255,255,0.1)',
                decreasing_fillcolor='rgba(255,165,0,0.1)'
            ))
            fig_both.update_layout(title="Actual (last week) + Predicted (next week) Candlesticks", xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig_both, use_container_width=True)

            # Step 6: Logging: write each predicted day's close to ai_predictions_log.csv
            st.markdown("#### Logging predicted daily closes to ai_predictions_log.csv")
            if st.button("Log predicted next-week closes", key="log_candles_button"):
                try:
                    log_weekly_candlestick_predictions(pred_candles, asset_name="Bitcoin", method=method_name)
                    st.success(f"Logged {len(pred_candles)} predicted days to {AI_PRED_FILE} with method='{method_name}' and asset='Bitcoin'.")
                    # Refresh in-memory df_ai_pred_log
                    try:
                        refreshed = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price", "method"])
                        ensure_timestamp(refreshed)
                        df_ai_pred_log = refreshed
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Failed to log predictions: {e}")

            # show predicted table for inspection
            st.markdown("#### Predicted candles (table)")
            st.dataframe(pred_candles.assign(timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%d")))

            # Optionally offer download of predicted candles
            csv_bytes = pred_candles.to_csv(index=False).encode("utf-8")
            st.download_button("Download predicted candles CSV", data=csv_bytes, file_name="predicted_candles_next_week.csv", mime="text/csv")


# -------------------------------------------------------------------
# JOBS MENU
# -------------------------------------------------------------------
elif menu == "Jobs":
    try:
        jobs_dashboard()
    except Exception as e:
        st.error(f"Error running jobs_dashboard(): {e}")


# -------------------------------------------------------------------
# REAL ESTATE BOT MENU
# -------------------------------------------------------------------
elif menu == "Real Estate Bot":
    try:
        real_estate_dashboard()
    except Exception as e:
        st.error(f"Error running real_estate_dashboard(): {e}")


# -------------------------------------------------------------------
# FOOTER: diagnostics and downloads
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("### Diagnostics & Data Files")
colA, colB, colC = st.columns([1, 1, 2])

with colA:
    st.markdown("**Data file paths**")
    st.write(f"PREDICTION_FILE = `{PREDICTION_FILE}`")
    st.write(f"AI_PRED_FILE = `{AI_PRED_FILE}`")
    st.write(f"ACTUAL_FILE = `{ACTUAL_FILE}`")

with colB:
    if st.button("Reload CSVs (refresh)"):
        try:
            df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
            df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
            df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price", "method"])
            ensure_timestamp(df_pred)
            ensure_timestamp(df_actual)
            ensure_timestamp(df_ai_pred_log)
            st.success("CSVs reloaded.")
        except Exception as e:
            st.warning(f"Reload failed: {e}")

with colC:
    st.markdown("**Download AI predictions (last 90 days)**")
    try:
        if not df_ai_pred_log.empty:
            tmp = df_ai_pred_log.copy()
            try:
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                tmp = tmp[tmp["timestamp"] >= cutoff]
            except Exception:
                pass
            csv_bytes = tmp.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (AI predictions 90d)", data=csv_bytes, file_name="ai_predictions_recent.csv", mime="text/csv")
        else:
            st.info("No AI predictions logged yet.")
    except Exception:
        st.info("No AI predictions available for download.")


# -------------------------------------------------------------------
# END NOTES
# -------------------------------------------------------------------
st.markdown("""
---
**Notes & Disclaimers**
- Candlestick-based predictions are rule-based heuristics for demonstration and educational purposes only. They are **not** financial advice.
- Synthetic predicted candles are simplistic: open = prior close, close = open * (1 + drift), high/low are small offsets. You can replace with any model.
- Logging writes to `data/ai_predictions_log.csv`. Ensure the Streamlit process can write to `data/`.
- If external modules (`jobs_app`, `ai_predictor`, `real_estate_bot`) are missing, the app will show placeholders but the rest of the UI will function.
""")
