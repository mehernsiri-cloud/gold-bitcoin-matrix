# app.py
# Rewritten full application file (expanded & documented)
# Features:
#  - Gold & Bitcoin dashboards (unchanged semantics)
#  - AI Forecast dashboard with Bitcoin candlestick chart + pattern detection
#  - Next-day Bitcoin candlestick rule-based prediction (only Bitcoin)
#  - Logging of candlestick next-day prediction into data/ai_predictions_log.csv
#  - Robust CSV load/save, defensive parsing, helpful UI components
#
# Author: ChatGPT (rewritten at user's request)
# Date: 2025-10-03 (app code is generic and not dependent on this date)
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any

# Keep these imports from your original app to preserve modular functionality.
# They may raise at runtime if missing; we catch errors around their usage.
try:
    from jobs_app import jobs_dashboard
except Exception as e:
    # Provide fallback function so the app doesn't crash if jobs_app is missing
    def jobs_dashboard():
        st.warning("jobs_app module unavailable. Install or restore jobs_app.py to use Jobs dashboard.")

try:
    from ai_predictor import predict_next_n, compare_predictions_vs_actuals
except Exception as e:
    # Fallback stand-ins so app is still runnable.
    def predict_next_n(asset_name="Bitcoin", n_steps=7):
        # A trivial fallback: returns empty DataFrame
        return pd.DataFrame()

    def compare_predictions_vs_actuals(*args, **kwargs):
        return {}

try:
    from real_estate_bot import real_estate_dashboard
except Exception as e:
    def real_estate_dashboard():
        st.warning("real_estate_bot module unavailable. Install or restore real_estate_bot.py to use Real Estate Bot dashboard.")


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide", initial_sidebar_state="expanded")


# ------------------------------
# FILES & PATHS
# ------------------------------
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        # If we cannot create, downstream saving will fail gracefully
        pass

PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# UTILITIES: CSV / YAML load save
# ------------------------------
def load_csv_safe(path: str, default_cols: List[str], parse_ts: bool = True) -> pd.DataFrame:
    """
    Load CSV if present. Return empty DataFrame with default_cols if not present.
    Attempts to parse a 'timestamp' column to datetime when parse_ts True.
    """
    if os.path.exists(path):
        try:
            if parse_ts:
                # try robust parsing with dayfirst=True fallback
                df = pd.read_csv(path)
                if "timestamp" in df.columns:
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, dayfirst=True)
                    except Exception:
                        try:
                            df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
                        except Exception:
                            # leave as-is
                            pass
                return df
            else:
                return pd.read_csv(path)
        except Exception:
            # if CSV read fails, return empty DataFrame with default columns
            return pd.DataFrame(columns=default_cols)
    else:
        return pd.DataFrame(columns=default_cols)


def save_df_to_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save DataFrame to CSV. Wrap in try/except to avoid crashing the app if write fails.
    """
    try:
        df.to_csv(path, index=index)
    except Exception as e:
        st.warning(f"Failed to write to {path}: {e}")


def append_prediction_to_log(path: str, row: Dict[str, Any], dedupe_on: Optional[List[str]] = None):
    """
    Append a single-row dict to CSV at 'path'. If file doesn't exist, create new file with header.
    If dedupe_on is provided, drop duplicates based on those columns keeping the latest timestamp.
    """
    try:
        new_row = pd.DataFrame([row])
        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            df = pd.concat([df_existing, new_row], ignore_index=True)
        else:
            df = new_row

        # ensure timestamp is datetime
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
            except Exception:
                pass

        if dedupe_on:
            # keep the latest timestamp per dedupe subset
            if "timestamp" in df.columns:
                df.sort_values("timestamp", ascending=True, inplace=True)
            df = df.drop_duplicates(subset=dedupe_on, keep="last")

        df.to_csv(path, index=False)
    except Exception as e:
        st.warning(f"Could not append to prediction log {path}: {e}")


# ------------------------------
# LOAD DATA
# ------------------------------
# Columns defaults chosen to match your original code expectations
df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price"])

# Normalize timestamp columns where present
def ensure_timestamp(df: pd.DataFrame, col: str = "timestamp"):
    if df is None or df.empty:
        return
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        except Exception:
            # keep as-is
            pass

ensure_timestamp(df_pred)
ensure_timestamp(df_actual)
ensure_timestamp(df_ai_pred_log)


# ------------------------------
# LOAD WEIGHTS (assumptions)
# ------------------------------
if os.path.exists(WEIGHT_FILE):
    try:
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f) or {}
    except Exception:
        weights = {"gold": {}, "bitcoin": {}}
else:
    weights = {"gold": {}, "bitcoin": {}}


# ------------------------------
# UI THEMES & ICONS
# ------------------------------
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


# ------------------------------
# SMALL PRESENTATIONAL HELPERS
# ------------------------------
def alert_badge(signal: str, asset_name: str) -> str:
    """
    Lightweight HTML badge for Buy / Sell / Hold
    """
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


# ------------------------------
# ASSUMPTIONS / EXPLANATION CARDS
# ------------------------------
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


# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_actual_pred(asset_name: str, actual_col: str) -> pd.DataFrame:
    """
    Merge predictions log (df_pred) with actuals (df_actual) using merge_asof on timestamp.
    Also calculates signals, trend, and attaches assumptions & horizon.
    """
    global df_pred, df_actual, weights

    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    # ensure timestamps are datetimes
    if "timestamp" in asset_pred.columns and "timestamp" in df_actual.columns:
        try:
            asset_pred["timestamp"] = pd.to_datetime(asset_pred["timestamp"], infer_datetime_format=True)
            df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"], infer_datetime_format=True)
        except Exception:
            pass

    if actual_col in df_actual.columns:
        try:
            # merge_asof to get nearest past actual within tolerance 1 day, fallback without tolerance
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

    # Signal: simple comparison predicted vs actual
    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if pd.notna(row["predicted_price"]) and pd.notna(row["actual"]) and row["predicted_price"] > row["actual"]
        else ("Sell" if pd.notna(row["predicted_price"]) and pd.notna(row["actual"]) and row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    # Trend: check last 3 predicted points
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

    # Attach assumptions as string
    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))

    # Determine horizon based on volatility if present
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


# Precompute merged frames for UI usage
gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")


# ------------------------------
# WHAT-IF SLIDERS AND SESSION STATE
# ------------------------------
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
    """
    Apply adjustments to predicted_price column based on session sliders.
    Returns new DataFrame (copy).
    """
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


# ------------------------------
# CANDLESTICK PATTERN DETECTION (BITCOIN ONLY)
# ------------------------------
def detect_candle_patterns(df_ohlc: pd.DataFrame) -> List[str]:
    """
    Inspect recent candles and return list of detected patterns.
    Works best if df_ohlc has columns: timestamp, open, high, low, close.
    We use simple, well-known rules for Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing.
    This is intentionally conservative and rule-based (not ML).
    """
    patterns: List[str] = []
    if df_ohlc is None or df_ohlc.shape[0] < 3:
        return patterns

    # Use last 3 candles for small pattern context
    recent = df_ohlc.tail(3).copy().reset_index(drop=True)
    # indexes: 0 (oldest), 1, 2 (latest)
    c0 = recent.iloc[-1]  # latest
    c1 = recent.iloc[-2]
    c2 = recent.iloc[-3]

    # Defensive conversions to floats
    try:
        c0_open = float(c0["open"])
        c0_close = float(c0["close"])
        c0_high = float(c0["high"])
        c0_low = float(c0["low"])
    except Exception:
        return patterns

    body0 = abs(c0_close - c0_open)
    range0 = (c0_high - c0_low) if (c0_high - c0_low) > 0 else 1.0
    upper_wick = c0_high - max(c0_open, c0_close)
    lower_wick = min(c0_open, c0_close) - c0_low

    # Doji: very small body relative to range (indecision)
    try:
        if body0 < 0.15 * range0:
            patterns.append("Doji (indecision)")
    except Exception:
        pass

    # Hammer: long lower wick, small body, at bottom of down move (we check latest candle only)
    try:
        if lower_wick > 2 * body0 and upper_wick < body0 and c0_close > c0_open:
            patterns.append("Hammer (bullish reversal)")
    except Exception:
        pass

    # Shooting Star: long upper wick, small body, near top of up move
    try:
        if upper_wick > 2 * body0 and lower_wick < body0 and c0_close < c0_open:
            patterns.append("Shooting Star (bearish reversal)")
    except Exception:
        pass

    # Bullish Engulfing: prior candle bearish, latest bullish and engulfs prior body
    try:
        if (float(c1["close"]) < float(c1["open"])) and (c0_close > c0_open) and (c0_close > float(c1["open"])) and (c0_open < float(c1["close"])):
            patterns.append("Bullish Engulfing")
    except Exception:
        pass

    # Bearish Engulfing: prior candle bullish, latest bearish and engulfs prior body
    try:
        if (float(c1["close"]) > float(c1["open"])) and (c0_close < c0_open) and (c0_open > float(c1["close"])) and (c0_close < float(c1["open"])):
            patterns.append("Bearish Engulfing")
    except Exception:
        pass

    # Optional: add two-candle patterns like Piercing Line or Dark Cloud Cover
    try:
        # Piercing Line (bullish two-candle): prior bearish, latest bullish and closes into prior body > 50%
        if (float(c1["close"]) < float(c1["open"])) and (c0_close > c0_open) and (c0_close > float(c1["open"])) and (c0_open < float(c1["close"])):
            # Already covered by Bullish Engulfing (above) but we can annotate
            if "Bullish Engulfing" not in patterns:
                patterns.append("Piercing Line-ish (bullish)")
    except Exception:
        pass

    try:
        # Dark Cloud Cover (bearish two-candle)
        if (float(c1["close"]) > float(c1["open"])) and (c0_close < c0_open) and (c0_open > float(c1["close"])) and (c0_close < float(c1["open"])):
            if "Bearish Engulfing" not in patterns:
                patterns.append("Dark Cloud Cover-ish (bearish)")
    except Exception:
        pass

    # Return unique patterns while preserving order
    seen = set()
    unique_patterns = []
    for p in patterns:
        if p not in seen:
            unique_patterns.append(p)
            seen.add(p)
    return unique_patterns


def pattern_to_signal(patterns: List[str]) -> str:
    """
    Convert textual pattern list into a simple directional signal:
    Bullish / Bearish / Neutral. Conservative mapping.
    """
    if not patterns:
        return "Neutral"
    bulls = sum(1 for p in patterns if any(k in p.lower() for k in ["bull", "hammer", "piercing"]))
    bears = sum(1 for p in patterns if any(k in p.lower() for k in ["bear", "shooting", "dark cloud", "star"]))
    if bulls > bears:
        return "Bullish"
    if bears > bulls:
        return "Bearish"
    return "Neutral"


# ------------------------------
# NEXT-DAY PREDICTION (BITCOIN) & LOGGING
# ------------------------------
def compute_next_day_prediction_from_candle(df_ohlc: pd.DataFrame, pct_move_if_bull: float = 0.01, pct_move_if_bear: float = -0.01) -> Tuple[float, str, List[str]]:
    """
    Given df_ohlc (with columns open, high, low, close, timestamp), compute
    a simple next-day predicted close price and a signal string.
    Returns: predicted_price (float), signal (Bullish/Bearish/Neutral), detected_patterns (list).
    The prediction rule is intentionally simple:
      - If Bullish pattern detected -> predicted close = last_close * (1 + pct_move_if_bull)
      - If Bearish pattern detected -> predicted close = last_close * (1 + pct_move_if_bear)
      - Else -> predicted close = last_close (Neutral)
    pct_move_if_bull & pct_move_if_bear are fractions (e.g., 0.01 => +1%).
    """
    if df_ohlc is None or df_ohlc.empty:
        raise ValueError("OHLC df is empty")

    # Ensure sorted ascending by timestamp
    try:
        df_ohlc = df_ohlc.sort_values("timestamp").copy()
    except Exception:
        pass

    patterns = detect_candle_patterns(df_ohlc)
    signal = pattern_to_signal(patterns)

    try:
        last_close = float(df_ohlc["close"].iloc[-1])
    except Exception:
        # fallback: take the last available numeric close or raise
        closes = pd.to_numeric(df_ohlc["close"], errors="coerce").dropna()
        if not closes.empty:
            last_close = float(closes.iloc[-1])
        else:
            raise ValueError("No valid close price available in OHLC")

    if signal == "Bullish":
        predicted = last_close * (1.0 + abs(pct_move_if_bull))
    elif signal == "Bearish":
        predicted = last_close * (1.0 + pct_move_if_bear)
    else:
        predicted = last_close

    return predicted, signal, patterns


def log_candlestick_prediction_to_ai_file(asset: str, predicted_price: float, timestamp: Optional[datetime] = None):
    """
    Append a candlestick-based prediction to the ai_predictions_log CSV.
    We set asset column (e.g., 'Bitcoin'), predicted_price numeric, timestamp as provided or now.
    We dedupe by (timestamp, asset) — keep last, in case of duplicates.
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    row = {
        "timestamp": timestamp.isoformat(),
        "asset": asset,
        "predicted_price": float(predicted_price)
    }
    # append safely, dedupe on asset+timestamp
    append_prediction_to_log(AI_PRED_FILE, row, dedupe_on=["timestamp", "asset"])


# ------------------------------
# MAIN MENU & DASHBOARD
# ------------------------------
menu = st.sidebar.radio(
    "📊 Choose Dashboard",
    ["Gold & Bitcoin", "AI Forecast", "Jobs", "Real Estate Bot"]
)

# Helper to render simple header information consistently
def render_header(title: str, subtitle: Optional[str] = None):
    st.title(title)
    if subtitle:
        st.markdown(f"_{subtitle}_")


# ------------------------------
# GOLD & BITCOIN DASHBOARD
# ------------------------------
if menu == "Gold & Bitcoin":
    render_header("🌸 Gold & Bitcoin Market Dashboard (Pastel Theme)", "Combined view of predictions, actuals and AI forecasts")

    col1, col2 = st.columns(2)

    # iterate both assets to keep your original layout
    for col, df, name, actual_col in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"], ["gold_actual", "bitcoin_actual"]):
        with col:
            st.subheader(name)
            if df is not None and not df.empty:
                # Last signal & trend
                last_signal = df.get("signal", pd.Series(["Hold"])).iloc[-1] if not df.empty else "Hold"
                st.markdown(alert_badge(last_signal, name), unsafe_allow_html=True)
                last_trend = df.get("trend", pd.Series(["Neutral ⚖️"])).iloc[-1] if not df.empty else "Neutral ⚖️"
                st.markdown(f"**Market Trend:** {last_trend}")
                st.markdown(generate_summary(df, name))
                # target card
                try:
                    tp = df["target_price"].iloc[-1]
                except Exception:
                    tp = "N/A"
                th = df["target_horizon"].iloc[-1] if "target_horizon" in df.columns and not df.empty else "Days"
                target_price_card(tp, name, th)

                # explanation & assumptions
                try:
                    explanation_card(df, name)
                    assumptions_card(df, name)
                except Exception:
                    st.info("Error showing explanation or assumptions.")

                # AI forecast
                n_steps = 7
                try:
                    df_ai = predict_next_n(asset_name=name, n_steps=n_steps)
                except Exception:
                    df_ai = pd.DataFrame()

                theme = ASSET_THEMES.get(name, ASSET_THEMES["Gold"])

                # Build figure:
                fig = go.Figure()

                # Actual
                if "timestamp" in df.columns and "actual" in df.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers",
                                                 name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                    except Exception:
                        pass

                # Predicted
                if "timestamp" in df.columns and "predicted_price" in df.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers",
                                                 name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                    except Exception:
                        pass

                # AI future
                if not df_ai.empty and "timestamp" in df_ai.columns and "predicted_price" in df_ai.columns:
                    try:
                        fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"], mode="lines+markers",
                                                 name="AI Forecast", line=dict(color=theme["chart_ai"], dash="dot")))
                    except Exception:
                        pass

                fig.update_layout(title=f"{name} Prices: Actual + Predicted + AI Forecast",
                                  xaxis_title="Date", yaxis_title="Price",
                                  plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")

# ------------------------------
# AI FORECAST DASHBOARD (candles + pattern detection for Bitcoin)
# ------------------------------
elif menu == "AI Forecast":
    render_header("🤖 AI Forecast Dashboard", "This dashboard shows AI-predicted prices and candlestick-based pattern predictions (Bitcoin only)")

    n_steps = st.sidebar.number_input("Forecast next days", min_value=1, max_value=30, value=7)

    col_left, col_right = st.columns(2)

    # ---------- GOLD (left) ----------
    with col_left:
        asset = "Gold"
        actual_col = "gold_actual"
        st.subheader(asset)

        # Historical AI predictions vs Actual from CSV files
        st.markdown("**Historical AI Predictions vs Actual**")
        df_hist_actual = pd.DataFrame()
        if actual_col in df_actual.columns:
            df_hist_actual = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}).dropna()
        df_hist_pred = pd.DataFrame()
        if "asset" in df_ai_pred_log.columns:
            df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == asset][["timestamp", "predicted_price"]].dropna()

        if not df_hist_actual.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=df_hist_actual["timestamp"], y=df_hist_actual["actual"],
                mode="lines+markers", name="Actual Price", line=dict(width=2)
            ))
            fig_cmp.add_trace(go.Scatter(
                x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"],
                mode="lines+markers", name="Predicted Price", line=dict(dash="dot")
            ))
            fig_cmp.update_layout(title=f"{asset} – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white", height=350)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info(f"No historical data available for {asset} or no AI predictions logged.")

        # Future AI Forecast (line)
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

    # ---------- BITCOIN (right) ----------
    with col_right:
        asset = "Bitcoin"
        st.subheader(asset)

        # Prepare OHLC data if available in df_actual
        ohlc_cols_present = all(c in df_actual.columns for c in ["bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
        if ohlc_cols_present:
            df_ohlc = df_actual[["timestamp", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]].dropna().copy()
            # rename to standard
            df_ohlc = df_ohlc.rename(columns={
                "bitcoin_open": "open", "bitcoin_high": "high", "bitcoin_low": "low", "bitcoin_close": "close"
            })
            # ensure timestamp is datetime
            try:
                df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], infer_datetime_format=True)
            except Exception:
                pass
            df_ohlc = df_ohlc.sort_values("timestamp")
        else:
            # fallback: if only bitcoin_actual exists, build pseudo-ohlc by using actual as close and synthetic open/high/low
            if "bitcoin_actual" in df_actual.columns:
                d = df_actual[["timestamp", "bitcoin_actual"]].dropna().copy().sort_values("timestamp")
                # create small synthetic OHLC (not ideal but visual)
                d["open"] = d["bitcoin_actual"].shift(1).fillna(d["bitcoin_actual"])
                d["close"] = d["bitcoin_actual"]
                d["high"] = d[["open", "close"]].max(axis=1) * 1.002
                d["low"] = d[["open", "close"]].min(axis=1) * 0.998
                df_ohlc = d.rename(columns={"timestamp": "timestamp"})
            else:
                df_ohlc = pd.DataFrame()

        # Candlestick chart
        st.markdown("**Candlestick Chart (Bitcoin Real-Time Index)**")
        if not df_ohlc.empty:
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_ohlc["timestamp"],
                open=df_ohlc["open"],
                high=df_ohlc["high"],
                low=df_ohlc["low"],
                close=df_ohlc["close"],
                name="BTC"
            )])
            fig_candle.update_layout(title="Bitcoin Candlesticks", xaxis_rangeslider_visible=False, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=420)
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.info("OHLC data not available for Bitcoin; cannot render candlestick chart.")

        # Pattern detection & prediction (rule-based)
        st.markdown("**Candlestick Pattern Detection & Rule-based Prediction**")
        patterns = detect_candle_patterns(df_ohlc) if not df_ohlc.empty else []
        if patterns:
            st.markdown(f"**Detected patterns (latest):** {', '.join(patterns)}")
            signal = pattern_to_signal(patterns)
            if signal == "Bullish":
                st.success("Pattern-based prediction: **Bullish** — short-term upward bias 🚀")
            elif signal == "Bearish":
                st.error("Pattern-based prediction: **Bearish** — short-term downward bias 📉")
            else:
                st.info("Pattern-based prediction: **Neutral** — no clear short-term signal ⚖️")
        else:
            st.info("No clear candlestick pattern detected (insufficient data or none matched).")

        # Historical AI predictions vs Actual using AI log + actual close
        st.markdown("**Historical AI Predictions vs Actual**")
        df_hist_actual_close = pd.DataFrame()
        if "bitcoin_close" in df_actual.columns:
            df_hist_actual_close = df_actual[["timestamp", "bitcoin_close"]].rename(columns={"bitcoin_close": "actual"}).dropna()
        elif "bitcoin_actual" in df_actual.columns:
            df_hist_actual_close = df_actual[["timestamp", "bitcoin_actual"]].rename(columns={"bitcoin_actual": "actual"}).dropna()
        else:
            df_hist_actual_close = pd.DataFrame()

        df_hist_pred = pd.DataFrame()
        if "asset" in df_ai_pred_log.columns:
            df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == "Bitcoin"][["timestamp", "predicted_price"]].dropna()

        if not df_hist_actual_close.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_hist_actual_close["timestamp"], y=df_hist_actual_close["actual"], mode="lines+markers", name="Actual Price", line=dict(width=2)))
            fig_cmp.add_trace(go.Scatter(x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"], mode="lines+markers", name="Predicted Price", line=dict(dash="dot")))
            fig_cmp.update_layout(title="Bitcoin – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white", height=350)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("No historical comparison available (missing actual OHLC/close or AI predictions).")

        # Future AI forecast overlayed on candlestick: we will append forecast curve starting from last close
        st.markdown("**Future AI Forecast (overlay)**")
        try:
            df_ai_future = predict_next_n(asset_name="Bitcoin", n_steps=n_steps)
        except Exception:
            df_ai_future = pd.DataFrame()

        if not df_ai_future.empty:
            fig_overlay = go.Figure()
            # show candlestick if available
            if not df_ohlc.empty:
                fig_overlay.add_trace(go.Candlestick(
                    x=df_ohlc["timestamp"],
                    open=df_ohlc["open"], high=df_ohlc["high"], low=df_ohlc["low"], close=df_ohlc["close"],
                    name="BTC"
                ))
                last_date = df_ohlc["timestamp"].max()
                try:
                    last_close = float(df_ohlc.loc[df_ohlc["timestamp"] == last_date, "close"].iloc[0])
                except Exception:
                    last_close = float(df_ai_future["predicted_price"].iloc[0])
            else:
                # fallback to line actuals if no ohlc
                if not df_hist_actual_close.empty:
                    fig_overlay.add_trace(go.Scatter(x=df_hist_actual_close["timestamp"], y=df_hist_actual_close["actual"], mode="lines+markers", name="Actual", line=dict(width=2)))
                    last_date = df_hist_actual_close["timestamp"].max()
                    last_close = float(df_hist_actual_close["actual"].iloc[-1])
                else:
                    last_date = pd.Timestamp.now()
                    last_close = float(df_ai_future["predicted_price"].iloc[0])

            # connector + forecast line
            x_join = [last_date] + list(pd.to_datetime(df_ai_future["timestamp"]))
            y_join = [last_close] + list(df_ai_future["predicted_price"])
            fig_overlay.add_trace(go.Scatter(x=x_join, y=y_join, mode="lines+markers", name="AI Forecast", line=dict(color=ASSET_THEMES["Bitcoin"]["chart_ai"], dash="dot")))
            fig_overlay.update_layout(title="Bitcoin Candlesticks + AI Forecast Overlay", xaxis_rangeslider_visible=False, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA", height=500)
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.info("No AI forecast available for Bitcoin.")

        # ------------------------------
        # NEXT-DAY PREDICTION FROM LAST CANDLE (and logging)
        # ------------------------------
        st.markdown("**Next-Day Candlestick Prediction (Rule-based) — Bitcoin only**")

        if not df_ohlc.empty:
            try:
                predicted_price, next_signal, detected_patterns = compute_next_day_prediction_from_candle(df_ohlc, pct_move_if_bull=0.01, pct_move_if_bear=-0.01)

                st.markdown(f"**Detected patterns:** {', '.join(detected_patterns) if detected_patterns else 'None'}")
                if next_signal == "Bullish":
                    st.success(f"Next-day bias: **Bullish** — predicted close ≈ {predicted_price:.2f}")
                elif next_signal == "Bearish":
                    st.error(f"Next-day bias: **Bearish** — predicted close ≈ {predicted_price:.2f}")
                else:
                    st.info(f"Next-day bias: **Neutral** — predicted close ≈ {predicted_price:.2f}")

                # Logging toggle
                log_toggle = st.checkbox("Log this candlestick prediction into ai_predictions_log.csv", value=True, key="log_btc_candle")
                if log_toggle:
                    try:
                        # Use timezone-naive UTC ISO timestamp for logging
                        log_candlestick_prediction_to_ai_file("Bitcoin", predicted_price, timestamp=datetime.utcnow())
                        st.success("Candlestick prediction logged to ai_predictions_log.csv")
                        # Refresh local df_ai_pred_log so the UI picks up the new row (weak refresh)
                        try:
                            updated = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price"])
                            ensure_timestamp(updated)
                            # update global var in memory for this streamlit session
                            df_ai_pred_log = updated
                        except Exception:
                            pass
                    except Exception as e:
                        st.warning(f"Could not log prediction: {e}")
            except Exception as e:
                st.error(f"Error computing next-day candlestick prediction: {e}")
        else:
            st.info("Cannot compute candlestick prediction because OHLC data is not available.")


# ------------------------------
# JOBS MENU
# ------------------------------
elif menu == "Jobs":
    try:
        jobs_dashboard()
    except Exception as e:
        st.error(f"Error running jobs_dashboard(): {e}")


# ------------------------------
# REAL ESTATE BOT MENU
# ------------------------------
elif menu == "Real Estate Bot":
    try:
        real_estate_dashboard()
    except Exception as e:
        st.error(f"Error running real_estate_dashboard(): {e}")


# ------------------------------
# FOOTER: small diagnostics & downloads
# ------------------------------
st.markdown("---")
st.markdown("### Diagnostics & Data")
colA, colB, colC = st.columns([1, 1, 2])

with colA:
    st.markdown("**Data files**")
    st.write(f"Predictions file: `{PREDICTION_FILE}`")
    st.write(f"AI predictions log: `{AI_PRED_FILE}`")
    st.write(f"Actuals file: `{ACTUAL_FILE}`")

with colB:
    if st.button("Reload CSVs (refresh)"):
        # attempt to reload dataframes and notify user
        try:
            df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
            df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
            df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price"])
            ensure_timestamp(df_pred)
            ensure_timestamp(df_actual)
            ensure_timestamp(df_ai_pred_log)
            st.success("CSVs reloaded.")
        except Exception as e:
            st.warning(f"Reload failed: {e}")

with colC:
    st.markdown("**Download recent AI predictions (last 90 days)**")
    try:
        if not df_ai_pred_log.empty:
            df_recent = df_ai_pred_log.copy()
            try:
                df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"], errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                df_recent = df_recent[df_recent["timestamp"] >= cutoff]
            except Exception:
                pass
            csv_bytes = df_recent.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (AI predictions 90d)", data=csv_bytes, file_name="ai_predictions_recent.csv", mime="text/csv")
        else:
            st.info("No AI predictions logged yet.")
    except Exception:
        st.info("No AI predictions available for download.")


# ------------------------------
# END OF FILE: helpful note for users
# ------------------------------
st.markdown("""
---
**Notes**
- Candlestick-based next-day predictions are rule-based, simple, and illustrate pattern-driven heuristics. They are **not** financial advice.
- Logging writes to `data/ai_predictions_log.csv`. Ensure the Streamlit process has write permissions to the `data` directory.
- If modules like `jobs_app`, `ai_predictor`, or `real_estate_bot` are missing, the UI will show a warning but the rest of the application remains usable.
""")
