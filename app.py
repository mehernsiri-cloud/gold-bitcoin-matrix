# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
from datetime import timedelta, datetime
from jobs_app import jobs_dashboard
from ai_predictor import predict_next_n, compare_predictions_vs_actuals
from real_estate_bot import real_estate_dashboard

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Dashboard", layout="wide")

# ------------------------------
# DATA FILES
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
AI_PRED_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# LOAD DATA
# ------------------------------
def load_csv_safe(path, default_cols, parse_ts=True):
    if os.path.exists(path):
        if parse_ts:
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=True)
            except Exception:
                # fallback if timestamp not present or parse fails
                df = pd.read_csv(path)
                if "timestamp" in df.columns:
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    except Exception:
                        pass
        else:
            df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=default_cols)
    return df

# Load files (safe)
df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"])
df_ai_pred_log = load_csv_safe(AI_PRED_FILE, ["timestamp", "asset", "predicted_price"])

# normalize timestamp columns where present
def ensure_timestamp(df, col="timestamp"):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            # leave as-is if cannot convert
            pass

ensure_timestamp(df_pred)
ensure_timestamp(df_actual)
ensure_timestamp(df_ai_pred_log)

# Load weights
if os.path.exists(WEIGHT_FILE):
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f) or {}
else:
    weights = {"gold": {}, "bitcoin": {}}

# ------------------------------
# EMOJI ICONS
# ------------------------------
INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

# ------------------------------
# THEMES
# ------------------------------
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
# HELPER FUNCTIONS (unchanged semantics)
# ------------------------------
def alert_badge(signal, asset_name):
    theme = ASSET_THEMES[asset_name]
    color = theme["buy"] if signal == "Buy" else theme["sell"] if signal == "Sell" else theme["hold"]
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'

def target_price_card(price, asset_name, horizon):
    theme = ASSET_THEMES[asset_name]
    st.markdown(f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price} <br>⏳ Horizon: {horizon}
        </div>
        """, unsafe_allow_html=True)

def explanation_card(asset_df, asset_name):
    st.subheader(f"📖 Explanation for {asset_name}")

    if "assumptions" in asset_df.columns and not asset_df.empty:
        try:
            assumptions_str = asset_df["assumptions"].iloc[-1]
            assumptions = eval(assumptions_str) if isinstance(assumptions_str, str) else {}
        except Exception:
            assumptions = {}
    else:
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

def assumptions_card(asset_df, asset_name):
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
    except:
        assumptions = {}

    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return

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
    theme = ASSET_THEMES[asset_name]
    colors = [theme["assumption_pos"] if v > 0 else theme["assumption_neg"] if v < 0 else theme["hold"] for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(x=[f"{icon} {ind}"], y=[val], marker_color=color, text=[f"{val:.2f}"], textposition='auto'))

    fig.update_layout(
        title=f"{asset_name} Assumptions & Target ({target_horizon})",
        yaxis_title="Weight / Impact",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_actual_pred(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    # ensure timestamps are same dtype
    if "timestamp" in asset_pred.columns and "timestamp" in df_actual.columns:
        try:
            asset_pred["timestamp"] = pd.to_datetime(asset_pred["timestamp"])
            df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"])
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
            # fallback: no tolerance merge
            asset_pred = pd.merge_asof(
                asset_pred.sort_values("timestamp"),
                df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
                on="timestamp",
                direction="backward"
            )
    else:
        asset_pred["actual"] = None

    asset_pred["predicted_price"] = pd.to_numeric(asset_pred["predicted_price"], errors="coerce")
    asset_pred["actual"] = pd.to_numeric(asset_pred["actual"], errors="coerce")

    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if row["predicted_price"] > row["actual"] else ("Sell" if row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    asset_pred["trend"] = ""
    if len(asset_pred) >= 3:
        last3 = asset_pred["predicted_price"].tail(3)
        if last3.is_monotonic_increasing:
            asset_pred["trend"] = "Bullish 📈"
        elif last3.is_monotonic_decreasing:
            asset_pred["trend"] = "Bearish 📉"
        else:
            asset_pred["trend"] = "Neutral ⚖️"

    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))

    horizon = "Days"
    if "volatility" in asset_pred.columns and not asset_pred["volatility"].isna().all():
        avg_vol = asset_pred["volatility"].mean()
        if avg_vol < 0.02:
            horizon = "Years"
        elif avg_vol < 0.05:
            horizon = "Months"
        else:
            horizon = "Days"
    asset_pred["target_horizon"] = horizon

    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"] == "Buy" else (row["predicted_price"] if row["signal"] == "Sell" else row["actual"]),
        axis=1
    )

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# WHAT-IF SLIDERS WITH SESSION STATE
# ------------------------------
st.sidebar.header("🔧 What-If Scenario")

def get_default_assumption(df, key, fallback):
    if df.empty:
        return fallback
    try:
        last_assumptions = eval(df["assumptions"].iloc[-1])
        return last_assumptions.get(key, fallback)
    except:
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

def apply_what_if(df):
    if df.empty:
        return df
    adj = 1 + st.session_state.inflation_adj * 0.01 - st.session_state.usd_adj * 0.01 + st.session_state.oil_adj * 0.01 - st.session_state.vix_adj * 0.005
    df = df.copy()
    df["predicted_price"] = df["predicted_price"] * adj
    df["target_price"] = df["predicted_price"]
    return df

gold_df_adj = apply_what_if(gold_df)
btc_df_adj = apply_what_if(btc_df)

def generate_summary(asset_df, asset_name):
    if asset_df.empty:
        return f"No data for {asset_name}"
    last_row = asset_df.iloc[-1]
    return f"**{asset_name} Market Summary:** Signal: {last_row['signal']} | Trend: {last_row['trend']} | Target Price: {last_row['target_price']}"

# ------------------------------
# Candlestick pattern detection (for Bitcoin)
# ------------------------------
def detect_candle_patterns(df_ohlc: pd.DataFrame) -> List[str]:
    """
    Inspect recent candles and return list of detected patterns (strings).
    Works best if df_ohlc has columns: timestamp, open, high, low, close.
    """
    patterns = []
    if df_ohlc is None or df_ohlc.shape[0] < 3:
        return patterns

    recent = df_ohlc.tail(3).copy().reset_index(drop=True)
    # last candle
    c0 = recent.iloc[-1]
    c1 = recent.iloc[-2]
    # basic helpers
    body0 = abs(c0["close"] - c0["open"])
    range0 = c0["high"] - c0["low"] if c0["high"] - c0["low"] > 0 else 1
    upper_wick = c0["high"] - max(c0["open"], c0["close"])
    lower_wick = min(c0["open"], c0["close"]) - c0["low"]

    # Doji
    if body0 < 0.15 * range0:
        patterns.append("Doji (indecision)")

    # Hammer
    if lower_wick > 2 * body0 and upper_wick < body0:
        if c0["close"] > c0["open"]:
            patterns.append("Hammer (bullish reversal)")

    # Shooting Star
    if upper_wick > 2 * body0 and lower_wick < body0:
        if c0["close"] < c0["open"]:
            patterns.append("Shooting Star (bearish reversal)")

    # Bullish Engulfing
    if (c1["close"] < c1["open"]) and (c0["close"] > c0["open"]) and (c0["close"] > c1["open"]) and (c0["open"] < c1["close"]):
        patterns.append("Bullish Engulfing")

    # Bearish Engulfing
    if (c1["close"] > c1["open"]) and (c0["close"] < c0["open"]) and (c0["open"] > c1["close"]) and (c0["close"] < c1["open"]):
        patterns.append("Bearish Engulfing")

    return patterns

def pattern_to_signal(patterns: List[str]) -> str:
    if not patterns:
        return "Neutral"
    bulls = sum(1 for p in patterns if any(k in p.lower() for k in ["bull", "hammer"]))
    bears = sum(1 for p in patterns if any(k in p.lower() for k in ["bear", "shooting", "star", "bearish"]))
    if bulls > bears:
        return "Bullish"
    if bears > bulls:
        return "Bearish"
    return "Neutral"

# ------------------------------
# MAIN MENU
# ------------------------------
menu = st.sidebar.radio(
    "📊 Choose Dashboard",
    ["Gold & Bitcoin", "AI Forecast", "Jobs", "Real Estate Bot"]
)

if menu == "Gold & Bitcoin":
    st.title("🌸 Gold & Bitcoin Market Dashboard (Pastel Theme)")
    col1, col2 = st.columns(2)

    for col, df, name, actual_col in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"], ["gold_actual", "bitcoin_actual"]):
        with col:
            st.subheader(name)
            if not df.empty:
                last_signal = df["signal"].iloc[-1]
                st.markdown(alert_badge(last_signal, name), unsafe_allow_html=True)
                last_trend = df["trend"].iloc[-1] if df["trend"].iloc[-1] else "Neutral ⚖️"
                st.markdown(f"**Market Trend:** {last_trend}")
                st.markdown(generate_summary(df, name))
                target_price_card(df["target_price"].iloc[-1], name, df["target_horizon"].iloc[-1])
                explanation_card(df, name)
                assumptions_card(df, name)

                # AI forecast
                n_steps = 7
                try:
                    df_ai = predict_next_n(asset_name=name, n_steps=n_steps)
                except TypeError:
                    df_ai = pd.DataFrame()

                theme = ASSET_THEMES[name]
                # Use line chart for Gold to keep original look; use candlestick for Bitcoin where OHLC available
                fig = go.Figure()
                # Actual
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers",
                                         name="Actual", line=dict(color=theme["chart_actual"], width=2)))
                # Predicted
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers",
                                         name="Predicted", line=dict(color=theme["chart_pred"], dash="dash")))
                # AI future
                if not df_ai.empty:
                    fig.add_trace(go.Scatter(x=df_ai["timestamp"], y=df_ai["predicted_price"], mode="lines+markers",
                                             name="AI Forecast", line=dict(color=theme["chart_ai"], dash="dot")))

                fig.update_layout(title=f"{name} Prices: Actual + Predicted + AI Forecast",
                                  xaxis_title="Date", yaxis_title="Price",
                                  plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")

elif menu == "AI Forecast":
    st.title("🤖 AI Forecast Dashboard")
    st.markdown("This dashboard shows **AI-predicted prices** and candlestick-based pattern predictions.")
    n_steps = st.sidebar.number_input("Forecast next days", min_value=1, max_value=30, value=7)

    # Two-column layout: left = Gold, right = Bitcoin (candles + pattern detection)
    col_left, col_right = st.columns(2)

    # ---------- GOLD (left) ----------
    with col_left:
        asset = "Gold"
        actual_col = "gold_actual"
        st.subheader(asset)

        # Historical AI predictions vs Actual from CSV files
        st.markdown("**Historical AI Predictions vs Actual**")
        df_hist_actual = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}) if actual_col in df_actual.columns else pd.DataFrame()
        df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == asset][["timestamp", "predicted_price"]] if "asset" in df_ai_pred_log.columns else pd.DataFrame()

        if not df_hist_actual.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=df_hist_actual["timestamp"], y=df_hist_actual["actual"],
                mode="lines+markers", name="Actual Price", line=dict(color="green")
            ))
            fig_cmp.add_trace(go.Scatter(
                x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"],
                mode="lines+markers", name="Predicted Price", line=dict(color="orange", dash="dot")
            ))
            fig_cmp.update_layout(title=f"{asset} – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
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
                fig.add_trace(go.Scatter(x=df_hist_actual["timestamp"], y=df_hist_actual["actual"], mode="lines+markers", name="Actual", line=dict(color="#42A5F5", width=2)))
            fig.add_trace(go.Scatter(x=df_ai_future["timestamp"], y=df_ai_future["predicted_price"], mode="lines+markers", name="AI Predicted", line=dict(color="#FF6F61", dash="dash")))
            fig.update_layout(title=f"{asset} AI Forecast vs Actual", xaxis_title="Date", yaxis_title="Price", plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
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
                df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"])
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
            fig_candle.update_layout(title="Bitcoin Candlesticks", xaxis_rangeslider_visible=False, plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
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
        df_hist_actual_close = None
        if "bitcoin_close" in df_actual.columns:
            df_hist_actual_close = df_actual[["timestamp", "bitcoin_close"]].rename(columns={"bitcoin_close": "actual"})
        elif "bitcoin_actual" in df_actual.columns:
            df_hist_actual_close = df_actual[["timestamp", "bitcoin_actual"]].rename(columns={"bitcoin_actual": "actual"})
        else:
            df_hist_actual_close = pd.DataFrame()

        df_hist_pred = df_ai_pred_log[df_ai_pred_log["asset"] == "Bitcoin"][["timestamp", "predicted_price"]] if "asset" in df_ai_pred_log.columns else pd.DataFrame()

        if not df_hist_actual_close.empty and not df_hist_pred.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_hist_actual_close["timestamp"], y=df_hist_actual_close["actual"], mode="lines+markers", name="Actual Price", line=dict(color="green")))
            fig_cmp.add_trace(go.Scatter(x=df_hist_pred["timestamp"], y=df_hist_pred["predicted_price"], mode="lines+markers", name="Predicted Price", line=dict(color="orange", dash="dot")))
            fig_cmp.update_layout(title="Bitcoin – Historical AI Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
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
                    fig_overlay.add_trace(go.Scatter(x=df_hist_actual_close["timestamp"], y=df_hist_actual_close["actual"], mode="lines+markers", name="Actual", line=dict(color="#42A5F5", width=2)))
                    last_date = df_hist_actual_close["timestamp"].max()
                    last_close = float(df_hist_actual_close["actual"].iloc[-1])
                else:
                    last_date = pd.Timestamp.now()
                    last_close = float(df_ai_future["predicted_price"].iloc[0])

            # connector + forecast line
            x_join = [last_date] + list(pd.to_datetime(df_ai_future["timestamp"]))
            y_join = [last_close] + list(df_ai_future["predicted_price"])
            fig_overlay.add_trace(go.Scatter(x=x_join, y=y_join, mode="lines+markers", name="AI Forecast", line=dict(color=ASSET_THEMES["Bitcoin"]["chart_ai"], dash="dot")))
            fig_overlay.update_layout(title="Bitcoin Candlesticks + AI Forecast Overlay", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.info("No AI forecast available for Bitcoin.")

elif menu == "Jobs":
    jobs_dashboard()

elif menu == "Real Estate Bot":
    real_estate_dashboard()
