# ai_predictor.py
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

# UI helpers (safe to import even if not used by app.py)
import streamlit as st
import plotly.graph_objects as go

# ------------------------------
# Paths
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Theme & Icons (reused by renderer)
# ------------------------------
ASSET_THEMES = {
    "Gold": {
        "chart_actual": "#FBC02D",
        "chart_pred": "#FFCC80",
        "chart_ai": "#FF6F61",
        "buy": "#FFF9C4",
        "sell": "#FFE0B2",
        "hold": "#E0E0E0",
        "target_bg": "#FFFDE7",
        "target_text": "black",
        "assumption_pos": "#FFD54F",
        "assumption_neg": "#FFAB91",
    },
    "Bitcoin": {
        "chart_actual": "#42A5F5",
        "chart_pred": "#81D4FA",
        "chart_ai": "#FF6F61",
        "buy": "#BBDEFB",
        "sell": "#FFCDD2",
        "hold": "#CFD8DC",
        "target_bg": "#E3F2FD",
        "target_text": "black",
        "assumption_pos": "#64B5F6",
        "assumption_neg": "#EF9A9A",
    },
}

INDICATOR_ICONS = {
    "inflation": "💹",
    "real_rates": "🏦",
    "bond_yields": "📈",
    "energy_prices": "🛢️",
    "usd_strength": "💵",
    "liquidity": "💧",
    "equity_flows": "📊",
    "regulation": "🏛️",
    "adoption": "🤝",
    "currency_instability": "⚖️",
    "recession_probability": "📉",
    "tail_risk_event": "🚨",
    "geopolitics": "🌍",
}

MACRO_COLS = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]

# ------------------------------
# Utilities
# ------------------------------
def _safe_numeric(series):
    """Coerce to numeric, keep NaNs if bad values exist."""
    return pd.to_numeric(series, errors="coerce")

def _file_exists(*paths):
    return all(os.path.exists(p) for p in paths)

# ------------------------------
# Load macro indicators from weight.yaml
# ------------------------------
def load_macro_indicators(asset_name: str) -> dict:
    """
    Reads static macro weights for the asset from weight.yaml.
    These are treated as the latest macro snapshot and used as features.
    """
    if not os.path.exists(WEIGHT_FILE):
        return {}
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f) or {}
    return weights.get(asset_name.lower(), {}) or {}

# ------------------------------
# Load historical data including macro indicators
# ------------------------------
def load_data(asset_name: str, actual_col: str) -> pd.DataFrame:
    """
    Returns a dataframe with:
      timestamp, actual, predicted_price, and MACRO_COLS
    Macro columns are filled from weight.yaml (static snapshot),
    so they are present even if not in actual CSV.
    """
    if not _file_exists(PREDICTION_FILE, ACTUAL_FILE):
        return pd.DataFrame()

    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])

    # Filter predictions for this asset
    df_asset = df_pred[df_pred["asset"] == asset_name].copy()
    if df_asset.empty:
        return pd.DataFrame()

    # Merge actuals (asof to align on or before the prediction timestamp)
    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward",
        )
    else:
        df_asset["actual"] = np.nan

    # Coerce numerics
    df_asset["predicted_price"] = _safe_numeric(df_asset["predicted_price"])
    df_asset["actual"] = _safe_numeric(df_asset["actual"])

    # Inject macro snapshot from weight.yaml (static values)
    macro_snapshot = load_macro_indicators(asset_name)
    for m in MACRO_COLS:
        df_asset[m] = float(macro_snapshot.get(m, 0.0))

    # Keep only rows where we have both actual & model predicted_price
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    df_asset.reset_index(drop=True, inplace=True)
    return df_asset[["timestamp", "actual", "predicted_price", *MACRO_COLS]]

# ------------------------------
# Create rolling window features (prices + macro)
# ------------------------------
def create_features(df: pd.DataFrame, window: int = 3):
    """
    Builds feature matrix X and target y.
    Features = last `window` actuals + last `window` model_predicted + last `window` macro vectors.
    """
    X, y = [], []
    n = len(df)
    a = df["actual"].to_numpy()
    p = df["predicted_price"].to_numpy()
    M = df[MACRO_COLS].to_numpy()  # shape (n, 4)

    for i in range(window, n):
        # prices
        price_feat = np.concatenate([a[i - window : i], p[i - window : i]])  # shape 2*window
        # macros (window x 4) -> flattened
        macro_feat = M[i - window : i].reshape(-1)  # shape window*4
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(a[i])

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

# ------------------------------
# Predict next n steps (iterative)
# ------------------------------
def predict_next_n(df_actual: pd.DataFrame,
                   df_pred: pd.DataFrame,
                   asset_name: str = "Gold",
                   n_steps: int = 7,
                   window: int = 3) -> pd.DataFrame:
    """
    Uses a RandomForestRegressor trained on rolling features (prices + macros) to produce
    n-step ahead forecasts. Macro snapshot is kept constant during roll-forward.
    """
    # Load aligned dataset (independent from df_actual/df_pred args for simplicity)
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    # Train
    X_train, y_train = create_features(df, window=window)
    if X_train.size == 0:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Rolling state
    last_actuals = df["actual"].to_numpy()[-window:].tolist()
    last_preds = df["predicted_price"].to_numpy()[-window:].tolist()
    last_macros = df[MACRO_COLS].to_numpy()[-window:]  # shape (window, 4)

    # Future dates (daily)
    start_ts = pd.to_datetime(df["timestamp"].max())
    future_dates = [start_ts + timedelta(days=i) for i in range(1, n_steps + 1)]
    out = []

    for _ in range(n_steps):
        price_feat = np.concatenate([np.array(last_actuals), np.array(last_preds)])  # 2*window
        macro_feat = last_macros.reshape(-1)  # window*4
        features = np.concatenate([price_feat, macro_feat]).reshape(1, -1)

        next_pred = float(model.predict(features)[0])
        out.append(next_pred)

        # advance windows
        last_actuals = last_actuals[1:] + [next_pred]     # use predicted as proxy for next actual
        last_preds = last_preds[1:] + [next_pred]
        # keep macro snapshot constant (append the last row)
        last_macros = np.vstack([last_macros[1:], last_macros[-1]])

    return pd.DataFrame({"timestamp": future_dates, "predicted_price": out})

# ------------------------------
# Lightweight UI helpers (for AI Forecast section)
# ------------------------------
def _alert_badge(signal: str, asset_name: str) -> str:
    theme = ASSET_THEMES[asset_name]
    color = theme["buy"] if signal == "Buy" else theme["sell"] if signal == "Sell" else theme["hold"]
    return (
        f'<div style="background-color:{color};color:black;'
        f'padding:8px;font-size:20px;text-align:center;border-radius:8px">'
        f'{signal.upper()}</div>'
    )

def _target_price_card(price, asset_name, horizon: str):
    theme = ASSET_THEMES[asset_name]
    st.markdown(
        f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price:.2f} <br>⏳ Horizon: {horizon}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _assumptions_panel(asset_name: str):
    # show raw macro snapshot coming from weight.yaml
    weights = load_macro_indicators(asset_name)
    if not weights:
        st.info("No macro assumptions found.")
        return
    shown = {k: v for k, v in weights.items() if k in MACRO_COLS}
    if not shown:
        st.info("No macro assumptions found for the selected indicators.")
        return
    st.markdown("**Macro snapshot (from weight.yaml)**")
    st.json(shown)

# ------------------------------
# Render AI Forecast section (drop-in)
# ------------------------------
def render_ai_forecast(df_actual: pd.DataFrame, df_pred: pd.DataFrame, n_steps: int = 7):
    """
    Side-by-side AI Forecast for Gold & Bitcoin, matching your pastel layout vibe.
    Safe to call directly from app.py:
        from ai_predictor import render_ai_forecast
        ...
        render_ai_forecast(df_actual, df_pred, n_steps)
    """
    assets = [("Gold", "gold_actual"), ("Bitcoin", "bitcoin_actual")]
    col1, col2 = st.columns(2)

    for col, (asset, actual_col) in zip([col1, col2], assets):
        with col:
            st.subheader(asset)

            # Build forecast
            df_ai = predict_next_n(df_actual, df_pred, asset, n_steps)
            if df_ai.empty:
                st.info(f"No AI prediction available for {asset}.")
                continue

            # Determine a simple signal vs latest actual
            latest_actual_series = df_actual.get(actual_col)
            latest_actual = None
            if latest_actual_series is not None:
                latest_actual_series = pd.to_numeric(latest_actual_series, errors="coerce").dropna()
                if not latest_actual_series.empty:
                    latest_actual = float(latest_actual_series.iloc[-1])

            last_pred = float(df_ai["predicted_price"].iloc[-1])
            if latest_actual is None:
                signal = "Hold"
            else:
                signal = "Buy" if last_pred > latest_actual else "Sell" if last_pred < latest_actual else "Hold"

            # Trend from last 3 predicted points
            trend = "Neutral ⚖️"
            if len(df_ai) >= 3:
                seq = df_ai["predicted_price"].tail(3)
                if seq.is_monotonic_increasing:
                    trend = "Bullish 📈"
                elif seq.is_monotonic_decreasing:
                    trend = "Bearish 📉"

            st.markdown(_alert_badge(signal, asset), unsafe_allow_html=True)
            st.markdown(f"**Market Trend:** {trend}")
            _target_price_card(last_pred, asset, "Days")
            _assumptions_panel(asset)

            # Plot actual + AI predictions
            theme = ASSET_THEMES[asset]
            if actual_col in df_actual.columns:
                df_hist = (
                    df_actual[["timestamp", actual_col]]
                    .rename(columns={actual_col: "actual"})
                    .copy()
                )
            else:
                df_hist = pd.DataFrame({"timestamp": [], "actual": []})

            fig = go.Figure()
            if not df_hist.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_hist["timestamp"],
                        y=pd.to_numeric(df_hist["actual"], errors="coerce"),
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color=theme["chart_actual"], width=2),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=df_ai["timestamp"],
                    y=df_ai["predicted_price"],
                    mode="lines+markers",
                    name="AI Forecast",
                    line=dict(color=theme["chart_ai"], dash="dot"),
                )
            )
            fig.update_layout(
                title=f"{asset} AI Forecast vs Actual",
                xaxis_title="Date",
                yaxis_title="Price",
                plot_bgcolor="#FAFAFA",
                paper_bgcolor="#FAFAFA",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(df_ai)
