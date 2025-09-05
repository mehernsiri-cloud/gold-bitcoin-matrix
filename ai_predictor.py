# ai_predictor.py
import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

# UI helpers
import streamlit as st
import plotly.graph_objects as go

# ------------------------------
# Paths
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
AI_LOG_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Theme & Icons
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
    },
}

MACRO_COLS = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]

# ------------------------------
# Helpers
# ------------------------------
def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def _append_ai_log(df_out, asset_name):
    """Append AI predictions to ai_predictions_log.csv with timestamp."""
    if df_out.empty:
        return
    os.makedirs(DATA_DIR, exist_ok=True)

    df_out = df_out.copy()
    df_out["asset"] = asset_name
    df_out["logged_at"] = datetime.utcnow()

    if os.path.exists(AI_LOG_FILE):
        old = pd.read_csv(AI_LOG_FILE, parse_dates=["timestamp", "logged_at"])
        df_out = pd.concat([old, df_out], ignore_index=True)

    df_out.to_csv(AI_LOG_FILE, index=False)

# ------------------------------
# Loaders
# ------------------------------
def load_macro_indicators(asset_name: str) -> dict:
    if not os.path.exists(WEIGHT_FILE):
        return {}
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f) or {}
    return weights.get(asset_name.lower(), {}) or {}

def load_data(asset_name: str, actual_col: str) -> pd.DataFrame:
    if not (os.path.exists(PREDICTION_FILE) and os.path.exists(ACTUAL_FILE)):
        return pd.DataFrame()

    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])

    df_asset = df_pred[df_pred["asset"] == asset_name].copy()
    if df_asset.empty:
        return pd.DataFrame()

    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward",
        )
    else:
        df_asset["actual"] = np.nan

    df_asset["predicted_price"] = _safe_numeric(df_asset["predicted_price"])
    df_asset["actual"] = _safe_numeric(df_asset["actual"])

    macro_snapshot = load_macro_indicators(asset_name)
    for m in MACRO_COLS:
        df_asset[m] = float(macro_snapshot.get(m, 0.0))

    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    return df_asset[["timestamp", "actual", "predicted_price", *MACRO_COLS]]

# ------------------------------
# Features
# ------------------------------
def create_features(df: pd.DataFrame, window: int = 3):
    X, y = [], []
    n = len(df)
    a = df["actual"].to_numpy()
    p = df["predicted_price"].to_numpy()
    M = df[MACRO_COLS].to_numpy()

    for i in range(window, n):
        price_feat = np.concatenate([a[i - window : i], p[i - window : i]])
        macro_feat = M[i - window : i].reshape(-1)
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(a[i])

    return np.asarray(X), np.asarray(y)

# ------------------------------
# Forecast
# ------------------------------
def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    X_train, y_train = create_features(df, window=window)
    if X_train.size == 0:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    last_actuals = df["actual"].to_numpy()[-window:].tolist()
    last_preds = df["predicted_price"].to_numpy()[-window:].tolist()
    last_macros = df[MACRO_COLS].to_numpy()[-window:]

    start_ts = pd.to_datetime(df["timestamp"].max())
    future_dates = [start_ts + timedelta(days=i) for i in range(1, n_steps + 1)]
    out = []

    for _ in range(n_steps):
        price_feat = np.concatenate([np.array(last_actuals), np.array(last_preds)])
        macro_feat = last_macros.reshape(-1)
        features = np.concatenate([price_feat, macro_feat]).reshape(1, -1)

        next_pred = float(model.predict(features)[0])
        out.append(next_pred)

        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        last_macros = np.vstack([last_macros[1:], last_macros[-1]])

    df_out = pd.DataFrame({"timestamp": future_dates, "predicted_price": out})
    _append_ai_log(df_out, asset_name)
    return df_out

# ------------------------------
# Backtest
# ------------------------------
def backtest_ai(asset_name="Gold", window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "asset", "predicted_ai", "actual"])

    preds, acts, dates = [], [], []
    for i in range(window, len(df) - 1):
        train_df = df.iloc[:i+1]
        X_train, y_train = create_features(train_df, window=window)
        if X_train.size == 0:
            continue

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        price_feat = np.concatenate([
            train_df["actual"].iloc[-window:].to_numpy(),
            train_df["predicted_price"].iloc[-window:].to_numpy()
        ])
        macro_feat = train_df[MACRO_COLS].iloc[-window:].to_numpy().reshape(-1)
        features = np.concatenate([price_feat, macro_feat]).reshape(1, -1)
        next_pred = float(model.predict(features)[0])

        preds.append(next_pred)
        acts.append(df["actual"].iloc[i+1])
        dates.append(df["timestamp"].iloc[i+1])

    df_out = pd.DataFrame({
        "timestamp": dates,
        "asset": asset_name,
        "predicted_ai": preds,
        "actual": acts
    })

    _append_ai_log(df_out.rename(columns={"predicted_ai": "predicted_price"}), asset_name)
    return df_out

# ------------------------------
# UI
# ------------------------------
def _alert_badge(signal: str, asset_name: str) -> str:
    theme = ASSET_THEMES[asset_name]
    color = theme["buy"] if signal == "Buy" else theme["sell"] if signal == "Sell" else theme["hold"]
    return f'<div style="background-color:{color};padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'

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

def render_ai_forecast(df_actual: pd.DataFrame, df_pred: pd.DataFrame, n_steps: int = 7):
    assets = [("Gold", "gold_actual"), ("Bitcoin", "bitcoin_actual")]
    col1, col2 = st.columns(2)

    for col, (asset, actual_col) in zip([col1, col2], assets):
        with col:
            st.subheader(asset)
            df_ai = predict_next_n(df_actual, df_pred, asset, n_steps)
            if df_ai.empty:
                st.info(f"No AI prediction available for {asset}.")
                continue

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

            st.markdown(_alert_badge(signal, asset), unsafe_allow_html=True)
            _target_price_card(last_pred, asset, "Days")

            theme = ASSET_THEMES[asset]
            if actual_col in df_actual.columns:
                df_hist = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}).copy()
            else:
                df_hist = pd.DataFrame({"timestamp": [], "actual": []})

            fig = go.Figure()
            if not df_hist.empty:
                fig.add_trace(go.Scatter(
                    x=df_hist["timestamp"],
                    y=pd.to_numeric(df_hist["actual"], errors="coerce"),
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color=theme["chart_actual"], width=2),
                ))
            fig.add_trace(go.Scatter(
                x=df_ai["timestamp"],
                y=df_ai["predicted_price"],
                mode="lines+markers",
                name="AI Forecast",
                line=dict(color=theme["chart_ai"], dash="dot"),
            ))
            fig.update_layout(title=f"{asset} AI Forecast vs Actual", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_ai)
