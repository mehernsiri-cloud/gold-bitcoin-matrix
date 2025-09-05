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
# Paths / constants
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
AI_LOG_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

MACRO_COLS = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]

ASSET_THEMES = {
    "Gold": {
        "chart_actual": "#FBC02D", "chart_pred": "#FFCC80", "chart_ai": "#FF6F61",
        "buy": "#FFF9C4", "sell": "#FFE0B2", "hold": "#E0E0E0",
        "target_bg": "#FFFDE7", "target_text": "black"
    },
    "Bitcoin": {
        "chart_actual": "#42A5F5", "chart_pred": "#81D4FA", "chart_ai": "#FF6F61",
        "buy": "#BBDEFB", "sell": "#FFCDD2", "hold": "#CFD8DC",
        "target_bg": "#E3F2FD", "target_text": "black"
    }
}

# ------------------------------
# Helpers
# ------------------------------
def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

# ------------------------------
# AI Log Writing
# ------------------------------
def _append_ai_log(df_out: pd.DataFrame, asset_name: str):
    """
    Append AI predictions (forecast or backtest) to ai_predictions_log.csv.
    Always ensures at least one row is written to avoid empty Git commits.
    """
    _ensure_data_dir()

    # Ensure at least one row exists
    if df_out is None or df_out.empty:
        df_out = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_price": np.nan
        }])
        print(f"[ai_predictor] No predictions, adding dummy row for {asset_name}")

    df_out["asset"] = asset_name
    df_out["logged_at"] = datetime.utcnow().isoformat()

    # Read existing AI log
    if os.path.exists(AI_LOG_FILE) and os.path.getsize(AI_LOG_FILE) > 0:
        try:
            old = pd.read_csv(AI_LOG_FILE)
            combined = pd.concat([old, df_out], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp", "asset"], keep="last")
        except Exception as e:
            print(f"[ai_predictor] Warning reading old AI log: {e}")
            combined = df_out
    else:
        combined = df_out

    combined.to_csv(AI_LOG_FILE, index=False)
    print(f"[ai_predictor] AI log updated for {asset_name}, total rows: {len(combined)}")

# ------------------------------
# Load macro indicators
# ------------------------------
def load_macro_indicators(asset_name: str) -> dict:
    if not os.path.exists(WEIGHT_FILE):
        return {}
    try:
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f) or {}
        return weights.get(asset_name.lower(), {}) or {}
    except Exception as e:
        print(f"[ai_predictor] Warning: failed to load weight.yaml ({e})")
        return {}

# ------------------------------
# Load predictions + actuals
# ------------------------------
def load_data(asset_name: str, actual_col: str) -> pd.DataFrame:
    if not (os.path.exists(PREDICTION_FILE) and os.path.exists(ACTUAL_FILE)):
        print(f"[ai_predictor] Missing data files. Expected {PREDICTION_FILE} and {ACTUAL_FILE}")
        return pd.DataFrame()

    try:
        df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
        df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])
    except Exception as e:
        print(f"[ai_predictor] ERROR reading CSVs: {e}")
        return pd.DataFrame()

    df_asset = df_pred[df_pred.get("asset") == asset_name].copy()
    if df_asset.empty:
        print(f"[ai_predictor] No prediction rows for {asset_name}")
        return pd.DataFrame()

    if actual_col in df_actual.columns:
        try:
            df_asset = pd.merge_asof(
                df_asset.sort_values("timestamp"),
                df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}).sort_values("timestamp"),
                on="timestamp", direction="backward"
            )
        except Exception as e:
            print(f"[ai_predictor] merge_asof failed: {e}")
            df_asset["actual"] = np.nan
    else:
        df_asset["actual"] = np.nan

    df_asset["predicted_price"] = _safe_numeric(df_asset.get("predicted_price"))
    df_asset["actual"] = _safe_numeric(df_asset.get("actual"))

    macro_snapshot = load_macro_indicators(asset_name)
    for m in MACRO_COLS:
        df_asset[m] = float(macro_snapshot.get(m, 0.0))

    before = len(df_asset)
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    after = len(df_asset)
    print(f"[ai_predictor] Loaded {before} rows for {asset_name}, kept {after} after dropping NaNs.")

    df_asset = df_asset.reset_index(drop=True)
    df_asset["timestamp"] = pd.to_datetime(df_asset["timestamp"], errors="coerce")
    return df_asset.loc[:, ["timestamp", "actual", "predicted_price", *MACRO_COLS]]

# ------------------------------
# Feature builder
# ------------------------------
def create_features(df: pd.DataFrame, window: int = 3):
    X, y = [], []
    n = len(df)
    a = df["actual"].to_numpy()
    p = df["predicted_price"].to_numpy()
    M = df[MACRO_COLS].to_numpy()

    for i in range(window, n):
        price_feat = np.concatenate([a[i - window:i], p[i - window:i]])
        macro_feat = M[i - window:i].reshape(-1)
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(a[i])
    if not X:
        return np.empty((0,)), np.empty((0,))
    return np.asarray(X), np.asarray(y)

# ------------------------------
# Forecast next n-steps
# ------------------------------
def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        print(f"[ai_predictor] Not enough data for {asset_name}")
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    X_train, y_train = create_features(df, window=window)
    if X_train.size == 0:
        print(f"[ai_predictor] No features generated for {asset_name}")
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    print(f"[ai_predictor] RandomForest trained for {asset_name} on {X_train.shape[0]} samples")

    last_actuals = df["actual"].to_numpy()[-window:].tolist()
    last_preds = df["predicted_price"].to_numpy()[-window:].tolist()
    last_macros = df[MACRO_COLS].to_numpy()[-window:]
    start_ts = pd.to_datetime(df["timestamp"].max())
    future_dates = [start_ts + timedelta(days=i) for i in range(1, n_steps + 1)]

    out = []
    for step in range(n_steps):
        price_feat = np.concatenate([np.array(last_actuals), np.array(last_preds)])
        macro_feat = last_macros.reshape(-1)
        features = np.concatenate([price_feat, macro_feat]).reshape(1, -1)

        try:
            next_pred = float(model.predict(features)[0])
        except Exception as e:
            print(f"[ai_predictor] Prediction error step {step}: {e}")
            next_pred = np.nan

        out.append(next_pred)
        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        last_macros = np.vstack([last_macros[1:], last_macros[-1]])

    df_out = pd.DataFrame({"timestamp": future_dates, "predicted_price": out})
    try:
        _append_ai_log(df_out, asset_name)
    except Exception as e:
        print(f"[ai_predictor] Failed to log AI predictions: {e}")
    return df_out

# ------------------------------
# Backtest historical AI predictions
# ------------------------------
def backtest_ai(asset_name="Gold", window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 2:
        print(f"[ai_predictor] Not enough data for backtest for {asset_name}")
        return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])

    preds, acts, dates = [], [], []
    for i in range(window, len(df) - 1):
        train_df = df.iloc[:i + 1].copy()
        X_train, y_train = create_features(train_df, window)
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
        acts.append(float(df["actual"].iloc[i + 1]))
        dates.append(df["timestamp"].iloc[i + 1])

    df_out = pd.DataFrame({
        "timestamp": dates,
        "asset": asset_name,
        "predicted_price": preds,
        "actual": acts
    })

    try:
        _append_ai_log(df_out[["timestamp", "predicted_price", "asset"]], asset_name)
        print(f"[ai_predictor] Backtest logged {len(df_out)} rows for {asset_name}")
    except Exception as e:
        print(f"[ai_predictor] Failed to log backtest: {e}")
    return df_out

# ------------------------------
# UI helpers
# ------------------------------
def _alert_badge(signal: str, asset_name: str) -> str:
    theme = ASSET_THEMES.get(asset_name, {})
    color = theme.get("buy") if signal == "Buy" else theme.get("sell") if signal == "Sell" else theme.get("hold")
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'

def _target_price_card(price, asset_name, horizon: str):
    theme = ASSET_THEMES.get(asset_name, {})
    st.markdown(f"""
        <div style='background-color:{theme.get("target_bg","#FFFDE7")};color:{theme.get("target_text","black")};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price:.2f} <br>⏳ Horizon: {horizon}
        </div>
        """, unsafe_allow_html=True)

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
            signal = "Hold"
            if latest_actual is not None:
                signal = "Buy" if last_pred > latest_actual else "Sell" if last_pred < latest_actual else "Hold"

            st.markdown(_alert_badge(signal, asset), unsafe_allow_html=True)
            _target_price_card(last_pred, asset, "Days")

            theme = ASSET_THEMES.get(asset, {})
            df_hist = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}) if actual_col in df_actual.columns else pd.DataFrame({"timestamp": [], "actual": []})

            fig = go.Figure()
            if not df_hist.empty:
                fig.add_trace(go.Scatter(
                    x=df_hist["timestamp"], y=pd.to_numeric(df_hist["actual"], errors="coerce"),
                    mode="lines+markers", name="Actual", line=dict(color=theme.get("chart_actual","#42A5F5"), width=2)
                ))
            fig.add_trace(go.Scatter(
                x=df_ai["timestamp"], y=df_ai["predicted_price"],
                mode="lines+markers", name="AI Forecast", line=dict(color=theme.get("chart_ai","#FF6F61"), dash="dot")
            ))
            fig.update_layout(title=f"{asset} AI Forecast vs Actual", xaxis_title="Date", yaxis_title="Price",
                              plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_ai)
