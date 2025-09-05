# ai_predictor.py
import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

# UI helpers (safe to import)
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
# Small helpers
# ------------------------------
def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# Logging / ai log write
# ------------------------------
def _append_ai_log(df_out: pd.DataFrame, asset_name: str):
    """
    Append AI predictions (or backtest rows) to AI_LOG_FILE.
    Adds columns: asset, logged_at (UTC ISO).
    Always writes the file, and prints debug info for CI/logs.
    """
    try:
        if df_out is None or df_out.empty:
            print(f"[ai_predictor] No rows to log for {asset_name}; skipping write.")
            return

        _ensure_data_dir()
        df_to_write = df_out.copy()
        df_to_write["asset"] = asset_name
        # logged_at as ISO string for readability in logs/CSV
        df_to_write["logged_at"] = datetime.utcnow().isoformat()

        if os.path.exists(AI_LOG_FILE) and os.path.getsize(AI_LOG_FILE) > 0:
            try:
                old = pd.read_csv(AI_LOG_FILE, parse_dates=["timestamp"], infer_datetime_format=True)
            except Exception as e:
                print(f"[ai_predictor] Warning: failed to read existing AI_LOG_FILE ({e}), will overwrite.")
                old = pd.DataFrame()

            if not old.empty:
                # unify column names if necessary
                if "predicted_price" not in old.columns and "predicted_ai" in old.columns:
                    old = old.rename(columns={"predicted_ai": "predicted_price"})
                combined = pd.concat([old, df_to_write], ignore_index=True, sort=False)
                # Drop duplicates by timestamp+asset keeping last (so new forecasts replace old)
                combined = combined.drop_duplicates(subset=["timestamp", "asset"], keep="last")
                combined.to_csv(AI_LOG_FILE, index=False)
                print(f"[ai_predictor] Appended {len(df_to_write)} rows for {asset_name}. AI log now has {len(combined)} rows.")
                return
        # else: write new file (or overwrite empty)
        df_to_write.to_csv(AI_LOG_FILE, index=False)
        print(f"[ai_predictor] Wrote {len(df_to_write)} rows for {asset_name} to new AI log.")
    except Exception as e:
        print(f"[ai_predictor] ERROR while writing AI log: {e}")

# ------------------------------
# Macro snapshot loader
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
# Data loader (predictions + actuals + macro snapshot)
# ------------------------------
def load_data(asset_name: str, actual_col: str) -> pd.DataFrame:
    """
    Returns dataframe with columns:
    timestamp, actual, predicted_price, <MACRO_COLS...>
    Macro columns are populated from weight.yaml snapshot (static).
    """
    if not (os.path.exists(PREDICTION_FILE) and os.path.exists(ACTUAL_FILE)):
        print(f"[ai_predictor] Missing data files. Expected {PREDICTION_FILE} and {ACTUAL_FILE}")
        return pd.DataFrame()

    try:
        df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"], infer_datetime_format=True)
        df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"], infer_datetime_format=True)
    except Exception as e:
        print(f"[ai_predictor] ERROR reading CSVs: {e}")
        return pd.DataFrame()

    df_asset = df_pred[df_pred.get("asset") == asset_name].copy()
    if df_asset.empty:
        print(f"[ai_predictor] No prediction rows found for asset {asset_name} in {PREDICTION_FILE}")
        return pd.DataFrame()

    if actual_col in df_actual.columns:
        try:
            df_asset = pd.merge_asof(
                df_asset.sort_values("timestamp"),
                df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
                on="timestamp",
                direction="backward",
            )
        except Exception as e:
            print(f"[ai_predictor] merge_asof failed: {e}")
            df_asset["actual"] = np.nan
    else:
        df_asset["actual"] = np.nan

    df_asset["predicted_price"] = _safe_numeric(df_asset.get("predicted_price"))
    df_asset["actual"] = _safe_numeric(df_asset.get("actual"))

    # macro snapshot injection (static snapshot from weight.yaml)
    macro_snapshot = load_macro_indicators(asset_name)
    for m in MACRO_COLS:
        df_asset[m] = float(macro_snapshot.get(m, 0.0))

    # keep rows where we have both actual and predicted_price
    before = len(df_asset)
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    after = len(df_asset)
    print(f"[ai_predictor] Loaded {before} rows for {asset_name}, kept {after} rows after dropping NaNs.")

    df_asset = df_asset.reset_index(drop=True)
    # Ensure timestamp column present and typed
    if "timestamp" in df_asset.columns:
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
        price_feat = np.concatenate([a[i - window : i], p[i - window : i]])
        macro_feat = M[i - window : i].reshape(-1)
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(a[i])

    if len(X) == 0:
        return np.empty((0,)), np.empty((0,))
    return np.asarray(X), np.asarray(y)

# ------------------------------
# Forecast (future n-steps) with iterative roll-forward
# ------------------------------
def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        print(f"[ai_predictor] Not enough data to train for {asset_name} (need >= {window+1} rows).")
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    X_train, y_train = create_features(df, window=window)
    if X_train.size == 0:
        print(f"[ai_predictor] No features generated for {asset_name}.")
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    print(f"[ai_predictor] Trained RandomForest on {X_train.shape[0]} samples for {asset_name}.")

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
            print(f"[ai_predictor] Prediction error at step {step}: {e}")
            next_pred = float(np.nan)

        out.append(next_pred)

        # roll windows
        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        # keep macro snapshot constant (repeat last row)
        last_macros = np.vstack([last_macros[1:], last_macros[-1]])

    df_out = pd.DataFrame({"timestamp": future_dates, "predicted_price": out})
    # Save to AI log (with debug info)
    try:
        _append_ai_log(df_out, asset_name)
    except Exception as e:
        print(f"[ai_predictor] Failed to append AI log: {e}")

    print(f"[ai_predictor] Generated {len(df_out)} forecast rows for {asset_name}.")
    return df_out

# ------------------------------
# Backtest (generate historical AI preds and log them)
# ------------------------------
def backtest_ai(asset_name="Gold", window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 2:
        print(f"[ai_predictor] Not enough data for backtest for {asset_name}.")
        return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])

    preds, acts, dates = [], [], []
    for i in range(window, len(df) - 1):
        train_df = df.iloc[: i + 1].copy()
        X_train, y_train = create_features(train_df, window=window)
        if X_train.size == 0:
            continue

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        # Build features from last window in train_df
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
        # write backtest rows to AI log as well
        if not df_out.empty:
            _append_ai_log(df_out[["timestamp", "predicted_price", "asset"]].rename(columns={"predicted_price": "predicted_price"}), asset_name)
            print(f"[ai_predictor] Backtest logged {len(df_out)} rows for {asset_name}.")
    except Exception as e:
        print(f"[ai_predictor] Failed to log backtest: {e}")

    return df_out

# ------------------------------
# Lightweight UI helpers
# ------------------------------
def _alert_badge(signal: str, asset_name: str) -> str:
    theme = ASSET_THEMES.get(asset_name, {})
    color = theme.get("buy", "#FFF9C4") if signal == "Buy" else theme.get("sell", "#FFE0B2") if signal == "Sell" else theme.get("hold", "#E0E0E0")
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'

def _target_price_card(price, asset_name, horizon: str):
    theme = ASSET_THEMES.get(asset_name, {})
    st.markdown(
        f"""
        <div style='background-color:{theme.get("target_bg","#FFFDE7")};color:{theme.get("target_text","black")};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price:.2f} <br>⏳ Horizon: {horizon}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_ai_forecast(df_actual: pd.DataFrame, df_pred: pd.DataFrame, n_steps: int = 7):
    """
    Renders a two-column AI forecast view (Gold | Bitcoin). Safe to call from app.py.
    """
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

            theme = ASSET_THEMES.get(asset, {})
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
                    line=dict(color=theme.get("chart_actual","#42A5F5"), width=2),
                ))
            fig.add_trace(go.Scatter(
                x=df_ai["timestamp"],
                y=df_ai["predicted_price"],
                mode="lines+markers",
                name="AI Forecast",
                line=dict(color=theme.get("chart_ai","#FF6F61"), dash="dot"),
            ))
            fig.update_layout(title=f"{asset} AI Forecast vs Actual", xaxis_title="Date", yaxis_title="Price",
                              plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_ai)
