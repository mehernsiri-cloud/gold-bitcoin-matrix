# ai_predictor.py
import os
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# Paths / constants
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
AI_LOG_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

MACRO_COLS = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]

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
    try:
        _ensure_data_dir()
        if df_out is None or df_out.empty:
            df_out = pd.DataFrame({
                "timestamp": [pd.Timestamp.utcnow()],
                "predicted_price": [0.0]
            })

        df_to_write = df_out.copy()
        df_to_write["asset"] = asset_name
        df_to_write["logged_at"] = datetime.utcnow().isoformat()

        if "predicted_price" not in df_to_write.columns and "predicted_ai" in df_to_write.columns:
            df_to_write = df_to_write.rename(columns={"predicted_ai": "predicted_price"})
        elif "predicted_price" not in df_to_write.columns:
            df_to_write["predicted_price"] = 0.0

        if not os.path.exists(AI_LOG_FILE) or os.path.getsize(AI_LOG_FILE) == 0:
            df_to_write.to_csv(AI_LOG_FILE, index=False)
            return

        old = pd.read_csv(AI_LOG_FILE, parse_dates=["timestamp"], infer_datetime_format=True)
        for col in ["predicted_price", "asset", "logged_at"]:
            if col not in old.columns:
                old[col] = np.nan

        combined = pd.concat([old, df_to_write], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["timestamp", "asset"], keep="last")
        combined.to_csv(AI_LOG_FILE, index=False)
    except Exception as e:
        print(f"[ai_predictor] ERROR while writing AI log: {e}")

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
        return pd.DataFrame()

    try:
        df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
        df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])
    except Exception as e:
        print(f"[ai_predictor] ERROR reading CSVs: {e}")
        return pd.DataFrame()

    df_asset = df_pred[df_pred.get("asset") == asset_name].copy()
    if df_asset.empty:
        return pd.DataFrame()

    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"}).sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
    else:
        df_asset["actual"] = np.nan

    df_asset["predicted_price"] = _safe_numeric(df_asset.get("predicted_price"))
    df_asset["actual"] = _safe_numeric(df_asset.get("actual"))

    macro_snapshot = load_macro_indicators(asset_name)
    for m in MACRO_COLS:
        df_asset[m] = float(macro_snapshot.get(m, 0.0))

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
def predict_next_n(df_actual=None, df_pred=None, asset_name="Gold", n_steps=7, window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < 1:
        df_out = pd.DataFrame({
            "timestamp": [pd.Timestamp.utcnow() + timedelta(days=i) for i in range(1, n_steps+1)],
            "predicted_price": [0.0]*n_steps
        })
        _append_ai_log(df_out, asset_name)
        return df_out

    X_train, y_train = create_features(df, window=window)

    # --- NEW: remove any NaNs ---
    mask = ~np.isnan(y_train)
    X_train, y_train = X_train[mask], y_train[mask]

    if X_train.size == 0 or y_train.size == 0:
        df_out = pd.DataFrame({
            "timestamp": [pd.Timestamp.utcnow() + timedelta(days=i) for i in range(1, n_steps+1)],
            "predicted_price": [0.0]*n_steps
        })
        _append_ai_log(df_out, asset_name)
        return df_out

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

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
        except Exception:
            next_pred = np.nan
        out.append(next_pred)
        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        last_macros = np.vstack([last_macros[1:], last_macros[-1]])

    df_out = pd.DataFrame({"timestamp": future_dates, "predicted_price": out})
    _append_ai_log(df_out, asset_name)
    return df_out


    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

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
        except Exception:
            next_pred = np.nan
        out.append(next_pred)
        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        last_macros = np.vstack([last_macros[1:], last_macros[-1]])

    df_out = pd.DataFrame({"timestamp": future_dates, "predicted_price": out})
    _append_ai_log(df_out, asset_name)
    return df_out

# ------------------------------
# Backtest AI predictions
# ------------------------------
def backtest_ai(asset_name="Gold", window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price", "actual", "asset"])

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
    _append_ai_log(df_out, asset_name)
    return df_out
