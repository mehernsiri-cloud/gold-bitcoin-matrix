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
AI_LOG_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
ACTUAL_DATA_FILE = os.path.join(DATA_DIR, "actual_data.csv")  # <-- add
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
            return

        df_to_write = df_out.copy()
        df_to_write["asset"] = asset_name
        df_to_write["logged_at"] = datetime.utcnow().isoformat()

        if "predicted_price" not in df_to_write.columns and "predicted_ai" in df_to_write.columns:
            df_to_write = df_to_write.rename(columns={"predicted_ai": "predicted_price"})

        if not os.path.exists(AI_LOG_FILE) or os.path.getsize(AI_LOG_FILE) == 0:
            df_to_write.to_csv(AI_LOG_FILE, index=False)
            return

        old = pd.read_csv(AI_LOG_FILE, parse_dates=["timestamp"])
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
# Get last known price (baseline)
# ------------------------------
def get_last_actual_price(asset_name: str) -> float:
    if os.path.exists(ACTUAL_DATA_FILE):
        try:
            df = pd.read_csv(ACTUAL_DATA_FILE)
            df = df[df["asset"].str.lower() == asset_name.lower()]
            if not df.empty and "actual" in df.columns:
                return float(df["actual"].iloc[-1])
        except Exception as e:
            print(f"[ai_predictor] Warning: cannot read actual_data.csv ({e})")

    # fallback baselines if no actual_data available
    if asset_name.lower() == "gold":
        return 2000.0
    elif asset_name.lower() == "bitcoin":
        return 30000.0
    else:
        return 100.0

# ------------------------------
# Forecast next n-steps (AI-driven only)
# ------------------------------
def predict_next_n(asset_name="Gold", n_steps=5):
    """
    Generate AI-driven forecast for the next n_steps.
    Baseline: last known actual price.
    AI: RandomForestRegressor using synthetic features + macro indicators.
    """
    future_dates = [pd.Timestamp.utcnow() + timedelta(days=i) for i in range(1, n_steps + 1)]

    # Load baseline last known price
    last_price = get_last_actual_price(asset_name)

    # Load macro indicators
    macro_snapshot = load_macro_indicators(asset_name)
    macro_values = [float(macro_snapshot.get(m, 0.0)) for m in MACRO_COLS]

    # Build synthetic training data (simple autoregression with noise)
    X_train = []
    y_train = []
    for i in range(30):  # last 30 "days"
        features = [last_price * (1 + 0.001 * np.random.randn())] + macro_values
        X_train.append(features)
        y_train.append(last_price * (1 + 0.01 * np.random.randn()))

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predict forward
    X_future = []
    base = last_price
    for _ in range(n_steps):
        features = [base] + macro_values
        pred = model.predict([features])[0]
        X_future.append(pred)
        base = pred  # autoregressive step

    predicted_prices = np.round(X_future, 2)

    df_out = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": predicted_prices
    })

    # Append AI-driven forecast to log
    _append_ai_log(df_out, asset_name)
    return df_out

# ------------------------------
# Optional: simple backtest function
# ------------------------------
def backtest_ai(asset_name="Gold"):
    print(f"[ai_predictor] backtest_ai for {asset_name} called (no log stored).")
    return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])
