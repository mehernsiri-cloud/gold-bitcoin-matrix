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

        old = pd.read_csv(AI_LOG_FILE, parse_dates=["timestamp"])
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
# Forecast next n-steps (AI-driven only)
# ------------------------------
def predict_next_n(asset_name="Gold", n_steps=5):
    """
    Generate AI-driven forecast for the next n_steps.
    Only store the predicted prices in ai_predictions_log.csv.
    """
    future_dates = [pd.Timestamp.utcnow() + timedelta(days=i) for i in range(1, n_steps + 1)]

    # Load macro indicators (optional for model input)
    macro_snapshot = load_macro_indicators(asset_name)
    macro_values = [float(macro_snapshot.get(m, 0.0)) for m in MACRO_COLS]

    # Generate dummy AI-driven predictions (replace with your AI model logic)
    np.random.seed(42)  # for reproducibility
    predicted_prices = np.round(100 + np.random.randn(n_steps) * 2, 2)

    df_out = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": predicted_prices
    })

    # Append AI-driven forecast to log
    _append_ai_log(df_out, asset_name)
    return df_out

# ------------------------------
# Optional: simple backtest function (for historical testing)
# ------------------------------
def backtest_ai(asset_name="Gold"):
    """
    Optional: placeholder backtest function.
    Does NOT affect ai_predictions_log.csv.
    """
    print(f"[ai_predictor] backtest_ai for {asset_name} called (no log stored).")
    return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])
