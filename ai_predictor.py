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
ACTUAL_DATA_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

MACRO_COLS = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]

# ------------------------------
# Helpers
# ------------------------------
def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# AI Log Writing
# ------------------------------
def _append_ai_log(df_out: pd.DataFrame, asset_name: str):
    """Append AI predictions to ai_predictions_log.csv"""
    try:
        _ensure_data_dir()
        if df_out is None or df_out.empty:
            return

        df_to_write = df_out.copy()
        df_to_write["asset"] = asset_name
        df_to_write["logged_at"] = datetime.utcnow().isoformat()

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
# Get historical actual prices
# ------------------------------
def load_historical_prices(asset_name: str) -> pd.Series:
    """Load historical prices for a given asset"""
    if not os.path.exists(ACTUAL_DATA_FILE):
        raise FileNotFoundError(f"{ACTUAL_DATA_FILE} not found")
    
    df = pd.read_csv(ACTUAL_DATA_FILE, parse_dates=["timestamp"])
    if asset_name.lower() == "gold":
        col = "gold_actual"
    elif asset_name.lower() == "bitcoin":
        col = "bitcoin_actual"
    else:
        raise ValueError(f"Unknown asset: {asset_name}")

    if col not in df.columns:
        raise ValueError(f"{col} column not found in actual_data.csv")

    df = df.sort_values("timestamp")
    return df[col].astype(float)

# ------------------------------
# Forecast next n-steps
# ------------------------------
# ------------------------------
# Forecast next n-steps
# ------------------------------
def predict_next_n(asset_name="Gold", n_steps=5):
    """
    Train on historical actual_data.csv and forecast next n_steps prices.
    Uses RandomForestRegressor with lag features (autoregressive).
    Handles missing values in historical data.
    """
    # 1. Load history
    prices = load_historical_prices(asset_name)

    # Drop NaNs if any
    prices = prices.dropna().reset_index(drop=True)

    # 2. Build lag features (autoregression)
    lags = 5
    X, y = [], []
    for i in range(lags, len(prices)):
        lag_window = prices[i-lags:i].values
        if np.any(np.isnan(lag_window)) or np.isnan(prices[i]):
            continue  # skip rows with NaNs
        X.append(lag_window)
        y.append(prices[i])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        # Not enough historical data, fallback to last known price
        last_price = prices.iloc[-1] if len(prices) > 0 else (2000.0 if asset_name.lower() == "gold" else 30000.0)
        future_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=n_steps)
        df_out = pd.DataFrame({
            "timestamp": future_dates,
            "predicted_price": [last_price] * n_steps
        })
        _append_ai_log(df_out, asset_name)
        return df_out

    # 3. Train model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    # 4. Predict forward autoregressively
    history = prices[-lags:].tolist()
    preds = []
    for _ in range(n_steps):
        # ensure last lags have no NaN
        input_features = [0 if np.isnan(v) else v for v in history[-lags:]]
        next_pred = model.predict([input_features])[0]
        preds.append(next_pred)
        history.append(next_pred)

    # 5. Prepare output dataframe
    df_out = pd.DataFrame({
        "timestamp": pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=n_steps),
        "predicted_price": np.round(preds, 2)
    })

    # 6. Save to log
    _append_ai_log(df_out, asset_name)

    return df_out


# ------------------------------
# Backtest (optional)
# ------------------------------
def backtest_ai(asset_name="Gold"):
    print(f"[ai_predictor] backtest_ai for {asset_name} called (no log stored).")
    return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])
