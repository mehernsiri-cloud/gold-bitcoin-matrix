# ai_predictor.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor
from functools import lru_cache

# ------------------------------
# Paths / constants
# ------------------------------
DATA_DIR = "data"
AI_LOG_FILE = os.path.join(DATA_DIR, "ai_predictions_log.csv")
ACTUAL_DATA_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

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
# Get historical daily prices
# ------------------------------
def load_historical_prices(asset_name: str) -> pd.Series:
    """Load daily average historical prices for a given asset"""
    if not os.path.exists(ACTUAL_DATA_FILE):
        raise FileNotFoundError(f"{ACTUAL_DATA_FILE} not found")

    df = pd.read_csv(ACTUAL_DATA_FILE, parse_dates=["timestamp"])
    col_map = {"gold": "gold_actual", "bitcoin": "bitcoin_actual"}
    col = col_map.get(asset_name.lower())
    if col is None or col not in df.columns:
        raise ValueError(f"Column for {asset_name} not found in actual_data.csv")

    # Convert to daily average
    df_daily = df.groupby(df["timestamp"].dt.date)[col].mean().reset_index()
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
    return df_daily[col].astype(float).reset_index(drop=True)

# ------------------------------
# Forecast next n-steps
# ------------------------------
@lru_cache(maxsize=10)
def predict_next_n(asset_name="Gold", n_steps=5, lags=5):
    """
    Train on historical daily averages + macro indicators to forecast next n_steps prices.
    Uses XGBoost Regressor with lag features + macro indicators.
    """
    # 1. Load history
    prices = load_historical_prices(asset_name).dropna().reset_index(drop=True)

    # 2. Load latest macro indicators
    macro = load_macro_indicators(asset_name)
    macro_features = [macro.get(k, 0.0) for k in macro.keys()]

    # 3. Build dataset with lag features + macro indicators
    X, y = [], []
    for i in range(lags, len(prices)):
        lag_window = prices[i - lags:i].tolist()
        if np.any(pd.isna(lag_window)) or pd.isna(prices[i]):
            continue
        X.append(lag_window + macro_features)
        y.append(prices[i])
    X, y = np.array(X), np.array(y)

    # 4. Handle insufficient data
    if len(X) == 0:
        last_price = prices.iloc[-1] if len(prices) > 0 else (
            2000.0 if asset_name.lower() == "gold" else 30000.0
        )
        future_dates = pd.date_range(
            start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=n_steps
        )
        df_out = pd.DataFrame(
            {"timestamp": future_dates, "predicted_price": [last_price] * n_steps}
        )
        _append_ai_log(df_out, asset_name)
        return df_out

    # 5. Train XGBoost model
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    # 6. Recursive prediction
    history = prices[-lags:].tolist()
    preds = []
    for _ in range(n_steps):
        input_features = history[-lags:] + macro_features
        input_features = [0 if pd.isna(v) else v for v in input_features]
        next_pred = model.predict([input_features])[0]
        preds.append(next_pred)
        history.append(next_pred)

    # 7. Output dataframe
    future_dates = pd.date_range(
        start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=n_steps
    )
    df_out = pd.DataFrame(
        {"timestamp": future_dates, "predicted_price": np.round(preds, 2)}
    )

    # 8. Save to log
    _append_ai_log(df_out, asset_name)
    return df_out

# ------------------------------
# Backtest (optional)
# ------------------------------
def backtest_ai(asset_name="Gold"):
    print(f"[ai_predictor] backtest_ai for {asset_name} called (no log stored).")
    return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])
