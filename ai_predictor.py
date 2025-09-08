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
    """Load macro indicators (weights) from weight.yaml for a given asset"""
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
def load_historical_prices(asset_name: str) -> pd.DataFrame:
    """Load daily average historical prices for a given asset (returns DataFrame with timestamp & actual_price)"""
    if not os.path.exists(ACTUAL_DATA_FILE):
        raise FileNotFoundError(f"{ACTUAL_DATA_FILE} not found")

    df = pd.read_csv(ACTUAL_DATA_FILE, parse_dates=["timestamp"], infer_datetime_format=True)
    col_map = {"gold": "gold_actual", "bitcoin": "bitcoin_actual"}
    col = col_map.get(asset_name.lower())
    if col is None or col not in df.columns:
        raise ValueError(f"Column for {asset_name} not found in actual_data.csv")

    # Convert to daily average (group by date)
    df_daily = df.groupby(df["timestamp"].dt.date)[col].mean().reset_index()
    # ensure timestamp is datetime (midnight)
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"]).dt.normalize()
    df_daily.rename(columns={col: "actual_price"}, inplace=True)
    return df_daily[["timestamp", "actual_price"]]


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
    df_hist = load_historical_prices(asset_name)
    prices = df_hist["actual_price"].dropna().reset_index(drop=True)

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
            start=pd.Timestamp.now().normalize() + pd.Timedelta(days=1), periods=n_steps
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
        start=pd.Timestamp.now().normalize() + pd.Timedelta(days=1), periods=n_steps
    )
    df_out = pd.DataFrame(
        {"timestamp": future_dates, "predicted_price": np.round(preds, 2)}
    )

    # 8. Save to log
    _append_ai_log(df_out, asset_name)
    return df_out


# ------------------------------
# Compare predictions vs actuals (robust)
# ------------------------------
def compare_predictions_vs_actuals(asset_name="Gold") -> pd.DataFrame:
    """
    Merge AI predictions from data/ai_predictions_log.csv with actual prices
    from data/actual_data.csv for plotting.

    This function:
      - parses timestamps from both files,
      - normalizes them to dates (midnight) to avoid time-of-day mismatches,
      - merges on the normalized date.
    """
    if not os.path.exists(AI_LOG_FILE):
        raise FileNotFoundError(f"{AI_LOG_FILE} not found")
    if not os.path.exists(ACTUAL_DATA_FILE):
        raise FileNotFoundError(f"{ACTUAL_DATA_FILE} not found")

    # Load predictions (don't trust dtypes — coerce)
    preds = pd.read_csv(AI_LOG_FILE)
    if "timestamp" not in preds.columns:
        raise ValueError(f"'timestamp' column not found in {AI_LOG_FILE}")

    # parse timestamps (coerce errors to NaT) and keep original timestamp for plotting
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], errors="coerce", infer_datetime_format=True)
    # Drop rows where timestamp couldn't be parsed
    bad_preds = preds["timestamp"].isna().sum()
    if bad_preds:
        print(f"[ai_predictor] Warning: {bad_preds} rows in {AI_LOG_FILE} have unparseable timestamps and will be dropped.")
    preds = preds.dropna(subset=["timestamp"]).copy()

    # filter by asset (case-insensitive)
    preds = preds[preds["asset"].str.lower() == asset_name.lower()].copy()
    if preds.empty:
        return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual_price"])

    # normalize to date for matching
    preds["date"] = preds["timestamp"].dt.normalize()

    # Load actuals (already returns timestamp normalized)
    actuals = load_historical_prices(asset_name)
    # ensure timestamp parsed and normalized
    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"], errors="coerce").dt.normalize()
    actuals = actuals.dropna(subset=["timestamp"]).copy()
    actuals = actuals.rename(columns={"timestamp": "date"})

    # Merge on normalized date
    merged = pd.merge(preds, actuals[["date", "actual_price"]], on="date", how="left")

    # Convert predicted_price to numeric (coerce)
    if "predicted_price" in merged.columns:
        merged["predicted_price"] = pd.to_numeric(merged["predicted_price"], errors="coerce")
    else:
        merged["predicted_price"] = np.nan

    # Return friendly columns: keep original timestamp (not the normalized date)
    out = merged.rename(columns={"timestamp": "timestamp"})[["timestamp", "asset", "predicted_price", "actual_price"]]
    # Ensure timestamp is datetime and sorted
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


# ------------------------------
# Backtest (optional)
# ------------------------------
def backtest_ai(asset_name="Gold"):
    """Backtest placeholder for comparing AI predictions vs actual prices"""
    print(f"[ai_predictor] backtest_ai for {asset_name} called (no log stored).")
    return pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "actual"])


# ------------------------------
# Main for manual run
# ------------------------------
if __name__ == "__main__":
    for asset in ["Gold", "Bitcoin"]:
        print(f"Predicting {asset}...")
        df_pred = predict_next_n(asset_name=asset, n_steps=7, lags=5)
        print(df_pred)
        try:
            cmp = compare_predictions_vs_actuals(asset)
            print(cmp.tail())
        except Exception as e:
            print(f"compare failed: {e}")
