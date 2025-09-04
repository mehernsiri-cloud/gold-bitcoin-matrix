# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import os
import yaml

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Load macro indicators from weight.yaml
# ------------------------------
def load_macro_indicators(asset_name):
    if not os.path.exists(WEIGHT_FILE):
        return {}
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
    return weights.get(asset_name.lower(), {})

# ------------------------------
# Load historical data including macro indicators
# ------------------------------
def load_data(asset_name, actual_col):
    if not os.path.exists(PREDICTION_FILE) or not os.path.exists(ACTUAL_FILE):
        return pd.DataFrame()

    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])

    df_asset = df_pred[df_pred["asset"] == asset_name].copy()

    # Merge with actuals
    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        df_asset["actual"] = np.nan

    # Load macro indicators from weight.yaml
    macro_indicators = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]
    macros = load_macro_indicators(asset_name)
    for ind in macro_indicators:
        df_asset[ind] = macros.get(ind, 0.0)

    df_asset["predicted_price"] = pd.to_numeric(df_asset["predicted_price"], errors="coerce")
    df_asset["actual"] = pd.to_numeric(df_asset["actual"], errors="coerce")
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    df_asset = df_asset.reset_index(drop=True)
    return df_asset

# ------------------------------
# Create rolling window features
# ------------------------------
def create_features(df, window=3):
    X, y = [], []
    n = len(df)
    macro_cols = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]
    for i in range(window, n):
        price_feat = np.concatenate([df["actual"].values[i - window:i],
                                     df["predicted_price"].values[i - window:i]])
        macro_feat = df[macro_cols].iloc[i - window:i].values.flatten()
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(df["actual"].values[i])
    return np.array(X), np.array(y)

# ------------------------------
# Predict next n steps
# ------------------------------
def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    X_train, y_train = create_features(df, window=window)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    last_actuals = df["actual"].values[-window:].tolist()
    last_preds = df["predicted_price"].values[-window:].tolist()
    last_macro = df[["inflation", "usd_strength", "energy_prices", "tail_risk_event"]].values[-window:].tolist()
    future_dates = [df["timestamp"].max() + timedelta(days=i) for i in range(1, n_steps + 1)]
    predictions = []

    for _ in range(n_steps):
        macro_feat = np.array(last_macro).flatten()
        features = np.array(last_actuals + last_preds + macro_feat.tolist()).reshape(1, -1)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)

        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        last_macro = last_macro[1:] + [last_macro[-1]]  # keep macro indicators constant

    df_future = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": predictions
    })
    return df_future
