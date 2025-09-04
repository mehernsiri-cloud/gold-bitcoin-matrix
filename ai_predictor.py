# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import os

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# Load historical data with macro indicators
# ------------------------------
def load_data(asset_name, actual_col):
    """Load historical asset data including macro indicators"""
    if not os.path.exists(PREDICTION_FILE) or not os.path.exists(ACTUAL_FILE):
        return pd.DataFrame()

    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])

    # Merge actual and predicted prices
    df_asset = df_pred[df_pred["asset"] == asset_name].copy()
    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col,
                                               "inflation", "usd_strength",
                                               "energy_prices", "tail_risk_event"]].rename(
                columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        df_asset["actual"] = None
        df_asset["inflation"] = 0.0
        df_asset["usd_strength"] = 0.0
        df_asset["energy_prices"] = 0.0
        df_asset["tail_risk_event"] = 0.0

    df_asset["predicted_price"] = pd.to_numeric(df_asset["predicted_price"], errors="coerce")
    df_asset["actual"] = pd.to_numeric(df_asset["actual"], errors="coerce")
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    df_asset = df_asset.reset_index(drop=True)
    return df_asset

# ------------------------------
# Create rolling window features including macro indicators
# ------------------------------
def create_features(df, window=3):
    """Generate features for ML model"""
    X, y = [], []
    n = len(df)
    for i in range(window, n):
        # past actuals and predicted prices
        price_feat = np.concatenate([df["actual"].values[i - window:i],
                                     df["predicted_price"].values[i - window:i]])
        # last macro indicators
        macro_feat = df[["inflation", "usd_strength", "energy_prices", "tail_risk_event"]].iloc[i - window:i].values.flatten()
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(df["actual"].values[i])
    return np.array(X), np.array(y)

# ------------------------------
# AI Forecast function
# ------------------------------
def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    """
    Predict next n_steps using Random Forest with rolling window features + macro indicators
    """
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    # Prepare features
    X_train, y_train = create_features(df, window=window)

    # Train model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Iterative multi-step forecast
    last_actuals = df["actual"].values[-window:].tolist()
    last_preds = df["predicted_price"].values[-window:].tolist()
    last_macro = df[["inflation", "usd_strength", "energy_prices", "tail_risk_event"]].values[-window:].tolist()
    future_dates = [df["timestamp"].max() + timedelta(days=i) for i in range(1, n_steps + 1)]
    predictions = []

    for i in range(n_steps):
        # flatten macro features
        macro_feat = np.array(last_macro).flatten()
        features = np.array(last_actuals + last_preds + macro_feat).reshape(1, -1)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)

        # update rolling windows
        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]

        # assume macro indicators remain constant (can enhance later)
        last_macro = last_macro[1:] + [last_macro[-1]]

    df_future = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": predictions
    })
    return df_future
