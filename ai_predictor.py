# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import os

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")


def load_data(asset_name, actual_col):
    """Load historical data for a given asset"""
    if not os.path.exists(PREDICTION_FILE) or not os.path.exists(ACTUAL_FILE):
        return pd.DataFrame()

    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])

    df_asset = df_pred[df_pred["asset"] == asset_name].copy()
    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual[["timestamp", actual_col]].sort_values("timestamp").rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        df_asset["actual"] = None

    df_asset["predicted_price"] = pd.to_numeric(df_asset["predicted_price"], errors="coerce")
    df_asset["actual"] = pd.to_numeric(df_asset["actual"], errors="coerce")
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    df_asset = df_asset.reset_index(drop=True)
    return df_asset


def create_rolling_features(df, window=3):
    """Create rolling window features for time series prediction"""
    X, y = [], []
    prices = df["actual"].values
    preds = df["predicted_price"].values
    n = len(df)

    for i in range(window, n):
        # Features: previous 'window' actuals + previous 'window' predicted prices
        feat = np.concatenate([prices[i - window:i], preds[i - window:i]])
        X.append(feat)
        y.append(prices[i])
    return np.array(X), np.array(y)


def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    """Predict next n_steps using Random Forest with rolling window"""
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    # Prepare rolling window features
    X_train, y_train = create_rolling_features(df, window=window)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Iterative multi-step prediction
    last_actuals = df["actual"].values[-window:].tolist()
    last_preds = df["predicted_price"].values[-window:].tolist()
    future_dates = [df["timestamp"].max() + timedelta(days=i) for i in range(1, n_steps + 1)]
    predictions = []

    for _ in range(n_steps):
        features = np.array(last_actuals + last_preds).reshape(1, -1)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)
        # update rolling window
        last_actuals = last_actuals[1:] + [next_pred]  # use predicted as next actual
        last_preds = last_preds[1:] + [next_pred]

    df_future = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": predictions
    })
    return df_future
