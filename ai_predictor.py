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
    return df_asset


def prepare_data(df, target_col="actual"):
    """Prepare data for ML model"""
    df = df.sort_values("timestamp")
    df["timestamp_ordinal"] = df["timestamp"].apply(lambda x: x.toordinal())
    X = df[["timestamp_ordinal", "predicted_price"]].values
    y = df[target_col].values
    return X, y


def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7):
    """Predict next n_steps based on historical actual + predicted prices using Random Forest"""
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < 5:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    X, y = prepare_data(df, target_col="actual")

    # Random Forest Regressor
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    last_date = df["timestamp"].max()
    last_pred_price = df["predicted_price"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps + 1)]
    X_future = np.array([[d.toordinal(), last_pred_price] for d in future_dates])

    y_pred = model.predict(X_future)

    df_future = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": y_pred
    })
    return df_future
