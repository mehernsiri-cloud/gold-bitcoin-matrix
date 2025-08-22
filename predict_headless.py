#!/usr/bin/env python3
import os
import pandas as pd
import yaml
from datetime import datetime

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "data"
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Load weights
# ------------------------------
if not os.path.exists(WEIGHT_FILE):
    raise FileNotFoundError(f"{WEIGHT_FILE} not found!")

with open(WEIGHT_FILE, "r") as f:
    weights = yaml.safe_load(f)

# ------------------------------
# Predict function
# ------------------------------
def predict_price(actual_df, asset_name):
    """
    Predict next price based on latest actual price and weighted score.
    """
    asset_map = {"Gold": "gold_actual", "Bitcoin": "bitcoin_actual"}
    col = asset_map.get(asset_name)
    if col not in actual_df.columns:
        raise KeyError(f"{col} not found in actual data CSV.")

    # Get latest price as float
    price_actual = actual_df[col].iloc[-1]
    try:
        price_actual = float(price_actual)
    except Exception:
        price_actual = 0.0

    # Compute simple weighted score from weight.yaml
    score = sum(weights[asset_name.lower()].values())
    predicted_price = price_actual * (1 + score)

    return round(predicted_price, 2)

# ------------------------------
# Main
# ------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Ensure prediction CSV exists
    if not os.path.exists(PREDICTION_FILE):
        pd.DataFrame(columns=["timestamp", "asset", "predicted_price"]).to_csv(PREDICTION_FILE, index=False)

    # Load actual data
    try:
        actual_df = pd.read_csv(ACTUAL_FILE)
    except Exception as e:
        print(f"Error reading {ACTUAL_FILE}: {e}")
        return

    if actual_df.empty:
        print("Actual data CSV is empty. Cannot predict.")
        return

    timestamp = pd.Timestamp.now().floor('h')  # hourly rounded timestamp

    results = []
    for asset in ["Gold", "Bitcoin"]:
        pred_price = predict_price(actual_df, asset)
        results.append({
            "timestamp": timestamp,
            "asset": asset,
            "predicted_price": pred_price
        })

    # Append predictions safely
    df_pred = pd.DataFrame(results)
    df_pred.to_csv(PREDICTION_FILE, mode='a', index=False, header=False)
    print(f"Predictions appended successfully at {timestamp}.")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    main()
