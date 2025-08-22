#!/usr/bin/env python3
import pandas as pd
import os
from datetime import datetime
import yaml

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Load indicator weights
# ------------------------------
with open(WEIGHT_FILE, "r") as f:
    weights = yaml.safe_load(f)

ASSETS = ["Gold", "Bitcoin"]

def safe_read_actuals():
    if os.path.exists(ACTUAL_FILE) and os.path.getsize(ACTUAL_FILE) > 0:
        return pd.read_csv(ACTUAL_FILE)
    else:
        return pd.DataFrame()

def predict_price(latest_actual, asset_name):
    """Simple weighted prediction based on indicators"""
    if latest_actual.empty or asset_name.lower() not in weights:
        return None
    w = weights[asset_name.lower()]
    indicators = latest_actual.iloc[-1].to_dict()
    price_actual = indicators[f"{asset_name.lower()}_actual"]
    score = sum([indicators.get(k,0)*v for k,v in w.items()])
    predicted_price = price_actual * (1 + score)
    return round(predicted_price, 2)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    actual_df = safe_read_actuals()
    if actual_df.empty:
        print("No actual data available. Run fetch_data.py first.")
        return

    timestamp = pd.Timestamp.now().floor('H')
    results = []
    for asset in ASSETS:
        pred_price = predict_price(actual_df, asset)
        price_actual = actual_df[f"{asset.lower()}_actual"].iloc[-1]
        results.append({
            "timestamp": timestamp,
            "asset": asset,
            "predicted_price": pred_price,
            "actual_price": price_actual
        })

    df_pred = pd.DataFrame(results)

    if not os.path.exists(PREDICTION_FILE) or os.path.getsize(PREDICTION_FILE) == 0:
        df_pred.to_csv(PREDICTION_FILE, index=False)
        print("Prediction CSV created.")
    else:
        df_pred.to_csv(PREDICTION_FILE, mode='a', index=False, header=False)
        print("Prediction appended.")

if __name__ == "__main__":
    main()
