#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime
import yaml

# Paths
DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")
PREDICTIONS_FILE = os.path.join(DATA_FOLDER, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

def ensure_predictions_file():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(PREDICTIONS_FILE):
        df = pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])
        df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"Created {PREDICTIONS_FILE} with headers.")

def load_weights():
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
    return weights

def compute_prediction(actual_price, asset, weights):
    """
    Simple weighted sum prediction based on indicators from weight.yaml
    For demo, we simulate a score and adjust actual price
    """
    indicators = weights.get(asset.lower(), {})
    score = sum(indicators.values())  # sum of weights as simple score
    predicted_price = actual_price * (1 + score)
    return round(predicted_price, 2)

def compute_volatility():
    # For simplicity, random small volatility; replace with real calculation if needed
    import random
    return round(random.uniform(0.01, 0.05), 4)

def compute_risk(vol):
    if vol < 0.02:
        return "Low"
    elif vol < 0.04:
        return "Medium"
    else:
        return "High"

def generate_predictions():
    ensure_predictions_file()
    weights = load_weights()

    if not os.path.exists(ACTUAL_FILE):
        print(f"{ACTUAL_FILE} not found! Cannot generate predictions.")
        return

    actual_df = pd.read_csv(ACTUAL_FILE)
    if actual_df.empty:
        print("No actual data available.")
        return

    # Use last row of actual prices
    last_row = actual_df.iloc[-1]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    results = []
    for asset in ["Gold", "Bitcoin"]:
        actual_price = last_row[f"{asset.lower()}_actual"]
        if pd.isna(actual_price):
            print(f"No actual price for {asset}, skipping prediction.")
            continue

        predicted_price = compute_prediction(actual_price, asset, weights)
        vol = compute_volatility()
        risk = compute_risk(vol)

        results.append({
            "timestamp": timestamp,
            "asset": asset,
            "predicted_price": predicted_price,
            "volatility": vol,
            "risk": risk
        })

    # Append to predictions CSV safely
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
    else:
        df = pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])

    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved predictions at {timestamp}")

if __name__ == "__main__":
    generate_predictions()
