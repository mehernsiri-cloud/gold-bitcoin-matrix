#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime
import yaml
import random

# Paths
DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")
PREDICTIONS_FILE = os.path.join(DATA_FOLDER, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

def ensure_predictions_file():
    """Ensure predictions file and folder exist"""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(PREDICTIONS_FILE):
        df = pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "volatility", "risk"])
        df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"Created {PREDICTIONS_FILE} with headers.")

def load_weights():
    """Load weights safely from YAML"""
    if not os.path.exists(WEIGHT_FILE):
        print(f"⚠️ {WEIGHT_FILE} not found! Using default weights.")
        return {"gold": {}, "bitcoin": {}}

    try:
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f) or {}
        return weights
    except Exception as e:
        print(f"Error loading {WEIGHT_FILE}: {e}")
        return {"gold": {}, "bitcoin": {}}

def compute_prediction(actual_price, asset, weights):
    """
    Simple weighted sum prediction based on indicators from weight.yaml.
    For demo, sum of weights = score, prediction = actual_price * (1 + score).
    """
    indicators = weights.get(asset.lower(), {})
    if not indicators:
        score = 0
    else:
        score = sum(indicators.values())
    predicted_price = actual_price * (1 + score)
    return round(float(predicted_price), 2)

def compute_volatility():
    """Random volatility between 1% and 5%"""
    return round(random.uniform(0.01, 0.05), 4)

def compute_risk(vol):
    """Risk category based on volatility"""
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
        print(f"⚠️ {ACTUAL_FILE} not found! Cannot generate predictions.")
        return

    actual_df = pd.read_csv(ACTUAL_FILE)
    if actual_df.empty:
        print("⚠️ No actual data available.")
        return

    # Use last row of actual prices
    last_row = actual_df.iloc[-1]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    results = []
    for asset in ["Gold", "Bitcoin"]:
        col_name = f"{asset.lower()}_actual"
        if col_name not in last_row or pd.isna(last_row[col_name]):
            print(f"No actual price for {asset}, skipping prediction.")
            continue

        try:
            actual_price = float(last_row[col_name])
        except Exception:
            print(f"Invalid actual price for {asset}, skipping.")
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

    if not results:
        print("⚠️ No predictions generated.")
        return

    # Append to predictions CSV safely
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
    else:
        df = pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "volatility", "risk"])

    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"✅ Saved predictions at {timestamp}")

if __name__ == "__main__":
    generate_predictions()
