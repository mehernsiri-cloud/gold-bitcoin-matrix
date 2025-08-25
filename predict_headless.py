#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime
import yaml
import random

DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")
PREDICTIONS_FILE = os.path.join(DATA_FOLDER, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

# Baseline realistic prices
BASELINE_PRICES = {"gold": 3409.0, "bitcoin": 47000.0}
MAX_INDICATOR_IMPACT = 0.05

def ensure_predictions_file():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(PREDICTIONS_FILE):
        df = pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "volatility", "risk"])
        df.to_csv(PREDICTIONS_FILE, index=False)

def load_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r") as f:
            return yaml.safe_load(f)
    return {"gold": {"dummy": 0.01}, "bitcoin": {"dummy": 0.01}}

def compute_prediction(baseline_price, asset, weights, max_impact=MAX_INDICATOR_IMPACT):
    indicators = weights.get(asset.lower(), {})
    total_adjustment = sum(v * max_impact for v in indicators.values() if isinstance(v, (int,float)))
    # Clip adjustment to realistic ±15% range
    total_adjustment = max(min(total_adjustment, 0.15), -0.15)
    predicted_price = baseline_price * (1 + total_adjustment)
    return round(predicted_price, 2)

def compute_volatility():
    return round(random.uniform(0.01, 0.05), 4)

def compute_risk(vol):
    if vol == "N/A":
        return "Unknown"
    if vol < 0.02:
        return "Low"
    elif vol < 0.04:
        return "Medium"
    else:
        return "High"

def generate_predictions():
    ensure_predictions_file()
    weights = load_weights()

    results = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for asset in ["gold", "bitcoin"]:
        baseline = BASELINE_PRICES[asset.lower()]
        predicted = compute_prediction(baseline, asset, weights)
        vol = compute_volatility()
        risk = compute_risk(vol)

        results.append({
            "timestamp": ts,
            "asset": asset.capitalize(),
            "predicted_price": predicted,
            "volatility": vol,
            "risk": risk
        })

    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    else:
        df = pd.DataFrame(results)

    df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"✅ [{ts}] Predictions saved.")

if __name__ == "__main__":
    generate_predictions()
