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


def compute_prediction(actual_price, asset, weights):
    if actual_price == "N/A":
        return "N/A"
    score = sum(weights.get(asset.lower(), {}).values())
    return round(actual_price * (1 + score), 2)


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

    if not os.path.exists(ACTUAL_FILE):
        print("No actual_data.csv found!")
        return

    actual_df = pd.read_csv(ACTUAL_FILE)
    if actual_df.empty:
        print("actual_data.csv is empty!")
        return

    last_row = actual_df.iloc[-1]
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    results = []
    for asset in ["Gold", "Bitcoin"]:
        actual_price = last_row[f"{asset.lower()}_actual"]
        predicted = compute_prediction(actual_price, asset, weights)
        vol = compute_volatility() if predicted != "N/A" else "N/A"
        risk = compute_risk(vol)

        results.append({
            "timestamp": ts,
            "asset": asset,
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
