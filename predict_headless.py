#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime
import random

DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")
PRED_FILE = os.path.join(DATA_FOLDER, "predictions_log.csv")

def ensure_predictions_file():
    if not os.path.exists(PRED_FILE):
        pd.DataFrame(columns=["timestamp", "asset", "predicted_price", "volatility", "risk"]).to_csv(PRED_FILE, index=False)

def mock_predict(price: float):
    """Simple mock model: add ±2% random fluctuation"""
    if price is None or price == "":
        return None, None, None
    change = random.uniform(-0.02, 0.02)  # +/- 2%
    prediction = round(price * (1 + change), 2)
    volatility = round(abs(change), 4)
    risk = "High" if volatility > 0.015 else "Medium" if volatility > 0.005 else "Low"
    return prediction, volatility, risk

def generate_predictions():
    ensure_predictions_file()
    if not os.path.exists(ACTUAL_FILE):
        print("No actual_data.csv found, skipping.")
        return

    actual_df = pd.read_csv(ACTUAL_FILE)
    if actual_df.empty:
        print("No actual data available.")
        return

    latest = actual_df.iloc[-1]  # last row
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    predictions = []
    for asset in ["Gold", "Bitcoin"]:
        actual_price = latest["gold_actual"] if asset == "Gold" else latest["bitcoin_actual"]
        if pd.isna(actual_price) or actual_price == "":
            continue
        pred, vol, risk = mock_predict(float(actual_price))
        if pred:
            predictions.append({
                "timestamp": timestamp,
                "asset": asset,
                "predicted_price": pred,
                "volatility": vol,
                "risk": risk
            })

    if predictions:
        pred_df = pd.read_csv(PRED_FILE)
        pred_df = pd.concat([pred_df, pd.DataFrame(predictions)], ignore_index=True)
        pred_df.to_csv(PRED_FILE, index=False)
        print(f"[{timestamp}] Predictions saved for {len(predictions)} assets.")
    else:
        print(f"[{timestamp}] No predictions generated (missing data).")

if __name__ == "__main__":
    generate_predictions()
