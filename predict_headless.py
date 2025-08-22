import pandas as pd
import yaml
from datetime import datetime
import os
from fetch_data import fetch_prices, fetch_indicators, DATA_DIR

PRED_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

# --- Load weights ---
with open(WEIGHT_FILE, "r") as f:
    weights = yaml.safe_load(f)

# --- Prediction Function ---
def predict(asset, current_price, indicators):
    total = 0
    for key, weight in weights[asset.lower()].items():
        total += weight * indicators.get(key, 0)
    predicted_price = current_price * (1 + total)
    return round(predicted_price, 2)

# --- Main ---
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    indicators = fetch_indicators()
    prices = fetch_prices()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    results = []
    for asset, current_price in prices.items():
        if current_price is None:
            continue
        predicted_price = predict(asset, current_price, indicators)
        # Simple volatility calculation placeholder
        volatility = 0.02
        # Risk based on volatility
        risk = "Low" if volatility < 0.03 else "Medium"
        results.append({
            "timestamp": timestamp,
            "asset": asset,
            "predicted_price": predicted_price,
            "volatility": volatility,
            "r
