import pandas as pd
import os
import yaml
from datetime import datetime
from fetch_data import fetch_prices, fetch_indicators

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# Load indicator weights
with open("weight.yaml","r") as f:
    weights = yaml.safe_load(f)

def compute_prediction(latest_price, indicators, weight_dict):
    factor = sum(indicators.get(k,0)*weight_dict.get(k,0) for k in indicators)
    return latest_price * (1 + factor)

def main():
    timestamp = datetime.utcnow()
    latest_prices = fetch_prices()
    indicators = fetch_indicators()

    results = []
    for asset in ["Gold","Bitcoin"]:
        pred_price = compute_prediction(latest_prices[asset], indicators, weights[asset.lower()])
        results.append({
            "timestamp": timestamp,
            "asset": asset,
            "predicted_price": round(pred_price,2),
            "volatility": 0.0,  # Can implement rolling std later
            "risk": "Dynamic"
        })

    # Save to CSV
    df_results = pd.DataFrame(results)
    if os.path.exists(PREDICTION_FILE):
        df_results.to_csv(PREDICTION_FILE, mode='a', index=False, header=False)
    else:
        df_results.to_csv(PREDICTION_FILE, index=False)
    print("Predictions saved.")

if __name__ == "__main__":
    main()
