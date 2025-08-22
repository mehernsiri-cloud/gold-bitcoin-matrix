#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import random
import yaml

from fetch_data import fetch_prices, fetch_indicators, DATA_DIR

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.path.join(DATA_DIR, "predictions_log.csv")

VOL_THRESHOLDS = {
    "Gold": 0.01,
    "Bitcoin": 0.02
}

WEIGHTS_FILE = "weight.yaml"

# ------------------------------
# Helper functions
# ------------------------------
def load_weights():
    try:
        with open(WEIGHTS_FILE, "r") as f:
            weights = yaml.safe_load(f)
        return weights
    except Exception as e:
        print(f"⚠️ Warning loading weights: {e}")
        return {}

def compute_volatility(df):
    df['returns'] = df['Close'].pct_change()
    return df['returns'].std()

def predict_price(df, asset, indicators, weights):
    """Weighted linear forecast based on indicators"""
    base_price = df['Close'].iloc[-1]
    asset_weights = weights.get(asset.lower(), {})
    delta = sum(indicators.get(k, 0)*v for k, v in asset_weights.items())
    predicted_price = base_price * (1 + delta)
    return predicted_price

def compute_risk(vol, threshold):
    if vol < threshold:
        return "Low"
    elif vol < threshold*2:
        return "Medium"
    else:
        return "High"

def ensure_csv_exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        df = pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])
        df.to_csv(file_path, index=False)
        print(f"Created CSV: {file_path}")

# ------------------------------
# Main
# ------------------------------
def main():
    ensure_csv_exists(CSV_FILE)
    results = []

    indicators = fetch_indicators()
    weights = load_weights()

    for asset, ticker in {"Gold":"GC=F", "Bitcoin":"BTC-USD"}.items():
        try:
            df = yf.Ticker(ticker).history(period="30d")
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            vol = compute_volatility(df)
            predicted_price = predict_price(df, asset, indicators, weights)
            risk = compute_risk(vol, VOL_THRESHOLDS[asset])
        except Exception as e:
            print(f"⚠️ Error computing {asset}: {e}")
            predicted_price = None
            vol = None
            risk = "Error"

        results.append({
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset,
            "predicted_price": round(predicted_price,2) if predicted_price else None,
            "volatility": round(vol,4) if vol else None,
            "risk": risk
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(CSV_FILE, mode='a', index=False, header=False)
    print("✅ Predictions saved to CSV")

if __name__ == "__main__":
    main()
