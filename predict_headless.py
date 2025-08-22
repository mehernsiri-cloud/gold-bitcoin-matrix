#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import random

# ------------------------------
# Config
# ------------------------------
ASSETS = {
    "gold": "GC=F",              # Gold futures
    "bitcoin": "BTC-USD",        # Bitcoin
    "fr_real_estate": "RWR",     # Proxy RE ETF
    "dubai_real_estate": "DXRE"  # Proxy RE ETF
}

CSV_FILE = "predictions_log.csv"

VOL_THRESHOLDS = {
    "gold": 0.005,
    "bitcoin": 0.02,
    "fr_real_estate": 0.01,
    "dubai_real_estate": 0.01
}

# ------------------------------
# Helper functions
# ------------------------------
def fetch_data(symbol):
    """Fetch last 30 days data"""
    try:
        df = yf.download(symbol, period="30d", interval="1d", progress=False)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        return df
    except Exception as e:
        print(f"Warning: Could not fetch {symbol} - {e}")
        return None

def compute_volatility(df):
    """Daily volatility as std of returns"""
    df['returns'] = df['Adj Close'].pct_change()
    return df['returns'].std()

def predict_price(df):
    """Simple forecast: latest price adjusted by last return"""
    last_price = df['Adj Close'].iloc[-1]
    last_return = df['Adj Close'].pct_change().iloc[-1]
    return last_price * (1 + last_return)

def compute_risk(vol, threshold):
    if vol < threshold:
        return "Low"
    elif vol < threshold * 2:
        return "Medium"
    else:
        return "High"

def ensure_csv_exists(file_path):
    """Create CSV with headers if missing"""
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
            "timestamp", "asset", "predicted_price", "volatility", "risk"
        ])
        df.to_csv(file_path, index=False)
        print(f"Created CSV: {file_path}")

def generate_placeholder(asset):
    """Generate safe placeholder predictions if fetch fails"""
    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "asset": asset,
        "predicted_price": round(random.uniform(100, 1000), 2),
        "volatility": round(random.uniform(0, 0.05), 4),
        "risk": "Placeholder"
    }

# ------------------------------
# Main
# ------------------------------
def main():
    ensure_csv_exists(CSV_FILE)
    results = []

    for asset, symbol in ASSETS.items():
        df = fetch_data(symbol)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        if df is not None:
            try:
                vol = compute_volatility(df)
                predicted_price = predict_price(df)
                risk = compute_risk(vol, VOL_THRESHOLDS[asset])

                results.append({
                    "timestamp": timestamp,
                    "asset": asset,
                    "predicted_price": round(predicted_price, 2),
                    "volatility": round(vol, 4),
                    "risk": risk
                })
            except Exception as e:
                print(f"Error computing {asset} - {e}")
                results.append(generate_placeholder(asset))
        else:
            results.append(generate_placeholder(asset))

    # Append to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(CSV_FILE, mode='a', index=False, header=False)
    print(f"Appended predictions to {CSV_FILE}")

if __name__ == "__main__":
    main()
