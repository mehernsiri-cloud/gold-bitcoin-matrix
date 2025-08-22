#!/usr/bin/env python3
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# Paths
DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")

# Assets to fetch
ASSETS = {
    "Gold": "GC=F",      # Gold futures
    "Bitcoin": "BTC-USD" # Bitcoin
}

def ensure_data_folder():
    """Create data folder and CSV with headers if missing"""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(ACTUAL_FILE):
        df = pd.DataFrame(columns=["timestamp", "gold_actual", "bitcoin_actual"])
        df.to_csv(ACTUAL_FILE, index=False)
        print(f"Created {ACTUAL_FILE} with headers.")

def fetch_latest_price(symbol):
    """Fetch latest price from Yahoo Finance"""
    try:
        df = yf.download(symbol, period="1d", interval="1m", progress=False)
        if df.empty:
            print(f"No data fetched for {symbol}")
            return None
        latest_price = df['Adj Close'].iloc[-1]
        return round(float(latest_price), 2)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def save_actual_data():
    ensure_data_folder()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gold_price = fetch_latest_price(ASSETS["Gold"])
    bitcoin_price = fetch_latest_price(ASSETS["Bitcoin"])

    # Prepare row
    row = {
        "timestamp": timestamp,
        "gold_actual": gold_price,
        "bitcoin_actual": bitcoin_price
    }

    # Append row to CSV safely
    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
    else:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ACTUAL_FILE, index=False)
    print(f"Saved actual prices at {timestamp}")

if __name__ == "__main__":
    save_actual_data()
