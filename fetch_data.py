#!/usr/bin/env python3
import os
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

# Constants
DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")


def ensure_data_folder():
    """Create data folder if missing"""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)


def fetch_bitcoin():
    """Fetch Bitcoin price in USD from CoinGecko"""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        r = requests.get(url, timeout=10).json()
        return float(r["bitcoin"]["usd"])
    except Exception as e:
        print("⚠️ Error fetching Bitcoin:", e)
        return None


def fetch_gold():
    """Fetch Gold price from Yahoo Finance (Gold Futures GC=F)"""
    try:
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print("⚠️ Error fetching Gold:", e)
    return None


def update_actual_data():
    """Append latest prices to actual_data.csv"""
    ensure_data_folder()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gold_price = fetch_gold()
    btc_price = fetch_bitcoin()

    # Replace None with "N/A" to avoid empty cells
    row = {
        "timestamp": ts,
        "gold_actual": gold_price if gold_price is not None else "N/A",
        "bitcoin_actual": btc_price if btc_price is not None else "N/A",
    }

    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print(f"✅ [{ts}] Saved: Gold={row['gold_actual']}, Bitcoin={row['bitcoin_actual']}")


if __name__ == "__main__":
    update_actual_data()
