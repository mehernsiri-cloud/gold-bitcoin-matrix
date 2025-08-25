#!/usr/bin/env python3
import os
import requests
import pandas as pd
from datetime import datetime

DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")

COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
METALS_API_URL = "https://metals-api.com/api/latest"
METALS_API_KEY = os.environ.get("METALS_API_KEY")  # Add your API key as secret

def ensure_data_folder():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(ACTUAL_FILE):
        pd.DataFrame(columns=["timestamp", "gold_actual", "bitcoin_actual"]).to_csv(ACTUAL_FILE, index=False)
        print(f"Initialized CSV with headers: {ACTUAL_FILE}")

def fetch_bitcoin_price():
    try:
        resp = requests.get(COINGECKO_API, timeout=10)
        data = resp.json()
        return data.get("bitcoin", {}).get("usd")
    except Exception as e:
        print(f"Error fetching Bitcoin price: {e}")
        return None

def fetch_gold_price():
    if not METALS_API_KEY:
        print("No Metals-API key configured.")
        return None
    try:
        resp = requests.get(METALS_API_URL, params={"access_key": METALS_API_KEY}, timeout=10)
        data = resp.json()
        # Assuming JSON like {rates: {'Gold(USDXAU)': price}}
        price = None
        for k,v in data.get("rates", {}).items():
            if "Gold" in k or "XAU" in k:
                price = v
                break
        return price
    except Exception as e:
        print(f"Error fetching Gold price: {e}")
        return None

def save_actual_data():
    ensure_data_folder()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    btc = fetch_bitcoin_price()
    gold = fetch_gold_price()

    if btc is None and gold is None:
        print(f"[{timestamp}] Skipped writing row (no data).")
        return

    row = {
        "timestamp": timestamp,
        "gold_actual": round(gold,2) if gold else "",
        "bitcoin_actual": round(btc,2) if btc else ""
    }

    df = pd.read_csv(ACTUAL_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ACTUAL_FILE, index=False)
    print(f"[{timestamp}] Saved data: Gold={row['gold_actual']}, Bitcoin={row['bitcoin_actual']}")

if __name__ == "__main__":
    save_actual_data()
