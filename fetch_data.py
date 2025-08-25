#!/usr/bin/env python3
import os
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

DATA_FOLDER = "data"
ACTUAL_FILE = os.path.join(DATA_FOLDER, "actual_data.csv")


def ensure_data_folder():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)


def fetch_bitcoin():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            timeout=10
        ).json()
        return float(r["bitcoin"]["usd"])
    except Exception as e:
        print("⚠️ Error fetching Bitcoin:", e)
        return None


def fetch_gold():
    try:
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print("⚠️ Error fetching Gold:", e)
    return None


def save_actual_data():
    ensure_data_folder()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gold = fetch_gold()
    btc = fetch_bitcoin()

    row = {
        "timestamp": ts,
        "gold_actual": gold if gold is not None else "N/A",
        "bitcoin_actual": btc if btc is not None else "N/A"
    }

    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print(f"✅ [{ts}] Saved actual prices: Gold={row['gold_actual']}, Bitcoin={row['bitcoin_actual']}")


if __name__ == "__main__":
    save_actual_data()
