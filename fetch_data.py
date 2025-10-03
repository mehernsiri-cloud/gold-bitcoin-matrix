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


def fetch_bitcoin_price():
    """Fetch current Bitcoin price from CoinGecko"""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            timeout=10
        ).json()
        return float(r["bitcoin"]["usd"])
    except Exception as e:
        print("⚠️ Error fetching Bitcoin price:", e)
        return None


def fetch_gold_price():
    """Fetch current Gold price from Yahoo Finance"""
    try:
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print("⚠️ Error fetching Gold price:", e)
    return None


def fetch_bitcoin_ohlc(period="7d", interval="1h"):
    """Fetch Bitcoin OHLC data from Yahoo Finance"""
    try:
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period=period, interval=interval)
        if hist.empty:
            return None
        df = hist.reset_index()
        df = df.rename(columns={
            "Open": "bitcoin_open",
            "High": "bitcoin_high",
            "Low": "bitcoin_low",
            "Close": "bitcoin_close",
            "Datetime": "timestamp"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Use only the latest OHLC row
        latest = df.iloc[-1]
        return {
            "bitcoin_open": float(latest["bitcoin_open"]),
            "bitcoin_high": float(latest["bitcoin_high"]),
            "bitcoin_low": float(latest["bitcoin_low"]),
            "bitcoin_close": float(latest["bitcoin_close"]),
        }
    except Exception as e:
        print("⚠️ Error fetching Bitcoin OHLC:", e)
        return {"bitcoin_open": "N/A", "bitcoin_high": "N/A", "bitcoin_low": "N/A", "bitcoin_close": "N/A"}


def save_actual_data():
    ensure_data_folder()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gold = fetch_gold_price()
    btc_price = fetch_bitcoin_price()
    btc_ohlc = fetch_bitcoin_ohlc()

    row = {
        "timestamp": ts,
        "gold_actual": gold if gold is not None else "N/A",
        "bitcoin_actual": btc_price if btc_price is not None else "N/A",
        "bitcoin_open": btc_ohlc.get("bitcoin_open", "N/A"),
        "bitcoin_high": btc_ohlc.get("bitcoin_high", "N/A"),
        "bitcoin_low": btc_ohlc.get("bitcoin_low", "N/A"),
        "bitcoin_close": btc_ohlc.get("bitcoin_close", "N/A"),
    }

    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print(f"✅ [{ts}] Saved actual prices: Gold={row['gold_actual']}, Bitcoin={row['bitcoin_actual']}, "
          f"OHLC=({row['bitcoin_open']}, {row['bitcoin_high']}, {row['bitcoin_low']}, {row['bitcoin_close']})")


if __name__ == "__main__":
    save_actual_data()
