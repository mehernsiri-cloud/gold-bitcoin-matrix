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
            return pd.DataFrame()
        df = hist.reset_index()
        df = df.rename(columns={
            "Open": "bitcoin_open",
            "High": "bitcoin_high",
            "Low": "bitcoin_low",
            "Close": "bitcoin_close",
            "Datetime": "timestamp"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df[["timestamp", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"]]
    except Exception as e:
        print("⚠️ Error fetching Bitcoin OHLC:", e)
        return pd.DataFrame()


def fill_missing_ohlc(df_actual: pd.DataFrame, df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing OHLC values in historical actual_data.csv using the closest OHLC row.
    """
    if df_ohlc.empty:
        return df_actual

    df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"])
    df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"])

    for idx, row in df_actual.iterrows():
        if pd.isna(row["bitcoin_open"]) or pd.isna(row["bitcoin_high"]) or pd.isna(row["bitcoin_low"]) or pd.isna(row["bitcoin_close"]):
            # Find the closest OHLC timestamp
            closest_idx = (df_ohlc["timestamp"] - row["timestamp"]).abs().idxmin()
            df_actual.at[idx, "bitcoin_open"] = df_ohlc.at[closest_idx, "bitcoin_open"]
            df_actual.at[idx, "bitcoin_high"] = df_ohlc.at[closest_idx, "bitcoin_high"]
            df_actual.at[idx, "bitcoin_low"] = df_ohlc.at[closest_idx, "bitcoin_low"]
            df_actual.at[idx, "bitcoin_close"] = df_ohlc.at[closest_idx, "bitcoin_close"]

    return df_actual


def save_actual_data():
    ensure_data_folder()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gold = fetch_gold_price()
    btc_price = fetch_bitcoin_price()
    btc_ohlc = fetch_bitcoin_ohlc(period="90d", interval="1h")  # fetch enough history to fill missing

    if os.path.exists(ACTUAL_FILE):
        df_actual = pd.read_csv(ACTUAL_FILE)
        # Fill missing historical OHLC values
        df_actual = fill_missing_ohlc(df_actual, btc_ohlc)
    else:
        df_actual = pd.DataFrame(columns=[
            "timestamp", "gold_actual", "bitcoin_actual", "bitcoin_open", "bitcoin_high", "bitcoin_low", "bitcoin_close"
        ])

    # Append new row
    latest_ohlc = btc_ohlc.iloc[-1] if not btc_ohlc.empty else {}
    row = {
        "timestamp": ts,
        "gold_actual": gold if gold is not None else "N/A",
        "bitcoin_actual": btc_price if btc_price is not None else "N/A",
        "bitcoin_open": latest_ohlc.get("bitcoin_open", "N/A"),
        "bitcoin_high": latest_ohlc.get("bitcoin_high", "N/A"),
        "bitcoin_low": latest_ohlc.get("bitcoin_low", "N/A"),
        "bitcoin_close": latest_ohlc.get("bitcoin_close", "N/A"),
    }
    df_actual = pd.concat([df_actual, pd.DataFrame([row])], ignore_index=True)
    df_actual.to_csv(ACTUAL_FILE, index=False)

    print(f"✅ [{ts}] Saved actual prices and updated historical OHLC values.")


if __name__ == "__main__":
    save_actual_data()
