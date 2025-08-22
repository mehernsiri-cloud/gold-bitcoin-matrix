#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import yaml

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# Helper functions
# ------------------------------

def fetch_prices():
    """Fetch latest gold and bitcoin prices from Yahoo Finance"""
    assets = {"Gold": "GC=F", "Bitcoin": "BTC-USD"}
    prices = {}
    for name, symbol in assets.items():
        try:
            df = yf.download(symbol, period="1d", interval="1h", progress=False)
            prices[name] = round(df["Adj Close"].iloc[-1], 2)
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            prices[name] = None
    return prices

def fetch_indicators():
    """Fetch macro indicators; placeholder values for now"""
    indicators = {
        "inflation": 0.03,
        "real_rates": 0.01,
        "usd_strength": 1.0,
        "liquidity": 0.05,
        "equity_flows": 0.01,
        "bond_yields": 0.03,
        "regulation": 0.0,
        "adoption": 0.1,
        "currency_instability": 0.02,
        "recession_probability": 0.05,
        "tail_risk_event": 0.1,
        "geopolitics": 0.1,
        "energy_prices": 70.0
    }
    return indicators

def save_actual_data():
    prices = fetch_prices()
    indicators = fetch_indicators()
    timestamp = pd.Timestamp.now().floor('H')  # hourly timestamp
    date_str = timestamp.date()

    columns = [
        "timestamp", "date", "gold_actual", "bitcoin_actual",
        "inflation", "real_rates", "usd_strength", "liquidity",
        "equity_flows", "bond_yields", "regulation", "adoption",
        "currency_instability", "recession_probability", "tail_risk_event",
        "geopolitics", "energy_prices"
    ]

    row = {
        "timestamp": timestamp,
        "date": date_str,
        "gold_actual": prices.get("Gold"),
        "bitcoin_actual": prices.get("Bitcoin"),
        **indicators
    }

    df = pd.DataFrame([row], columns=columns)

    if os.path.exists(ACTUAL_FILE):
        df.to_csv(ACTUAL_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(ACTUAL_FILE, index=False)
    print("Actual data saved.")

if __name__ == "__main__":
    save_actual_data()
