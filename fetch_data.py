#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import requests
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
    """Fetch macro indicators; placeholder values now, replace with API calls"""
    indicators = {
        "inflation": 0.03,
        "real_rates": 0.01,
        "usd_strength": 1.0,
        "liquidity": 0.05,
        "equity_flows": 0.01,
        "bond
