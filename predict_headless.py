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

CSV_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Load weights
# ------------------------------
with open(WEIGHT_FILE, "r") as f:
    weights = yaml.safe_load(f)

# ------------------------------
# Helper functions
# ------------------------------

ASSETS = {
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD"
}

def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="30d", interval="1h", progress=False)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def predict_price(df, asset):
    """Predict using weighted indicators"""
    last_price = df["Adj Close"].iloc[-1]
    # Use indicators weights as a rough multiplier
    weight_sum = sum(abs(v) for v in weights.get(asset.lower(), {}).values())
    adjustment = 0.01 * weight_sum  # simplified adjustment
    return last_price * (1 + adjustment)

def compute_volatility(df):
    df["returns"] = df["Adj Close"].pct_change()
    return df["returns"].std()

def ensure_csv_exists(file_path):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])
        df.to_csv(file_path, index=False)

def compute_risk(vol):
    if vol < 0.02:
        return "Low"
    elif vol < 0.05:
        return "Medium"
    else:
        return "High"

# ------------------------------
# Main
# ------------------------------
def main():
    ensure_csv_exists(CSV_FILE)
    re
