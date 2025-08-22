#!/usr/bin/env python3
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = "data"
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# ------------------------------
# Helper Functions
# ------------------------------
def fetch_yfinance_price(ticker):
    """Fetch latest close price from Yahoo Finance."""
    try:
        df = yf.download(ticker, period="1d", interval="1h", progress=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        return df['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def get_macro_indicators():
    """
    Fetch or compute macro indicators dynamically.
    Placeholder values here; replace with real APIs if available.
    """
    return {
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
        "energy_prices": 65.0
    }

# ------------------------------
# Main Function
# ------------------------------
def save_actual_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Ensure CSV exists with header if missing
    if not os.path.exists(ACTUAL_FILE) or os.path.getsize(ACTUAL_FILE) == 0:
        columns = ["timestamp", "date", "gold_actual", "bitcoin_actual",
                   "inflation", "real_rates","usd_strength","liquidity",
                   "equity_flows","bond_yields","regulation","adoption",
                   "currency_instability","recession_probability",
                   "tail_risk_event","geopolitics","energy_prices"]
        pd.DataFrame(columns=columns).to_csv(ACTUAL_FILE, index=False)
        print(f"{ACTUAL_FILE} created with headers.")

    # Fetch real-time prices
    gold_price = fetch_yfinance_price("GC=F")
    bitcoin_price = fetch_yfinance_price("BTC-USD")

    if gold_price is None:
        gold_price = 0.0
    if bitcoin_price is None:
        bitcoin_price = 0.0

    # Get macro indicators
    indicators = get_macro_indicators()

    # Prepare row
    now = datetime.utcnow()
    row = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "gold_actual": gold_price,
        "bitcoin_actual": bitcoin_price
    }
    row.update(indicators)

    # Append row safely
    df = pd.read_csv(ACTUAL_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ACTUAL_FILE, index=False)
    print(f"Appended actual data at {row['timestamp']}")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    save_actual_data()
