# fetch_data.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = "data"
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
ASSETS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD"
}

# Ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# FETCH REAL-TIME DATA
# ------------------------------
def fetch_price(symbol):
    try:
        df = yf.download(symbol, period="1d", interval="1h", progress=False)
        if df.empty:
            return None
        return df['Adj Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ------------------------------
# SAVE ACTUAL DATA
# ------------------------------
def save_actual_data():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    gold_price = fetch_price(ASSETS["gold"])
    btc_price = fetch_price(ASSETS["bitcoin"])

    # Initialize CSV if missing
    if not os.path.exists(ACTUAL_FILE):
        df_init = pd.DataFrame(columns=["timestamp","gold_actual","bitcoin_actual"])
        df_init.to_csv(ACTUAL_FILE, index=False)

    # Append new row
    df = pd.DataFrame([{
        "timestamp": now,
        "gold_actual": gold_price,
        "bitcoin_actual": btc_price
    }])
    df.to_csv(ACTUAL_FILE, mode="a", index=False, header=not os.path.exists(ACTUAL_FILE))
    print(f"Saved actual data at {now}")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    save_actual_data()
