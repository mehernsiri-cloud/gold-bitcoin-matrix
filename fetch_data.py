import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# --- Paths ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# --- Functions ---
def fetch_gold():
    try:
        df = yf.Ticker("GC=F").history(period="1d")
        return round(float(df["Close"].iloc[-1]), 2)
    except:
        return None

def fetch_bitcoin():
    try:
        df = yf.Ticker("BTC-USD").history(period="1d")
        return round(float(df["Close"].iloc[-1]), 2)
    except:
        return None

def fetch_real_estate():
    # Placeholder values; replace with real API if available
    return {
        "france_studio_actual": 120000,
        "france_2bed_actual": 220000,
        "france_3bed_actual": 320000,
        "dubai_studio_actual": 150000,
        "dubai_2bed_actual": 300000,
        "dubai_3bed_actual": 450000
    }

# --- Main ---
def main():
    today = datetime.today().strftime("%Y-%m-%d")
    gold = fetch_gold()
    bitcoin = fetch_bitcoin()
    real_estate = fetch_real_estate()

    row = {
        "date": today,
        "gold_actual": gold,
        "bitcoin_actual": bitcoin,
        **real_estate
    }

    # Append to CSV
    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print("✅ Actual data updated:", row)

if __name__ == "__main__":
    main()
