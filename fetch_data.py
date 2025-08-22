import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Fetch functions ---
def fetch_gold():
    try:
        return yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print("Error fetching gold:", e)
        return None

def fetch_btc():
    try:
        return yf.Ticker("BTC-USD").history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print("Error fetching BTC:", e)
        return None

def save_actuals():
    today = datetime.utcnow().strftime("%Y-%m-%d")

    gold = fetch_gold()
    btc = fetch_btc()

    # Real estate reference values (update quarterly if needed)
    france_studio = 9400     # €/m²
    france_2bed   = 9500     # €/m²
    dubai_studio  = 700000   # AED avg price (Q1 2025)
    dubai_2bed    = 2170000  # AED avg price (Q1 2025)

    data = {
        "date": today,
        "gold_actual": round(gold, 2) if gold else None,
        "bitcoin_actual": round(btc, 2) if btc else None,
        "france_studio_price": france_studio,
        "france_2bed_price": france_2bed,
        "dubai_studio_price": dubai_studio,
        "dubai_2bed_price": dubai_2bed
    }

    df = pd.DataFrame([data])
    df.to_csv(
        ACTUAL_FILE,
        mode="a",
        index=False,
        header=not os.path.exists(ACTUAL_FILE)
    )

if __name__ == "__main__":
    save_actuals()
