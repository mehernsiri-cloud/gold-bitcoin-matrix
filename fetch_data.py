import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# File path
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

def fetch_yahoo_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1d")
        return round(data["Close"].iloc[-1], 2)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_real_estate_price(region="france", type="studio"):
    # Placeholder – in real case, pull from API or dataset
    if region == "france":
        return {"studio": 120000, "2bed": 220000, "3bed": 320000}[type]
    elif region == "dubai":
        return {"studio": 150000, "2bed": 300000, "3bed": 450000}[type]
    return None

def main():
    today = datetime.today().strftime("%Y-%m-%d")

    gold_price = fetch_yahoo_price("GC=F")        # Gold futures
    btc_price = fetch_yahoo_price("BTC-USD")      # Bitcoin in USD

    france_studio = fetch_real_estate_price("france", "studio")
    france_2bed = fetch_real_estate_price("france", "2bed")
    france_3bed = fetch_real_estate_price("france", "3bed")

    dubai_studio = fetch_real_estate_price("dubai", "studio")
    dubai_2bed = fetch_real_estate_price("dubai", "2bed")
    dubai_3bed = fetch_real_estate_price("dubai", "3bed")

    new_row = {
        "date": today,
        "gold_actual": gold_price,
        "bitcoin_actual": btc_price,
        "france_studio_actual": france_studio,
        "france_2bed_actual": france_2bed,
        "france_3bed_actual": france_3bed,
        "dubai_studio_actual": dubai_studio,
        "dubai_2bed_actual": dubai_2bed,
        "dubai_3bed_actual": dubai_3bed,
    }

    # Append to CSV
    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(ACTUAL_FILE, index=False)
    print("✅ Actual data updated:", new_row)

if __name__ == "__main__":
    main()
