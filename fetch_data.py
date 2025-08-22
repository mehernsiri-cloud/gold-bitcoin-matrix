import requests
import pandas as pd
from datetime import datetime
import os

DATA_FILE = "data/actual_data.csv"

def fetch_gold_price():
    url = "https://metals-api.com/api/latest?access_key=YOUR_API_KEY&base=USD&symbols=XAU"
    try:
        r = requests.get(url).json()
        return r['rates']['XAU']
    except:
        return None

def fetch_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        r = requests.get(url).json()
        return r['bitcoin']['usd']
    except:
        return None

def fetch_real_estate_france():
    return 4000  # €/m2 placeholder

def fetch_real_estate_dubai():
    return 13000  # AED/m2 placeholder

def save_actual_data():
    data = {
        "date": datetime.today().strftime("%Y-%m-%d"),
        "gold_actual": fetch_gold_price(),
        "bitcoin_actual": fetch_bitcoin_price(),
        "real_estate_france_actual": fetch_real_estate_france(),
        "real_estate_dubai_actual": fetch_real_estate_dubai()
    }
    df = pd.DataFrame([data])
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(DATA_FILE, index=False)

if __name__ == "__main__":
    save_actual_data()
