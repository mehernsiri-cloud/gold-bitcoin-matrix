import requests
import pandas as pd
from datetime import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

def fetch_gold_price():
    # Replace with your API key if needed
    return 1950  # placeholder

def fetch_bitcoin_price():
    return 30000  # placeholder

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
    if os.path.exists(ACTUAL_FILE):
        df.to_csv(ACTUAL_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(ACTUAL_FILE, index=False)

if __name__=="__main__":
    save_actual_data()
