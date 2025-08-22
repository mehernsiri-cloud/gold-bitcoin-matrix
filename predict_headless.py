# predict_headless.py
import pandas as pd
import yfinance as yf
import os
from datetime import datetime

# ------------------------------
# Config
# ------------------------------
ASSETS = {
    "gold": "GC=F",        # Gold futures
    "bitcoin": "BTC-USD",  # Bitcoin
    "fr_real_estate": "^FCHI",  # proxy France RE via CAC40 real estate ETF / index
    "dubai_real_estate": "^DFMGI"  # placeholder index, replace if real data
}

CSV_FILE = "predictions_log.csv"

VOL_THRESHOLDS = {
    "gold": 0.005,
    "bitcoin": 0.02,
    "fr_real_estate": 0.01,
    "dubai_real_estate": 0.01
}

# ------------------------------
# Helper functions
# ------------------------------
def fetch_data(symbol):
    """Fetch last 30 days data"""
    df = yf.download(symbol, period="30d", interval="1d")
    return df

def compute_volatility(df):
    """Daily volatility as std of returns"""
    df['returns'] = df['Adj Close'].pct_change()
    vol = df['returns'].std()
    return vol

def predict_price(df):
    """Simple forecast: latest price adjusted by last return"""
    last_price = df['Adj Close'].iloc[-1]
    last_return = df['Adj Close'].pct_change().iloc[-1]
    predicted_price = last_price * (1 + last_return)
    return predicted_price

def compute_risk(vol, threshold):
    if vol < threshold:
        return "Low"
    elif vol < threshold * 2:
        return "Medium"
    else:
        return "High"

# ------------------------------
# Main
# ------------------------------
results = []

for asset, symbol in ASSETS.items():
    try:
        df = fetch_data(symbol)
        if df.empty:
            raise ValueError(f"No data fetched for {asset}")
        
        vol = compute_volatility(df)
        predicted_price = predict_price(df)
        risk = compute_risk(vol, VOL_THRESHOLDS[asset])
        
        results.append({
            "date": datetime.today().strftime("%Y-%m-%d"),
            "asset": asset,
            "predicted_price": round(predicted_price, 2),
            "volatility": round(vol, 4),
            "risk": risk
        })
    except Exception as e:
        print(f"Error fetching {asset}: {e}")

# ------------------------------
# Save to CSV (append)
# ------------------------------
df_results = pd.DataFrame(results)

if os.path.exists(CSV_FILE):
    df_old = pd.read_csv(CSV_FILE, parse_dates=["date"])
    df_results = pd.concat([df_old, df_results], ignore_index=True)

df_results.to_csv(CSV_FILE, index=False)
print(f"Saved daily predictions to {CSV_FILE}")
