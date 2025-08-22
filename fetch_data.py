import yfinance as yf
import pandas as pd
import os
from datetime import datetime

DATA_DIR = "data"
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")

# Ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_prices():
    """Fetch latest Gold and Bitcoin prices"""
    prices = {}
    try:
        gold = yf.Ticker("GC=F").history(period="1d")['Close'].iloc[-1]
        btc = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
        prices["Gold"] = gold
        prices["Bitcoin"] = btc
    except Exception as e:
        print(f"Error fetching prices: {e}")
        prices["Gold"] = None
        prices["Bitcoin"] = None
    return prices

def fetch_indicators():
    """Fetch key macro indicators for prediction"""
    indicators = {}

    # Inflation (CPI YoY)
    try:
        cpi = yf.Ticker("^CPI").history(period="1d")['Close'].iloc[-1]
        indicators['inflation'] = float(cpi)
    except:
        indicators['inflation'] = 0.03

    # USD strength (DXY)
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="1d")['Close'].iloc[-1]
        indicators['usd_strength'] = float(dxy)
    except:
        indicators['usd_strength'] = 1.0

    # Real rates (US 10Y - CPI)
    try:
        tn10 = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]/100
        indicators['real_rates'] = tn10 - indicators['inflation']
    except:
        indicators['real_rates'] = 0.01

    # Bond yields
    try:
        indicators['bond_yields'] = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]/100
    except:
        indicators['bond_yields'] = 0.03

    # Energy prices (WTI Crude)
    try:
        indicators['energy_prices'] = yf.Ticker("CL=F").history(period="1d")['Close'].iloc[-1]
    except:
        indicators['energy_prices'] = 70.0

    # Tail risk (VIX)
    try:
        indicators['tail_risk_event'] = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
    except:
        indicators['tail_risk_event'] = 20.0

    # Default/fixed values for remaining indicators
    indicators.update({
        'geopolitics': 0.1,
        'equity_flows': 0.01,
        'regulation': 0.0,
        'adoption': 0.1,
        'currency_instability': 0.02,
        'recession_probability': 0.05,
        'liquidity': 0.05
    })

    return indicators

def save_actual_data():
    prices = fetch_prices()
    indicators = fetch_indicators()
    timestamp = datetime.utcnow()

    # Save to CSV
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "gold_actual": prices.get("Gold"),
        "bitcoin_actual": prices.get("Bitcoin"),
        **indicators
    }])
    if os.path.exists(ACTUAL_FILE):
        df.to_csv(ACTUAL_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(ACTUAL_FILE, index=False)
    print("Actual data saved.")

if __name__ == "__main__":
    save_actual_data()
