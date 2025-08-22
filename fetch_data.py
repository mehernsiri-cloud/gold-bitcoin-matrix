import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import os
import yaml

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# --- Function to fetch live indicators ---
def fetch_indicators():
    # Example: real APIs or placeholder fetching
    indicators = {}

    # Inflation: Using FRED CSV endpoint (example)
    try:
        inflation = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL")['VALUE'].iloc[-1]
        indicators['inflation'] = float(inflation)
    except:
        indicators['inflation'] = 0.03

    # Real Rates: Treasury yields - inflation (example placeholder)
    try:
        real_rate = 0.01  # Example
        indicators['real_rates'] = real_rate
    except:
        indicators['real_rates'] = 0.01

    # USD Strength
    try:
        usd_index = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS")['VALUE'].iloc[-1]
        indicators['usd_strength'] = float(usd_index)
    except:
        indicators['usd_strength'] = 1.0

    # Liquidity (M2)
    indicators['liquidity'] = 0.05

    # Equity flows
    indicators['equity_flows'] = 0.01

    # Bond yields (10Y)
    indicators['bond_yields'] = 0.03

    # Regulation
    indicators['regulation'] = 0.0

    # Adoption
    indicators['adoption'] = 0.1

    # Currency instability
    indicators['currency_instability'] = 0.02

    # Recession probability
    indicators['recession_probability'] = 0.05

    # Energy prices (WTI Crude)
    try:
        energy = yf.Ticker("CL=F").history(period="1d")['Close'].iloc[-1]
        indicators['energy_prices'] = float(energy)
    except:
        indicators['energy_prices'] = 70.0

    # Tail risk event
    indicators['tail_risk_event'] = 0.1

    return indicators

# --- Function to fetch live prices ---
def fetch_prices():
    prices = {}
    for asset, ticker in {"Gold": "GC=F", "Bitcoin": "BTC-USD"}.items():
        try:
            data = yf.Ticker(ticker).history(period="1d")
            prices[asset] = float(data['Close'].iloc[-1])
        except:
            prices[asset] = None
    return prices

# --- Save actual data ---
def save_actual_data():
    today = datetime.today().strftime("%Y-%m
