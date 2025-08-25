# update_weights.py
import yaml
import os
import numpy as np
from datetime import datetime
from fredapi import Fred
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from pycoingecko import CoinGeckoAPI

# ------------------------------
# CONFIG
# ------------------------------
WEIGHT_FILE = "weight.yaml"

# API KEYS
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY")
FRED_KEY = os.environ.get("FRED_KEY")

# ------------------------------
# FETCH DATA FUNCTIONS
# ------------------------------

# --- FRED API ---
fred = Fred(api_key=FRED_KEY)

def fetch_inflation():
    """US CPI YoY Inflation"""
    try:
        cpi_series = fred.get_series('CPIAUCSL')  # US CPI
        inflation = (cpi_series[-1] - cpi_series[-12]) / cpi_series[-12]  # YoY
        return float(np.clip(inflation, -0.2, 0.2))  # normalize
    except:
        return 0.0

def fetch_bond_yield():
    """US 10Y Treasury Yield"""
    try:
        yield_10y = fred.get_series('DGS10')[-1]
        return float(np.clip(yield_10y / 100, -0.2, 0.2))  # normalize to [-0.2,0.2]
    except:
        return 0.0

def fetch_recession_prob():
    """US Recession Probability (GDP gap approximation)"""
    try:
        gdp = fred.get_series('GDPC1')[-1]
        gdp_prev = fred.get_series('GDPC1')[-4]  # 1 year ago
        gap = (gdp - gdp_prev) / gdp_prev
        prob = -gap  # negative growth => higher recession probability
        return float(np.clip(prob, -0.2, 0.2))
    except:
        return 0.0

# --- Alpha Vantage ---
fx = ForeignExchange(key=ALPHA_VANTAGE_KEY)
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

def fetch_usd_strength():
    """USD vs EUR"""
    try:
        data, _ = fx.get_currency_exchange_rate(from_currency='USD', to_currency='EUR')
        usd_rate = float(data['5. Exchange Rate'])
        # normalize USD strength around 1.0
        strength = (usd_rate - 1.0) / 0.1
        return float(np.clip(strength, -1.0, 1.0))
    except:
        return 0.0

def fetch_equity_flows():
    """S&P500 % change today"""
    try:
        sp_data, _ = ts.get_daily(symbol='SPY', outputsize='compact')
        today = sp_data['4. close'].iloc[-1]
        prev = sp_data['4. close'].iloc[-2]
        change = (today - prev) / prev
        return float(np.clip(change, -0.2, 0.2))
    except:
        return 0.0

# --- CoinGecko ---
cg = CoinGeckoAPI()

def fetch_btc_price():
    try:
        price = cg.get_price(ids='bitcoin', vs_currencies='usd')['bitcoin']['usd']
        return float(price)
    except:
        return 0.0

# ------------------------------
# MAIN UPDATE FUNCTION
# ------------------------------
def update_weights():
    # Load existing weights
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, 'r') as f:
            weights = yaml.safe_load(f)
    else:
        weights = {"gold": {}, "bitcoin": {}}

    # --- Update gold indicators dynamically ---
    weights['gold']['inflation'] = fetch_inflation()
    weights['gold']['bond_yields'] = fetch_bond_yield()
    weights['gold']['usd_strength'] = fetch_usd_strength()
    weights['gold']['equity_flows'] = fetch_equity_flows()
    weights['gold']['recession_probability'] = fetch_recession_prob()
    # Other indicators can be static or updated via news/trend separately
    static_gold = ['geopolitics','real_rates','liquidity','regulation','adoption','currency_instability','energy_prices','tail_risk_event']
    for k in static_gold:
        if k not in weights['gold']:
            weights['gold'][k] = 0.0

    # --- Update bitcoin indicators dynamically ---
    weights['bitcoin']['inflation'] = fetch_inflation()
    weights['bitcoin']['bond_yields'] = fetch_bond_yield()
    weights['bitcoin']['usd_strength'] = fetch_usd_strength()
    weights['bitcoin']['equity_flows'] = fetch_equity_flows()
    weights['bitcoin']['recession_probability'] = fetch_recession_prob()
    btc_price = fetch_btc_price()
    # Other indicators
    static_btc = ['geopolitics','real_rates','liquidity','regulation','adoption','currency_instability','energy_prices','tail_risk_event']
    for k in static_btc:
        if k not in weights['bitcoin']:
            weights['bitcoin'][k] = 0.0
    weights['bitcoin']['btc_price'] = btc_price

    # Convert all np.float64 to native float to avoid YAML errors
    def floatify(d):
        for k,v in d.items():
            if isinstance(v, dict):
                floatify(v)
            else:
                d[k] = float(v)
    floatify(weights)

    # Save updated weights
    with open(WEIGHT_FILE, 'w') as f:
        yaml.safe_dump(weights, f, sort_keys=False)

    print(f"✅ Updated weights at {datetime.now()}")
    print(weights)

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    update_weights()
