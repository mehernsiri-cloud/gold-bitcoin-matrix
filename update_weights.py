# update_weights.py
import requests
import pandas as pd
import yaml
import os
from datetime import datetime
import numpy as np

WEIGHT_FILE = "weight.yaml"

# ---------------------------
# UTILITIES
# ---------------------------
def fetch_fred_series(series_id):
    """Fetches last available value from FRED CSV."""
    url = f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv"
    try:
        df = pd.read_csv(url)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE')
        last_value = df['VALUE'].iloc[-1]
        return float(last_value)
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return 0.0

def fetch_ecb_fx_rate(pair="USD/EUR"):
    """Fetch USD/EUR rate from ECB."""
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
    try:
        resp = requests.get(url)
        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.content)
        ns = {'gesmes': 'http://www.gesmes.org/xml/2002-08-01',
              'ecb': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'}
        cubes = root.findall('.//ecb:Cube/ecb:Cube/ecb:Cube', ns)
        rates = {c.attrib['currency']: float(c.attrib['rate']) for c in cubes}
        usd_eur = 1 / rates['USD']  # ECB reports EUR base
        return usd_eur
    except Exception as e:
        print(f"Error fetching ECB FX: {e}")
        return 1.0

def fetch_bitcoin_price():
    """Fetch current BTC price in USD from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        resp = requests.get(url)
        data = resp.json()
        return float(data['bitcoin']['usd'])
    except Exception as e:
        print(f"Error fetching BTC price: {e}")
        return 0.0

def normalize(value, min_val=-0.2, max_val=0.2):
    """Clamp values to a range."""
    return max(min(value, max_val), min_val)

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def update_weights():
    weights = {"gold": {}, "bitcoin": {}}

    # ---------------------------
    # GOLD
    # ---------------------------
    weights['gold']['inflation'] = normalize(fetch_fred_series('CPIAUCSL') / 5000)
    weights['gold']['real_rates'] = normalize((fetch_fred_series('TB3MS') - fetch_fred_series('CPIAUCSL')) / 100)
    weights['gold']['bond_yields'] = normalize(fetch_fred_series('GS10') / 100)
    weights['gold']['energy_prices'] = normalize(fetch_fred_series('PPIACO') / 100)
    weights['gold']['usd_strength'] = normalize(fetch_ecb_fx_rate() - 1.0)

    # Other indicators (temporary heuristics)
    for k in ['geopolitics', 'liquidity', 'equity_flows', 'regulation', 'adoption',
              'currency_instability', 'recession_probability', 'tail_risk_event']:
        weights['gold'][k] = normalize(np.random.uniform(-0.2, 0.2))

    # ---------------------------
    # BITCOIN
    # ---------------------------
    btc_price = fetch_bitcoin_price()
    weights['bitcoin']['adoption'] = normalize(btc_price / 100000)
    weights['bitcoin']['liquidity'] = normalize(btc_price / 50000 - 0.5)
    weights['bitcoin']['usd_strength'] = normalize(fetch_ecb_fx_rate() - 1.0)
    weights['bitcoin']['inflation'] = normalize(fetch_fred_series('CPIAUCSL') / 5000)
    weights['bitcoin']['real_rates'] = normalize((fetch_fred_series('TB3MS') - fetch_fred_series('CPIAUCSL')) / 100)
    weights['bitcoin']['bond_yields'] = normalize(fetch_fred_series('GS10') / 100)
    weights['bitcoin']['energy_prices'] = normalize(fetch_fred_series('PPIACO') / 100)

    for k in ['geopolitics', 'regulation', 'currency_instability', 'recession_probability',
              'tail_risk_event', 'equity_flows']:
        weights['bitcoin'][k] = normalize(np.random.uniform(-0.2, 0.2))

    # ---------------------------
    # ENSURE ALL FLOATS ARE NATIVE PYTHON
    # ---------------------------
    for asset in weights:
        for key in weights[asset]:
            val = weights[asset][key]
            if isinstance(val, (np.float64, np.float32)):
                weights[asset][key] = float(val)

    # ---------------------------
    # SAVE TO YAML
    # ---------------------------
    with open(WEIGHT_FILE, "w") as f:
        yaml.safe_dump(weights, f, sort_keys=False)

    print("Weights updated successfully!")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    update_weights()
