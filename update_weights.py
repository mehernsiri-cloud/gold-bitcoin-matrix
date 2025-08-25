# update_weights.py
import pandas as pd
import requests
import yaml
import numpy as np
from datetime import datetime

WEIGHT_FILE = "weight.yaml"

# ------------------------------
# FRED SERIES FETCHER
# ------------------------------
def fetch_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url, skiprows=10)  # skip metadata rows
        if "DATE" not in df.columns or "VALUE" not in df.columns:
            raise ValueError("Unexpected CSV format, missing DATE or VALUE columns")
        df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
        df = df[df['VALUE'].notna()]
        last_value = df['VALUE'].iloc[-1]
        return float(last_value)
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return 0.0

# ------------------------------
# CoinGecko BTC FETCHER
# ------------------------------
def fetch_bitcoin_trend():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7"
        r = requests.get(url)
        data = r.json()
        prices = [p[1] for p in data['prices']]
        trend = (prices[-1] - prices[0]) / prices[0]  # 7-day % change
        return float(trend)
    except:
        return 0.0

# ------------------------------
# ECB FX USD INDEX
# ------------------------------
def fetch_usd_index():
    try:
        # ECB publishes EUR/USD as rates, invert to USD/EUR
        url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
        r = requests.get(url)
        from xml.etree import ElementTree as ET
        tree = ET.fromstring(r.content)
        namespaces = {"gesmes": "http://www.gesmes.org/xml/2002-08-01", "e": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
        usd_rate = None
        for cube in tree.findall(".//e:Cube/e:Cube/e:Cube", namespaces):
            if cube.attrib['currency'] == 'USD':
                usd_rate = float(cube.attrib['rate'])
        if usd_rate:
            usd_index = 1 / usd_rate  # USD strength
            return float(usd_index)
        return 0.0
    except:
        return 0.0

# ------------------------------
# NORMALIZATION FUNCTION
# ------------------------------
def normalize(val, min_val=-1, max_val=1, scale=None):
    """Normalize value to [-1,1] or [0,1] depending on scale"""
    if scale:
        # scale = (min_val, max_val)
        min_s, max_s = scale
        norm = (val - min_s) / (max_s - min_s)
        return max(0.0, min(1.0, norm))
    # default [-1,1]
    return max(-1.0, min(1.0, val))

# ------------------------------
# UPDATE WEIGHTS
# ------------------------------
def update_weights():
    weights = {}

    # GOLD
    weights['gold'] = {}

    # Macro indicators from FRED
    cpi = fetch_fred_series("CPIAUCSL")          # US CPI
    t10 = fetch_fred_series("GS10")              # 10Y treasury
    tb3 = fetch_fred_series("TB3MS")             # 3M treasury
    ppi = fetch_fred_series("PPIACO")            # PPI
    oil = fetch_fred_series("DCOILBRENTEU")      # Brent oil
    if oil == 0.0:  # fallback crude oil from FRED
        oil = fetch_fred_series("DCOILWTICO")
    
    # Inflation and real rates
    inflation = normalize(cpi/300.0)             # rough scaling
    real_rates = normalize(t10 - cpi)           # nominal - inflation
    bond_yields = normalize(t10)
    energy_prices = normalize(oil/100.0)

    weights['gold']['inflation'] = inflation
    weights['gold']['real_rates'] = real_rates
    weights['gold']['bond_yields'] = bond_yields
    weights['gold']['energy_prices'] = energy_prices

    # Other factors: random or neutral for now (replace with news-based in future)
    weights['gold']['geopolitics'] = 0.1
    weights['gold']['usd_strength'] = normalize(fetch_usd_index())
    weights['gold']['liquidity'] = 0.05
    weights['gold']['equity_flows'] = 0.0
    weights['gold']['regulation'] = 0.0
    weights['gold']['adoption'] = 0.0
    weights['gold']['currency_instability'] = 0.05
    weights['gold']['recession_probability'] = 0.0
    weights['gold']['tail_risk_event'] = 0.05

    # BITCOIN
    weights['bitcoin'] = {}
    btc_trend = fetch_bitcoin_trend()

    weights['bitcoin']['inflation'] = inflation
    weights['bitcoin']['real_rates'] = real_rates
    weights['bitcoin']['bond_yields'] = bond_yields
    weights['bitcoin']['energy_prices'] = energy_prices
    weights['bitcoin']['geopolitics'] = 0.05
    weights['bitcoin']['usd_strength'] = normalize(fetch_usd_index())
    weights['bitcoin']['liquidity'] = 0.1
    weights['bitcoin']['equity_flows'] = 0.1
    weights['bitcoin']['regulation'] = 0.05
    weights['bitcoin']['adoption'] = 0.1
    weights['bitcoin']['currency_instability'] = 0.05
    weights['bitcoin']['recession_probability'] = 0.0
    weights['bitcoin']['tail_risk_event'] = btc_trend

    # Save YAML (convert all np.float64 to float)
    def convert(obj):
        if isinstance(obj, np.generic):
            return float(obj)
        return obj

    weights_clean = {k:{ik:convert(iv) for ik, iv in v.items()} for k,v in weights.items()}

    with open(WEIGHT_FILE, "w") as f:
        yaml.safe_dump(weights_clean, f, sort_keys=False)

    print(f"Updated weights.yaml at {datetime.now()}")
    print(weights_clean)


if __name__ == "__main__":
    update_weights()
