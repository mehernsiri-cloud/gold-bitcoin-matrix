# update_weights.py
import pandas as pd
import numpy as np
import yaml
import requests
from datetime import datetime

# ------------------------------
# UTILS: Fetch latest FRED series value
# ------------------------------
def fetch_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df[df['VALUE'].notna()]
        last_value = df['VALUE'].iloc[-1]
        return float(last_value)
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return 0.0

# ------------------------------
# UTILS: Fetch Bitcoin price (CoinGecko)
# ------------------------------
def fetch_btc_price():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        price = r.json()["bitcoin"]["usd"]
        return float(price)
    except Exception as e:
        print(f"Error fetching BTC price: {e}")
        return 0.0

# ------------------------------
# UTILS: Fetch EUR/USD FX rate (ECB)
# ------------------------------
def fetch_fx_rate():
    try:
        r = requests.get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml")
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(r.content)
        namespaces = {"gesmes": "http://www.gesmes.org/xml/2002-08-01", "e": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
        cube = tree.find(".//e:Cube/e:Cube/e:Cube[@currency='USD']", namespaces)
        return float(cube.attrib["rate"])
    except Exception as e:
        print(f"Error fetching EUR/USD FX rate: {e}")
        return 0.0

# ------------------------------
# NORMALIZE INDICATORS
# ------------------------------
def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.0
    return max(min((value - min_val) / (max_val - min_val), 1.0), -1.0)

# ------------------------------
# UPDATE WEIGHTS FUNCTION
# ------------------------------
def update_weights():
    weights = {"gold": {}, "bitcoin": {}}

    # ----- Realistic macro indicators -----
    # Inflation: CPIAUCSL (US CPI)
    inflation = fetch_fred_series("CPIAUCSL")
    weights['gold']['inflation'] = float(normalize(inflation, 0, 500))

    # Real rates: 3-mo Treasury (TB3MS) minus inflation
    short_rate = fetch_fred_series("TB3MS")
    real_rate = short_rate - inflation
    weights['gold']['real_rates'] = float(normalize(real_rate, -10, 10))

    # Bond yields: 10-year Treasury GS10
    bond_yields = fetch_fred_series("GS10")
    weights['gold']['bond_yields'] = float(normalize(bond_yields, 0, 20))

    # Energy prices: PPIACO (Commodities PPI)
    energy_prices = fetch_fred_series("PPIACO")
    weights['gold']['energy_prices'] = float(normalize(energy_prices, 0, 500))

    # USD strength: based on EUR/USD
    fx_rate = fetch_fx_rate()
    weights['gold']['usd_strength'] = float(normalize(1/fx_rate, 0.5, 2.0))  # stronger USD => higher

    # Liquidity & equity_flows & other market indicators (random placeholders)
    weights['gold']['liquidity'] = float(np.random.uniform(-0.2,0.2))
    weights['gold']['equity_flows'] = float(np.random.uniform(-0.2,0.2))

    # News/trend-driven indicators (random placeholders)
    weights['gold']['geopolitics'] = float(np.random.uniform(-0.2,0.2))
    weights['gold']['regulation'] = float(np.random.uniform(-0.2,0.2))
    weights['gold']['adoption'] = float(np.random.uniform(-0.2,0.2))
    weights['gold']['currency_instability'] = float(np.random.uniform(-0.2,0.2))
    weights['gold']['recession_probability'] = float(np.random.uniform(-0.2,0.2))
    weights['gold']['tail_risk_event'] = float(np.random.uniform(-0.2,0.2))

    # ----- Bitcoin weights -----
    # BTC-specific macro indicators (some same as gold)
    weights['bitcoin']['inflation'] = weights['gold']['inflation']
    weights['bitcoin']['real_rates'] = weights['gold']['real_rates']
    weights['bitcoin']['bond_yields'] = weights['gold']['bond_yields']
    weights['bitcoin']['energy_prices'] = weights['gold']['energy_prices']
    weights['bitcoin']['usd_strength'] = weights['gold']['usd_strength']
    weights['bitcoin']['liquidity'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['equity_flows'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['geopolitics'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['regulation'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['adoption'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['currency_instability'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['recession_probability'] = float(np.random.uniform(-0.2,0.2))
    weights['bitcoin']['tail_risk_event'] = float(np.random.uniform(-0.2,0.2))

    # ------------------------------
    # Save to YAML
    # ------------------------------
    with open("weight.yaml", "w") as f:
        yaml.safe_dump(weights, f, sort_keys=False)

    print("✅ weight.yaml updated successfully with live market indicators.")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    update_weights()
