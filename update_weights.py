# update_weights.py
import pandas as pd
import numpy as np
import yaml
import requests
from io import StringIO
from datetime import datetime

# ------------------------------
# UTILS: Fetch latest FRED series value (robust)
# ------------------------------
def fetch_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        csv_data = StringIO(r.text)

        # skip lines until header is found
        for i, line in enumerate(csv_data):
            if line.startswith("DATE"):
                header_line = i
                break
        csv_data.seek(0)
        df = pd.read_csv(csv_data, skiprows=header_line)
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df[df['VALUE'].notna()]
        last_value = df['VALUE'].iloc[-1]
        return float(last_value)
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return 0.0

# ------------------------------
# Fetch Bitcoin price (CoinGecko)
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
# Fetch EUR/USD FX rate (ECB)
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
# NORMALIZE
# ------------------------------
def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.0
    return max(min((value - min_val) / (max_val - min_val), 1.0), -1.0)

# ------------------------------
# UPDATE WEIGHTS
# ------------------------------
def update_weights():
    weights = {"gold": {}, "bitcoin": {}}

    # --- Macro indicators ---
    inflation = fetch_fred_series("CPIAUCSL")
    short_rate = fetch_fred_series("TB3MS")
    bond_yields = fetch_fred_series("GS10")
    energy_prices = fetch_fred_series("PPIACO")

    real_rate = short_rate - inflation
    fx_rate = fetch_fx_rate()

    # Normalize realistic ranges
    weights['gold']['inflation'] = float(normalize(inflation, 0, 500))
    weights['gold']['real_rates'] = float(normalize(real_rate, -10, 10))
    weights['gold']['bond_yields'] = float(normalize(bond_yields, 0, 20))
    weights['gold']['energy_prices'] = float(normalize(energy_prices, 0, 500))
    weights['gold']['usd_strength'] = float(normalize(1/fx_rate, 0.5, 2.0))

    # Random placeholders for other indicators
    for key in ['liquidity','equity_flows','geopolitics','regulation','adoption','currency_instability','recession_probability','tail_risk_event']:
        weights['gold'][key] = float(np.random.uniform(-0.2,0.2))
        weights['bitcoin'][key] = float(np.random.uniform(-0.2,0.2))

    # Copy macro indicators to Bitcoin
    for key in ['inflation','real_rates','bond_yields','energy_prices','usd_strength']:
        weights['bitcoin'][key] = weights['gold'][key]

    # Save YAML
    with open("weight.yaml", "w") as f:
        yaml.safe_dump(weights, f, sort_keys=False)

    print("✅ weight.yaml updated successfully.")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    update_weights()
