# update_weights.py
import pandas as pd
import requests
import yaml
import numpy as np
import io
import xml.etree.ElementTree as ET

WEIGHT_FILE = "weight.yaml"

# ----------------------------
# FRED fetcher (robust)
# ----------------------------
def fetch_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        lines = r.text.splitlines()
        # Find header line
        header_index = next((i for i, line in enumerate(lines) if 'DATE' in line and 'VALUE' in line), None)
        if header_index is None:
            raise ValueError("No header line found with DATE and VALUE")
        df = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
        df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
        df = df[df['VALUE'].notna()]
        if df.empty:
            return 0.0
        return float(df['VALUE'].iloc[-1])
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return 0.0

# ----------------------------
# CoinGecko fetcher for BTC price
# ----------------------------
def fetch_coingecko_price(coin_id="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return float(r.json()[coin_id]['usd'])
    except Exception as e:
        print(f"Error fetching CoinGecko price {coin_id}: {e}")
        return 0.0

# ----------------------------
# ECB FX rates fetcher (USD index proxy)
# ----------------------------
def fetch_ecb_usd_index():
    try:
        url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"gesmes":"http://www.gesmes.org/xml/2002-08-01","ecb":"http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
        usd_rate = None
        for cube in root.findall(".//ecb:Cube[@currency='USD']", ns):
            usd_rate = float(cube.attrib['rate'])
        return usd_rate or 0.0
    except Exception as e:
        print(f"Error fetching ECB USD rate: {e}")
        return 0.0

# ----------------------------
# Normalization helper [-1,1]
# ----------------------------
def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return 2 * ((value - min_val) / (max_val - min_val)) - 1

# ----------------------------
# Update weights
# ----------------------------
def update_weights():
    weights = {"gold": {}, "bitcoin": {}}

    # ----------------------------
    # Macro indicators (live)
    # ----------------------------
    inflation = fetch_fred_series("CPIAUCSL")          # US CPI
    real_rate = fetch_fred_series("TB3MS")             # 3-mo T-bill
    bond_yield = fetch_fred_series("GS10")             # 10-yr yield
    energy_price = fetch_fred_series("DCOILBRENTEU")   # Brent oil

    # Normalize values
    weights_macro = {
        "inflation": normalize(inflation, 0, 15),
        "real_rates": normalize(real_rate, -5, 15),
        "bond_yields": normalize(bond_yield, 0, 15),
        "energy_prices": normalize(energy_price, 0, 300),
    }

    # ----------------------------
    # Other indicators (simplified)
    # ----------------------------
    usd_strength = fetch_ecb_usd_index()
    weights_other = {
        "usd_strength": normalize(usd_strength, 0.8, 1.4),
        "liquidity": 0.1,
        "equity_flows": -0.05,
        "regulation": 0.0,
        "adoption": 0.05,
        "currency_instability": 0.1,
        "recession_probability": 0.0,
        "tail_risk_event": 0.05,
        "geopolitics": 0.05
    }

    # ----------------------------
    # Assign weights
    # ----------------------------
    for k,v in {**weights_macro, **weights_other}.items():
        weights["gold"][k] = float(v)
        weights["bitcoin"][k] = float(v)

    # ----------------------------
    # Save YAML
    # ----------------------------
    try:
        with open(WEIGHT_FILE, "w") as f:
            yaml.safe_dump(weights, f, sort_keys=False)
        print(f"Updated weights saved to {WEIGHT_FILE}")
    except Exception as e:
        print(f"Error saving YAML: {e}")

if __name__ == "__main__":
    update_weights()
