# update_weights.py
import pandas as pd
import requests
import yaml
import numpy as np
from datetime import datetime

WEIGHT_FILE = "weight.yaml"

def fetch_eurostat_cpi():
    """Fetch Eurozone CPI from Eurostat (latest available monthly data)."""
    try:
        url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/prc_hicp_aind.csv"
        df = pd.read_csv(url, sep=';', encoding='latin1')
        df = df[df['geo\\time'] == 'EA19']  # Euro area
        df = df.drop(columns=['unit', 's_adj', 'freq', 'geo\\time'], errors='ignore')
        latest_value = pd.to_numeric(df.iloc[:, -1], errors='coerce')
        return latest_value.values[0] / 100  # convert to fraction
    except Exception as e:
        print(f"Error fetching Eurostat CPI: {e}")
        return None

def fetch_us_treasury_yields():
    """Fetch US Treasury daily yields CSV (10y & 3mo)."""
    try:
        url = "https://home.treasury.gov/sites/default/files/interest-rates/yield.xml"  # placeholder
        # simplified example: in production use treasury.gov CSV parsing
        # here we just mock values to avoid crashing
        ten_year = 0.045  # 4.5%
        three_month = 0.015  # 1.5%
        return ten_year, three_month
    except Exception as e:
        print(f"Error fetching US Treasury yields: {e}")
        return 0.04, 0.015

def fetch_energy_prices():
    """Fetch Brent crude price from EIA CSV."""
    try:
        url = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"
        df = pd.read_excel(url, sheet_name='Data 1', skiprows=2)
        latest_value = df.iloc[-1, 1]
        return float(latest_value)
    except Exception as e:
        print(f"Error fetching energy prices: {e}")
        return 80.0

def fetch_usd_strength():
    """Fetch USD index via ECB FX rates."""
    try:
        url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
        resp = requests.get(url)
        if resp.status_code == 200:
            # very simplified: assume USD/EUR rate in XML
            import xml.etree.ElementTree as ET
            tree = ET.fromstring(resp.content)
            ns = {'gesmes': 'http://www.gesmes.org/xml/2002-08-01',
                  'e': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'}
            cube = tree.find('.//e:Cube/e:Cube/e:Cube[@currency="USD"]', ns)
            rate = float(cube.attrib['rate'])
            return 1 / rate  # USD strength vs EUR
        return 1.0
    except Exception as e:
        print(f"Error fetching USD strength: {e}")
        return 1.0

def fetch_bitcoin_price():
    """Fetch Bitcoin price from CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        data = requests.get(url).json()
        return data['bitcoin']['usd']
    except Exception as e:
        print(f"Error fetching Bitcoin price: {e}")
        return 50000

def normalize(value, min_val=-1, max_val=1):
    """Normalize to [-1,1] scale."""
    return max(min(float(value), max_val), min_val)

def update_weights():
    # Fetch macro data
    cpi = fetch_eurostat_cpi() or 0.02
    ten_year, three_month = fetch_us_treasury_yields()
    real_rate = ten_year - cpi
    energy = fetch_energy_prices()
    usd_strength = fetch_usd_strength()
    
    # Prepare new weights
    weights = {
        "gold": {
            "inflation": normalize(cpi),
            "real_rates": normalize(real_rate),
            "bond_yields": normalize(ten_year),
            "energy_prices": normalize(energy/100),  # crude price scaled
            "usd_strength": normalize(usd_strength),
            "liquidity": 0.1,
            "equity_flows": -0.05,
            "regulation": 0.0,
            "adoption": 0.05,
            "currency_instability": 0.1,
            "recession_probability": 0.0,
            "tail_risk_event": 0.05,
            "geopolitics": 0.05
        },
        "bitcoin": {
            "inflation": normalize(cpi),
            "real_rates": normalize(real_rate),
            "bond_yields": normalize(ten_year),
            "energy_prices": normalize(energy/100),
            "usd_strength": normalize(usd_strength),
            "liquidity": 0.1,
            "equity_flows": -0.05,
            "regulation": 0.0,
            "adoption": 0.05,
            "currency_instability": 0.1,
            "recession_probability": 0.0,
            "tail_risk_event": 0.05,
            "geopolitics": 0.05
        }
    }

    # Write YAML
    with open(WEIGHT_FILE, "w") as f:
        yaml.safe_dump(weights, f, sort_keys=False)
    print("weight.yaml updated successfully.")

if __name__ == "__main__":
    update_weights()
