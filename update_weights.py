import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FRED_SERIES = {
    "cpi": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
    "us10y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
    "dxy": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS",
    "oil": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO",
    "sp500": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
}

def fetch_csv_last_value(url, value_col="VALUE"):
    try:
        df = pd.read_csv(url)
        if value_col not in df.columns:
            df = df.iloc[:, :2]
            df.columns = ["DATE","VALUE"]
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna().sort_values("DATE")
        return float(df.iloc[-1][value_col])
    except Exception as e:
        logging.warning(f"⚠️ Error fetching CSV {url}: {e}")
        return None

def compute_inflation_yoy(cpi_url):
    try:
        df = pd.read_csv(cpi_url)
        if "VALUE" not in df.columns:
            df = df.iloc[:, :2]
            df.columns = ["DATE","VALUE"]
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna().sort_values("DATE")
        latest = df.iloc[-1]
        one_year_ago = df[df["DATE"] <= (latest["DATE"] - timedelta(days=365))].iloc[-1]
        inflation = ((latest["VALUE"] - one_year_ago["VALUE"])/one_year_ago["VALUE"])*100
        return round(float(inflation),2)
    except Exception as e:
        logging.warning(f"⚠️ Error computing inflation YoY: {e}")
        return None

def normalize(value, ref=100.0):
    try:
        return (value - ref)/abs(ref)
    except Exception:
        return 0.0

def normalize_range(value, min_val=-0.2, max_val=0.2, value_min=0, value_max=50):
    """Scale raw value into a small ±0.2 range for safe impact"""
    if value is None:
        return 0.0
    value_clamped = max(min(value, value_max), value_min)
    scaled = (value_clamped - value_min) / (value_max - value_min) * (max_val - min_val) + min_val
    return round(scaled, 3)

# Asset-specific sensitivities
SENSITIVITY = {
    "gold": {
        "inflation": 1.0,
        "real_rates": -1.0,
        "bond_yields": -0.5,
        "energy_prices": 0.3,
        "usd_strength": -0.8,
        "liquidity": 0.5,
        "equity_flows": 0.2,
        "regulation": 0.0,
        "adoption": 0.0,
        "currency_instability": 0.3,
        "recession_probability": 0.5,
        "tail_risk_event": 0.7,
        "geopolitics": 0.6
    },
    "bitcoin": {
        "inflation": 0.2,
        "real_rates": -0.2,
        "bond_yields": -0.3,
        "energy_prices": 0.1,
        "usd_strength": -0.3,
        "liquidity": 0.8,
        "equity_flows": 0.7,
        "regulation": -0.2,
        "adoption": 0.2,
        "currency_instability": 0.5,
        "recession_probability": 0.2,
        "tail_risk_event": 0.1,
        "geopolitics": 0.1
    }
}

def calc_sentiment_proxies(sp500_change, vix_level, usd_change, oil_change, inflation):
    regulation = normalize_range(vix_level, -0.2, 0.2, 10, 40)
    adoption = max(min(sp500_change, 0.1), -0.1)
    currency_instability = -usd_change
    recession_probability = min(max((sp500_change - inflation/100.0)/2.0, 0.0), 1.0)
    tail_risk_event = normalize_range(vix_level, -0.2, 0.2, 10, 60)
    geopolitics = normalize_range(oil_change*100, -0.2, 0.2, -10, 10)
    return regulation, adoption, currency_instability, recession_probability, tail_risk_event, geopolitics

def build_weights():
    inflation = compute_inflation_yoy(FRED_SERIES["cpi"]) or 2.5
    bond_yield = fetch_csv_last_value(FRED_SERIES["us10y"]) or 3.0
    usd_index = fetch_csv_last_value(FRED_SERIES["dxy"]) or 100.0
    oil_price = fetch_csv_last_value(FRED_SERIES["oil"]) or 70.0
    sp500 = fetch_csv_last_value(FRED_SERIES["sp500"]) or 4500.0
    vix = fetch_csv_last_value(FRED_SERIES["vix"], value_col="CLOSE") or 20.0

    sp500_change = (sp500 - 4500)/4500
    usd_change = (usd_index - 100)/100
    oil_change = (oil_price - 70)/70
    vix_level = vix

    regulation, adoption, currency_instability, recession_probability, tail_risk_event, geopolitics = \
        calc_sentiment_proxies(sp500_change, vix_level, usd_change, oil_change, inflation)

    liquidity = normalize_range(vix_level, -0.2, 0.2, 10, 40)

    weights = {}
    for asset in ["gold","bitcoin"]:
        sens = SENSITIVITY[asset]
        weights[asset] = {
            "inflation": normalize(inflation, 3) * sens["inflation"],
            "real_rates": normalize(bond_yield - inflation, 1) * sens["real_rates"],
            "bond_yields": normalize(bond_yield, 3) * sens["bond_yields"],
            "energy_prices": normalize(oil_price, 60) * sens["energy_prices"],
            "usd_strength": normalize(usd_index, 100) * sens["usd_strength"],
            "liquidity": liquidity * sens["liquidity"],
            "equity_flows": sp500_change * sens["equity_flows"],
            "regulation": regulation * sens["regulation"],
            "adoption": adoption * sens["adoption"],
            "currency_instability": currency_instability * sens["currency_instability"],
            "recession_probability": recession_probability * sens["recession_probability"],
            "tail_risk_event": tail_risk_event * sens["tail_risk_event"],
            "geopolitics": geopolitics * sens["geopolitics"]
        }

    return weights

if __name__ == "__main__":
    weights = build_weights()
    with open("weight.yaml", "w") as f:
        yaml.dump(weights,f)
    print("✅ weight.yaml updated with live indicators")
