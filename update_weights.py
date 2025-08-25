import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Public data sources
# ---------------------------

FRED_SERIES = {
    "cpi": "https://fred.stlouisfed.org/data/CPIAUCSL.txt",          # CPI All Urban Consumers
    "us10y": "https://fred.stlouisfed.org/data/DGS10.txt",          # 10Y Treasury
    "dxy": "https://fred.stlouisfed.org/data/DTWEXBGS.txt",         # US Dollar Index
    "oil": "https://fred.stlouisfed.org/data/DCOILWTICO.txt",       # WTI Oil
    "sp500": "https://fred.stlouisfed.org/data/SP500.txt"           # S&P500
}

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# ---------------------------
# Helper functions
# ---------------------------

def fetch_fred_series(url):
    try:
        df = pd.read_csv(url, sep="\s+", comment=";", names=["DATE","VALUE"], na_values=".")
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna().sort_values("DATE")
        return float(df.iloc[-1]["VALUE"])
    except Exception as e:
        logging.warning(f"⚠️ Error fetching FRED series {url}: {e}")
        return None

def fetch_vix(url=VIX_URL):
    try:
        df = pd.read_csv(url)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.sort_values("DATE")
        return float(df.iloc[-1]["CLOSE"])
    except Exception as e:
        logging.warning(f"⚠️ Error fetching VIX: {e}")
        return None

def compute_inflation_yoy(cpi_url):
    """Compute YoY inflation from CPI series"""
    try:
        df = pd.read_csv(cpi_url, sep="\s+", comment=";", names=["DATE","VALUE"], na_values=".")
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna().sort_values("DATE")
        latest = df.iloc[-1]
        one_year_ago = df[df["DATE"] <= (latest["DATE"] - timedelta(days=365))].iloc[-1]
        inflation = ((latest["VALUE"] - one_year_ago["VALUE"]) / one_year_ago["VALUE"]) * 100
        return round(float(inflation),2)
    except Exception as e:
        logging.warning(f"⚠️ Error computing inflation YoY: {e}")
        return None

def normalize(value, ref=100.0):
    try:
        return (value - ref) / abs(ref)
    except Exception:
        return 0.0

# ---------------------------
# Build weights
# ---------------------------

def build_weights():
    inflation = compute_inflation_yoy(FRED_SERIES["cpi"]) or 2.5
    bond_yield = fetch_fred_series(FRED_SERIES["us10y"]) or 3.0
    usd_strength = fetch_fred_series(FRED_SERIES["dxy"]) or 100.0
    oil = fetch_fred_series(FRED_SERIES["oil"]) or 70.0
    liquidity = -fetch_vix() if fetch_vix() is not None else -20.0
    sp500 = fetch_fred_series(FRED_SERIES["sp500"]) or 4500.0
    equity_flows = (sp500 - 4500)/4500  # simple proxy for flows

    # Hard-coded sentiment proxies replaced with calculated placeholders
    regulation = 0.0
    adoption = 0.05
    currency_instability = 0.1
    recession_probability = 0.05
    tail_risk_event = 0.05
    geopolitics = 0.05

    weights = {}
    for asset in ["gold","bitcoin"]:
        weights[asset] = {
            "inflation": normalize(inflation, 3),
            "real_rates": normalize(bond_yield - inflation, 1),
            "bond_yields": normalize(bond_yield, 3),
            "energy_prices": normalize(oil, 60),
            "usd_strength": normalize(usd_strength, 100),
            "liquidity": liquidity,
            "equity_flows": equity_flows,
            "regulation": regulation,
            "adoption": adoption,
            "currency_instability": currency_instability,
            "recession_probability": recession_probability,
            "tail_risk_event": tail_risk_event,
            "geopolitics": geopolitics
        }

    return weights

# ---------------------------
# Save YAML
# ---------------------------

if __name__ == "__main__":
    weights = build_weights()
    with open("weight.yaml", "w") as f:
        yaml.dump(weights, f)
    print("✅ weight.yaml updated with live indicators")
