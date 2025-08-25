import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Public data sources (all CSV)
# ---------------------------

FRED_SERIES = {
    "cpi": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
    "us10y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
    "dxy": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS",
    "oil": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO",
    "sp500": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
}

# ---------------------------
# Helper functions
# ---------------------------

def fetch_csv_last_value(url, value_col="VALUE"):
    try:
        df = pd.read_csv(url)
        if value_col not in df.columns:
            # fallback for non-standard CSVs
            df = df.iloc[:, :2]
            df.columns = ["DATE","VALUE"]
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna().sort_values("DATE")
        return float(df.iloc[-1][value_col])
    except Exception as e:
        logging.warning(f"⚠️ Error fetching CSV {url}: {e}")
        return None

def compute_inflation_yoy(cpi_url):
    """Compute YoY inflation from CPI series"""
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

# ---------------------------
# Proxy calculations for sentiment indicators
# ---------------------------

def calc_sentiment_proxies(sp500_change, vix_level, usd_change, oil_change, inflation):
    """
    Simple dynamic proxies for non-macro indicators:
      regulation → proxy by VIX increase
      adoption → proxy by SP500 positive momentum
      currency_instability → proxy by USD change volatility
      recession_probability → proxy by inflation + bond yield trends
      tail_risk_event → proxy by VIX extreme moves
      geopolitics → proxy by oil price volatility
    """
    regulation = -vix_level/100.0
    adoption = max(min(sp500_change, 0.1), -0.1)
    currency_instability = -usd_change
    recession_probability = min((sp500_change - inflation/100.0)/2.0, 1.0)
    tail_risk_event = -vix_level/150.0
    geopolitics = oil_change / 100.0
    return regulation, adoption, currency_instability, recession_probability, tail_risk_event, geopolitics

# ---------------------------
# Build weights
# ---------------------------

def build_weights():
    # Macro indicators
    inflation = compute_inflation_yoy(FRED_SERIES["cpi"]) or 2.5
    bond_yield = fetch_csv_last_value(FRED_SERIES["us10y"]) or 3.0
    usd_index = fetch_csv_last_value(FRED_SERIES["dxy"]) or 100.0
    oil_price = fetch_csv_last_value(FRED_SERIES["oil"]) or 70.0
    sp500 = fetch_csv_last_value(FRED_SERIES["sp500"]) or 4500.0
    vix = fetch_csv_last_value(FRED_SERIES["vix"], value_col="CLOSE") or 20.0

    # Derived changes for proxies
    sp500_change = (sp500 - 4500)/4500
    usd_change = (usd_index - 100)/100
    oil_change = (oil_price - 70)/70
    vix_level = vix

    regulation, adoption, currency_instability, recession_probability, tail_risk_event, geopolitics = \
        calc_sentiment_proxies(sp500_change, vix_level, usd_change, oil_change, inflation)

    weights = {}
    for asset in ["gold","bitcoin"]:
        weights[asset] = {
            "inflation": normalize(inflation, 3),
            "real_rates": normalize(bond_yield - inflation, 1),
            "bond_yields": normalize(bond_yield, 3),
            "energy_prices": normalize(oil_price, 60),
            "usd_strength": normalize(usd_index, 100),
            "liquidity": -vix_level,
            "equity_flows": sp500_change,
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
        yaml.dump(weights,f)
    print("✅ weight.yaml updated with live indicators")
