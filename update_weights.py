# update_weights.py
import yfinance as yf
import yaml
import os
import numpy as np
import random

WEIGHT_FILE = "weight.yaml"

# ------------------------------
# UTILITIES
# ------------------------------
def fetch_price_safe(ticker, period="1d"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            print(f"⚠️ Warning: no price data for {ticker}")
            return None
        return df["Close"].iloc[-1]
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {e}")
        return None

def fetch_news_sentiment_dummy(keyword):
    """Simulate news/trend sentiment [-1,1]"""
    return round(random.uniform(-1, 1), 2)

def normalize_value(value, min_val, max_val):
    """Normalize value to [-1, 1] safely"""
    if value is None:
        return 0.0
    norm = (2 * (value - min_val) / (max_val - min_val)) - 1
    return max(-1.0, min(1.0, norm))

def convert_to_python(obj):
    """Recursively convert numpy types to native Python and handle None"""
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif obj is None:
        return 0.0
    else:
        return obj

# ------------------------------
# FETCH INDICATORS
# ------------------------------
def get_indicator_values():
    indicators = {}

    # News/Trend-based
    for key in ["geopolitics", "regulation", "adoption", "tail_risk_event", 
                "currency_instability", "recession_probability"]:
        indicators[key] = fetch_news_sentiment_dummy(key)

    # Inflation: placeholder with safe fallback
    cpi = fetch_price_safe("^CPI")
    indicators["inflation"] = normalize_value(cpi, 200, 350)

    # Real rates: 10y yield minus inflation
    yield_10y = fetch_price_safe("^TNX")
    if yield_10y is not None and cpi is not None:
        indicators["real_rates"] = normalize_value(yield_10y - cpi, -5, 10)
    else:
        indicators["real_rates"] = 0.0

    # USD strength
    dxy = fetch_price_safe("^DXY")
    indicators["usd_strength"] = normalize_value(dxy, 80, 120)

    # Liquidity proxy
    sp500 = fetch_price_safe("^GSPC")
    indicators["liquidity"] = normalize_value(sp500, 3000, 5000)

    # Equity flows
    sp500_prev = fetch_price_safe("^GSPC", period="5d")
    if sp500 is not None and sp500_prev is not None:
        indicators["equity_flows"] = normalize_value((sp500 - sp500_prev)/sp500_prev, -0.05, 0.05)
    else:
        indicators["equity_flows"] = 0.0

    # Bond yields
    indicators["bond_yields"] = normalize_value(yield_10y, 0, 5)

    # Energy prices (WTI)
    oil = fetch_price_safe("CL=F")
    indicators["energy_prices"] = normalize_value(oil, 50, 120)

    return indicators

# ------------------------------
# UPDATE YAML
# ------------------------------
def update_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f)
    else:
        weights = {"gold": {}, "bitcoin": {}}

    new_values = get_indicator_values()

    for asset in ["gold", "bitcoin"]:
        if asset not in weights:
            weights[asset] = {}
        for k, v in new_values.items():
            weights[asset][k] = float(v)  # ensure float type

    weights_safe = convert_to_python(weights)

    with open(WEIGHT_FILE, "w") as f:
        yaml.safe_dump(weights_safe, f, sort_keys=False)

    print("✅ weight.yaml updated successfully.")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    update_weights()
