# update_weights.py
import yfinance as yf
import requests
import yaml
import os
import numpy as np

WEIGHT_FILE = "weight.yaml"

# ------------------------------
# UTILITIES
# ------------------------------

def fetch_price_safe(ticker, period="1d"):
    """Fetches the last close price safely, returns None if unavailable"""
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
    """Placeholder function to simulate news/trend sentiment [-1,1]"""
    # In production, replace this with real sentiment API
    import random
    return round(random.uniform(-1, 1), 2)

def normalize_value(value, min_val, max_val):
    """Normalize value to [-1, 1] range"""
    if value is None:
        return 0
    norm = (2 * (value - min_val) / (max_val - min_val)) - 1
    return max(-1, min(1, norm))

def convert_numpy(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# ------------------------------
# FETCH AND CALCULATE INDICATORS
# ------------------------------

def get_indicator_values():
    indicators = {}

    # ------------------------------
    # GOLD indicators
    # ------------------------------
    # Geopolitics, Regulation, Adoption, Tail Risk, Currency Instability, Recession Probability
    indicators["geopolitics"] = fetch_news_sentiment_dummy("geopolitics")
    indicators["regulation"] = fetch_news_sentiment_dummy("regulation")
    indicators["adoption"] = fetch_news_sentiment_dummy("adoption")
    indicators["tail_risk_event"] = fetch_news_sentiment_dummy("tail risk")
    indicators["currency_instability"] = fetch_news_sentiment_dummy("currency instability")
    indicators["recession_probability"] = fetch_news_sentiment_dummy("recession probability")

    # Inflation: fetch US CPI (e.g., via yfinance ^CPI)
    cpi = fetch_price_safe("^CPI")  # placeholder ticker
    indicators["inflation"] = normalize_value(cpi, 200, 350)  # adjust min/max realistic range

    # Real rates: US 10y yield - inflation
    yield_10y = fetch_price_safe("^TNX")  # 10y US Treasury yield
    if yield_10y is not None and cpi is not None:
        indicators["real_rates"] = normalize_value(yield_10y - cpi, -5, 10)
    else:
        indicators["real_rates"] = 0

    # USD strength: DXY index
    dxy = fetch_price_safe("^DXY")
    indicators["usd_strength"] = normalize_value(dxy, 80, 120)

    # Liquidity: proxy via SP500 volume
    sp500 = fetch_price_safe("^GSPC")
    indicators["liquidity"] = normalize_value(sp500, 3000, 5000)

    # Equity flows: S&P500 returns
    sp500_prev = fetch_price_safe("^GSPC", period="5d")
    if sp500 is not None and sp500_prev is not None:
        indicators["equity_flows"] = normalize_value((sp500 - sp500_prev)/sp500_prev, -0.05, 0.05)
    else:
        indicators["equity_flows"] = 0

    # Bond yields
    indicators["bond_yields"] = normalize_value(yield_10y, 0, 5)

    # Energy prices: crude oil WTI
    oil = fetch_price_safe("CL=F")
    indicators["energy_prices"] = normalize_value(oil, 50, 120)

    return indicators

# ------------------------------
# MAIN UPDATE FUNCTION
# ------------------------------

def update_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f)
    else:
        weights = {"gold": {}, "bitcoin": {}}

    new_values = get_indicator_values()

    # Update indicators for both gold and bitcoin
    for asset in ["gold", "bitcoin"]:
        if asset not in weights:
            weights[asset] = {}
        for k, v in new_values.items():
            weights[asset][k] = v

    # Convert numpy types before saving
    weights_to_save = convert_numpy(weights)

    with open(WEIGHT_FILE, "w") as f:
        yaml.safe_dump(weights_to_save, f, sort_keys=False)

    print("✅ weight.yaml updated successfully.")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    update_weights()
