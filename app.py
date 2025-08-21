import streamlit as st
import pandas as pd
import yaml
import requests
import os

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Predictor", layout="wide")

# Drivers list
DRIVERS = [
    "geopolitics", "inflation", "real_rates", "usd_strength", "liquidity",
    "equity_flows", "bond_yields", "regulation", "adoption", 
    "currency_instability", "recession_probability", 
    "energy_prices", "tail_risk_event"
]

# ------------------------------
# Load Weights
# ------------------------------
@st.cache_data
def load_weights(path="weights.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------
# Auto fetch with real APIs
# ------------------------------
def auto_fetch():
    params = {}

    # --- Inflation (US CPI from FRED) ---
    try:
        fred_key = st.secrets.get("FRED_KEY")
        cpi = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={fred_key}&file_type=json"
        ).json()
        latest_cpi = float(cpi["observations"][-1]["value"])
        params["inflation"] = 2 if latest_cpi > 3 else (1 if latest_cpi > 2 else 0)
    except:
        params["inflation"] = 0

    # --- Real Rates (10y nominal - 10y TIPS) ---
    try:
        t10 = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={fred_key}&file_type=json"
        ).json()
        tips10 = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id=DFII10&api_key={fred_key}&file_type=json"
        ).json()
        real_rate = float(t10["observations"][-1]["value"]) - float(tips10["observations"][-1]["value"])
        params["real_rates"] = -2 if real_rate > 2 else (-1 if real_rate > 1 else 0)
    except:
        params["real_rates"] = 0

    # --- USD Strength (DXY via Alpha Vantage) ---
    try:
        av_key = st.secrets.get("ALPHAVANTAGE_KEY")
        dxy = requests.get(
            f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=EUR&apikey={av_key}"
        ).json()
        dxy_val = float(dxy["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        params["usd_strength"] = 2 if dxy_val < 0.9 else (1 if dxy_val < 0.95 else 0)  # USD strong = low EURUSD
    except:
        params["usd_strength"] = 0

    # --- Bitcoin adoption (CoinGecko) ---
    try:
        cg = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin").json()
        followers = cg["community_data"]["twitter_followers"] or 0
        params["adoption"] = 2 if followers > 6000000 else (1 if followers > 5000000 else 0)
    except:
        params["adoption"] = 0

    # Fill the rest with neutral
    for k in DRIVERS:
        if k not in params:
            params[k] = 0

    return params

# ------------------------------
# Scoring
# ------------------------------
def score_assets(params, weights):
    results = {}
    for asset in ["gold", "bitcoin"]:
        score = sum(params[k] * weights[asset].get(k, 0) for k in params)
        results[asset] = {
            "score": score,
            "direction": "Bullish" if score > 20 else "Bearish" if score < -20 else "Neutral",
            "expected_return": round((score / 100) * (15 if asset == "gold" else 30), 2),
            "confidence": f"{min(abs(score), 100)}%"
        }
    return results

# ------------------------------
# UI
# ------------------------------
st.title("📈 Gold & Bitcoin Predictive Matrix")

mode = st.sidebar.radio("Mode", ["Manual", "Auto (with APIs)"])

# Load weights
uploaded = st.sidebar.file_uploader("Upload custom weights.yaml", type="yaml")
if uploaded:
    weights = yaml.safe_load(uploaded)
else:
    weights = load_weights()

# Params
if mode.startswith("Manual"):
    params = {}
    st.subheader("Set Drivers")
    for d in DRIVERS:
        params[d] = st.slider(d, -2, 2, 0)
else:
    st.subheader("Auto mode fetching real data...")
    params = auto_fetch()
    st.write(params)

# Score
results = score_assets(params, weights)

st.subheader("Results")
df = pd.DataFrame(results).T
st.dataframe(df)

st.download_button("Download results (CSV)", df.to_csv().encode(), "predictions.csv")
