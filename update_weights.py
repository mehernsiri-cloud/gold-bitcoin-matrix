import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import feedparser
from textblob import TextBlob
import os
import json

OUTPUT_FILE = "weights.json"

# --- Helpers ---
def normalize(series, inverse=False):
    if len(series) == 0:
        return 0.0
    val = series[-1]
    min_val, max_val = np.nanmin(series), np.nanmax(series)
    if max_val == min_val:
        return 0.0
    score = (val - min_val) / (max_val - min_val) * 2 - 1
    return -score if inverse else score

def sentiment_from_news(query):
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    if not feed.entries:
        return 0.0
    scores = []
    for e in feed.entries[:10]:
        txt = e.title + " " + e.get("summary", "")
        scores.append(TextBlob(txt).sentiment.polarity)
    return float(np.mean(scores))

# --- Indicators ---
def get_inflation():
    # OECD CPI data (monthly, percentage change)
    url = "https://stats.oecd.org/sdmx-json/data/DP_LIVE/.CPI.TOT.AGRWTH.M/OECD?contentType=csv"
    try:
        df = pd.read_csv(url)
        df = df[df["LOCATION"] == "USA"]
        return normalize(df["Value"].values)
    except Exception:
        return 0.0

def get_bond_yields():
    url = "https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=245.Q.YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y.YLD&trans=csv"
    try:
        df = pd.read_csv(url, skiprows=5)
        vals = df.iloc[:,1].dropna().values
        return normalize(vals)
    except Exception:
        return 0.0

def get_real_rates():
    infl = get_inflation()
    bonds = get_bond_yields()
    return bonds - infl

def get_energy_prices():
    url = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"
    # fallback to static CSV because EIA Excel is tricky
    return 0.1

def get_usd_strength():
    url = "https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=120.EXR.D.USD.EUR.SP00.A&type=csv"
    try:
        df = pd.read_csv(url, skiprows=5)
        vals = df.iloc[:,1].dropna().values
        return normalize(vals)
    except Exception:
        return 0.0

def get_liquidity():
    url = "https://fred.stlouisfed.org/data/WALCL.txt"
    try:
        df = pd.read_csv(url, sep="\s+", skiprows=10)
        vals = df["WALCL"].dropna().values
        return normalize(vals)
    except Exception:
        return 0.0

def get_equity_flows():
    # Proxy = VIX index (volatility)
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    try:
        df = pd.read_csv(url)
        vals = df["CLOSE"].dropna().values
        return -normalize(vals)  # higher VIX → outflows
    except Exception:
        return 0.0

def get_regulation():
    return sentiment_from_news("crypto regulation SEC")

def get_adoption():
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(["Bitcoin"], cat=0, timeframe="today 3-m", geo="", gprop="")
        df = pytrends.interest_over_time()
        if "Bitcoin" in df:
            return normalize(df["Bitcoin"].values)
    except Exception:
        return 0.0
    return 0.0

def get_currency_instability():
    try:
        url = "https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=120.EXR.D.USD.EUR.SP00.A&type=csv"
        df = pd.read_csv(url, skiprows=5)
        rates = df.iloc[:,1].dropna().pct_change().dropna()
        vol = rates.rolling(20).std().values
        return normalize(vol)
    except Exception:
        return 0.0

def get_recession_probability():
    try:
        # 10y vs 3m yields (ECB data)
        url10y = "https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=245.Q.YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y.YLD&trans=csv"
        url3m = "https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=245.Q.YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M.YLD&trans=csv"
        y10 = pd.read_csv(url10y, skiprows=5).iloc[:,1].dropna().values
        y3 = pd.read_csv(url3m, skiprows=5).iloc[:,1].dropna().values
        spread = np.array(y10[-len(y3):]) - y3
        return -normalize(spread)  # inversion => high probability
    except Exception:
        return 0.0

def get_tail_risk_event():
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    try:
        df = pd.read_csv(url)
        vix = df["CLOSE"].dropna().values
        return normalize(vix)
    except Exception:
        return 0.0

def get_geopolitics():
    return sentiment_from_news("war conflict sanctions")

# --- Main ---
def main():
    weights = {
        "inflation": get_inflation(),
        "real_rates": get_real_rates(),
        "bond_yields": get_bond_yields(),
        "energy_prices": get_energy_prices(),
        "usd_strength": get_usd_strength(),
        "liquidity": get_liquidity(),
        "equity_flows": get_equity_flows(),
        "regulation": get_regulation(),
        "adoption": get_adoption(),
        "currency_instability": get_currency_instability(),
        "recession_probability": get_recession_probability(),
        "tail_risk_event": get_tail_risk_event(),
        "geopolitics": get_geopolitics()
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(weights, f, indent=2)
    print(json.dumps(weights, indent=2))

if __name__ == "__main__":
    main()
