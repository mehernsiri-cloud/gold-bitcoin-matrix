import requests
import pandas as pd
import numpy as np
import yaml
from bs4 import BeautifulSoup

# ---------------------------
# Helper functions
# ---------------------------

def get_cpi():
    """Get US Inflation (CPI YoY %) from investing.com API (scraped)."""
    url = "https://www.investing.com/economic-calendar/consumer-price-index-cpi-733"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        val = soup.find("td", {"class": "bold"}).get_text()
        return float(val.replace("%",""))
    except Exception:
        return 0.0

def get_bond_yield():
    """Get US 10Y Treasury Yield from MarketWatch."""
    url = "https://www.marketwatch.com/investing/bond/tmubmusd10y?mod=home-page"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return float(val)
    except Exception:
        return 0.0

def get_usd_index():
    """Get USD Index (DXY) from MarketWatch."""
    url = "https://www.marketwatch.com/investing/index/dxy"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return float(val)
    except Exception:
        return 100.0

def get_oil_price():
    """Get Brent crude oil price from MarketWatch."""
    url = "https://www.marketwatch.com/investing/future/brn00"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return float(val.replace(",", ""))
    except Exception:
        return 0.0

def get_liquidity_proxy():
    """Use VIX index as proxy for liquidity/risk sentiment."""
    url = "https://www.marketwatch.com/investing/index/vix"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return -float(val)  # higher VIX = lower liquidity
    except Exception:
        return 0.0

def get_equity_flows_proxy():
    """Use S&P500 daily % change as proxy for equity flows."""
    url = "https://www.marketwatch.com/investing/index/spx"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        val = soup.find("span", {"class": "change--percent--q"}).get_text()
        return float(val.replace("%", "").replace("+", "").strip()) / 100.0
    except Exception:
        return 0.0

def get_sentiment_news(keyword="regulation"):
    """Scrape Google News headlines for simple keyword sentiment scoring."""
    url = f"https://news.google.com/search?q={keyword}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    headlines = [a.get_text() for a in soup.find_all("a", {"class": "DY5T1d"})][:5]
    score = 0
    for h in headlines:
        if any(word in h.lower() for word in ["ban", "crackdown", "restrict", "lawsuit"]):
            score -= 0.2
        if any(word in h.lower() for word in ["support", "approve", "adopt", "positive"]):
            score += 0.2
    return score

# ---------------------------
# Main calculation
# ---------------------------

def normalize(value, ref=100.0):
    return (value - ref) / ref

def build_weights():
    inflation = get_cpi()
    bond_yield = get_bond_yield()
    usd_strength = get_usd_index()
    oil = get_oil_price()
    liquidity = get_liquidity_proxy()
    equity_flows = get_equity_flows_proxy()

    regulation = get_sentiment_news("crypto regulation")
    adoption = get_sentiment_news("bitcoin adoption")
    currency_instability = get_sentiment_news("currency crisis")
    recession_probability = get_sentiment_news("recession")
    tail_risk_event = get_sentiment_news("financial crisis")
    geopolitics = get_sentiment_news("geopolitics")

    data = {
        "gold": {
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
            "geopolitics": geopolitics,
        },
        "bitcoin": {
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
            "geopolitics": geopolitics,
        }
    }

    return data

# ---------------------------
# Save to YAML
# ---------------------------

if __name__ == "__main__":
    weights = build_weights()
    with open("weights.yaml", "w") as f:
        yaml.dump(weights, f)
    print("✅ weights.yaml updated with live indicators")
