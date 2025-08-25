import requests
import pandas as pd
import numpy as np
import yaml
from bs4 import BeautifulSoup
from textblob import TextBlob

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------------------
# Helper functions
# ---------------------------

def safe_request(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"⚠️ Request failed for {url}: {e}")
        return ""

def get_cpi():
    """Get US Inflation YoY from TradingEconomics."""
    url = "https://tradingeconomics.com/united-states/inflation-cpi"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    try:
        val = soup.find("td", text="Inflation Rate YoY").find_next("td").get_text()
        return float(val.replace("%","").strip())
    except Exception:
        return 3.0  # fallback avg

def get_bond_yield():
    """US 10Y Treasury from MarketWatch (live)."""
    url = "https://www.marketwatch.com/investing/bond/tmubmusd10y"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return float(val)
    except Exception:
        return 3.5

def get_usd_index():
    url = "https://www.marketwatch.com/investing/index/dxy"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return float(val)
    except Exception:
        return 100.0

def get_oil_price():
    url = "https://www.marketwatch.com/investing/future/brn00"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return float(val.replace(",", ""))
    except Exception:
        return 70.0

def get_liquidity_proxy():
    """VIX index as liquidity proxy (inverse)."""
    url = "https://www.marketwatch.com/investing/index/vix"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    try:
        val = soup.find("bg-quote", {"class": "value"}).get_text()
        return -float(val)
    except Exception:
        return -20.0

def get_equity_flows_proxy():
    """S&P500 daily % change as proxy for flows."""
    url = "https://www.marketwatch.com/investing/index/spx"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    try:
        val = soup.find("span", {"class": "change--percent--q"}).get_text()
        return float(val.replace("%", "").replace("+", "").strip()) / 100.0
    except Exception:
        return 0.0

def get_sentiment_news(keyword="regulation"):
    """Google News headlines → sentiment score via TextBlob."""
    url = f"https://news.google.com/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    html = safe_request(url)
    soup = BeautifulSoup(html, "html.parser")
    headlines = [a.get_text() for a in soup.find_all("a", {"class": "DY5T1d"})][:5]
    if not headlines:
        return 0.0
    polarity = np.mean([TextBlob(h).sentiment.polarity for h in headlines])
    return round(polarity, 3)

# ---------------------------
# Main calculation
# ---------------------------

def normalize(value, ref=100.0):
    try:
        return (value - ref) / abs(ref)
    except Exception:
        return 0.0

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
