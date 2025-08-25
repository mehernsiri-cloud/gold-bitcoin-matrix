# update_weights.py
import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime
from textblob import TextBlob  # for sentiment analysis
import requests

WEIGHT_FILE = "weight.yaml"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def normalize(value, min_val=-1, max_val=1):
    """Normalize a value to [min_val, max_val]"""
    return max(min(round(value,2), max_val), min_val)

def fetch_latest_price(ticker):
    try:
        data = yf.Ticker(ticker)
        price = data.history(period="1d")['Close'].iloc[-1]
        return price
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_news_sentiment(query, n_articles=5):
    """
    Fetch news headlines and calculate sentiment.
    Uses Google News RSS as example; can be replaced by professional APIs.
    Returns normalized sentiment [-1,1]
    """
    try:
        url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=en-US&gl=US&ceid=US:en"
        feed = requests.get(url).text
        headlines = pd.Series([line.split('<title>')[1].split('</title>')[0] 
                               for line in feed.split('<item>')[1:n_articles+1]])
        sentiment = headlines.apply(lambda x: TextBlob(x).sentiment.polarity)
        avg_sentiment = sentiment.mean()  # average sentiment [-1,1]
        return normalize(avg_sentiment)
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return 0

# ------------------------------
# FETCH LIVE INDICATORS
# ------------------------------
def get_dynamic_indicators():
    indicators = {}

    # Market data proxies
    gld_price = fetch_latest_price("GLD")
    if gld_price:
        indicators['inflation'] = normalize((gld_price - 150)/50*2-1)  # example

    dxy_price = fetch_latest_price("^DXY")
    if dxy_price:
        indicators['usd_strength'] = normalize((dxy_price - 80)/40*2-1)

    tnx = fetch_latest_price("^TNX")
    if tnx:
        indicators['bond_yields'] = normalize((tnx - 0)/5*2-1)

    oil_price = fetch_latest_price("BZ=F")
    if oil_price:
        indicators['energy_prices'] = normalize((oil_price - 50)/100*2-1)

    spy_price = fetch_latest_price("SPY")
    if spy_price:
        indicators['equity_flows'] = normalize((spy_price - 300)/200*2-1)

    # Real rates = bond_yields - inflation
    if 'bond_yields' in indicators and 'inflation' in indicators:
        indicators['real_rates'] = round(indicators['bond_yields'] - indicators['inflation'], 2)

    # News/Sentiment based indicators
    indicators['geopolitics'] = fetch_news_sentiment("geopolitical OR war OR conflict")
    indicators['regulation'] = fetch_news_sentiment("cryptocurrency regulation OR financial regulation")
    indicators['adoption'] = fetch_news_sentiment("bitcoin adoption OR cryptocurrency adoption")
    indicators['tail_risk_event'] = fetch_news_sentiment("financial crisis OR market crash OR tail risk")
    indicators['currency_instability'] = fetch_news_sentiment("currency devaluation OR forex instability")
    indicators['recession_probability'] = fetch_news_sentiment("recession OR economic slowdown")

    # Liquidity left static for now
    indicators['liquidity'] = 0

    return indicators

# ------------------------------
# UPDATE weight.yaml
# ------------------------------
def update_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f)
    else:
        weights = {"gold": {}, "bitcoin": {}}

    dynamic_indicators = get_dynamic_indicators()

    for asset in ["gold", "bitcoin"]:
        for k, v in dynamic_indicators.items():
            weights[asset][k] = v

    with open(WEIGHT_FILE, "w") as f:
        yaml.safe_dump(weights, f, sort_keys=False)

    print(f"[{datetime.now()}] weight.yaml updated with live market & news indicators!")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    update_weights()
