import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data"
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Fetch live prices
# -----------------------------
def fetch_prices():
    prices = {}
    tickers = {"Gold": "GC=F", "Bitcoin": "BTC-USD"}
    for asset, ticker in tickers.items():
        try:
            df = yf.Ticker(ticker).history(period="1d")
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            prices[asset] = float(df['Close'].iloc[-1])
        except Exception as e:
            print(f"⚠️ Warning fetching price for {asset} ({ticker}): {e}")
            prices[asset] = None
    return prices

# -----------------------------
# Fetch indicators
# -----------------------------
def fetch_indicators():
    # Placeholder example; can replace with real APIs
    indicators = {
        'inflation': 0.03,
        'real_rates': 0.01,
        'usd_strength': 1.0,
        'liquidity': 0.05,
        'equity_flows': 0.01,
        'bond_yields': 0.03,
        'regulation': 0.0,
        'adoption': 0.1,
        'currency_instability': 0.02,
        'recession_probability': 0.05,
        'tail_risk_event': 0.1,
        'geopolitics': 0.1,
    }
    try:
        energy = yf.Ticker("CL=F").history(period="1d")
        indicators['energy_prices'] = float(energy['Close'].iloc[-1])
    except:
        indicators['energy_prices'] = 70.0
    return indicators

# -----------------------------
# Save actual data
# -----------------------------
def save_actual_data():
    today = datetime.today().strftime("%Y-%m-%d")
    prices = fetch_prices()
    indicators = fetch_indicators()

    row = {"date": today, "gold_actual": prices.get("Gold"), "bitcoin_actual": prices.get("Bitcoin"), **indicators}

    # Read existing CSV safely
    if os.path.exists(ACTUAL_FILE) and os.path.getsize(ACTUAL_FILE) > 0:
        df = pd.read_csv(ACTUAL_FILE)
        # Prevent duplicates for the same day
        if today not in df['date'].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print("✅ Actual data updated:", row)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    save_actual_data()
