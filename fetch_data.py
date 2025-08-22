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
# Fetch macro indicators
# -----------------------------
def fetch_indicators():
    indicators = {}
    try:
        # Example values; can be replaced with real-time APIs
        indicators['inflation'] = 0.03
        indicators['real_rates'] = 0.01
        indicators['usd_strength'] = 1.0
        indicators['liquidity'] = 0.05
        indicators['equity_flows'] = 0.01
        indicators['bond_yields'] = 0.03
        indicators['regulation'] = 0.0
        indicators['adoption'] = 0.1
        indicators['currency_instability'] = 0.02
        indicators['recession_probability'] = 0.05
        indicators['tail_risk_event'] = 0.1
        indicators['geopolitics'] = 0.1
        # Energy price: WTI Crude
        try:
            energy = yf.Ticker("CL=F").history(period="1d")
            indicators['energy_prices'] = float(energy['Close'].iloc[-1])
        except:
            indicators['energy_prices'] = 70.0
    except Exception as e:
        print(f"⚠️ Warning fetching indicators: {e}")
    return indicators

# -----------------------------
# Save actual data safely
# -----------------------------
def save_actual_data():
    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")

    prices = fetch_prices()
    indicators = fetch_indicators()

    row = {
        "timestamp": timestamp,
        "date": today,
        "gold_actual": prices.get("Gold"),
        "bitcoin_actual": prices.get("Bitcoin"),
        **indicators
    }

    # Read existing CSV safely
    if os.path.exists(ACTUAL_FILE) and os.path.getsize(ACTUAL_FILE) > 0:
        try:
            df = pd.read_csv(ACTUAL_FILE)
            # Avoid duplicate timestamp entries
            if timestamp not in df['timestamp'].values:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print(f"✅ Actual data updated: {row}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    save_actual_data()
