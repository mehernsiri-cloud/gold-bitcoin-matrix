#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import random
import yaml

# ------------------------------
ASSETS = {
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
    "Real_Estate_France": "RWR",
    "Real_Estate_Dubai": "DXRE"
}
CSV_FILE = "data/predictions_log.csv"
VOL_THRESHOLDS = {"Gold":0.005,"Bitcoin":0.02,"Real_Estate_France":0.01,"Real_Estate_Dubai":0.01}

# Load weights
with open("weight.yaml","r") as f:
    WEIGHTS = yaml.safe_load(f)

# ------------------------------
def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="30d", interval="1d", progress=False)
        if df.empty: raise ValueError(f"No data for {symbol}")
        return df
    except:
        return None

def compute_volatility(df):
    df['returns'] = df['Adj Close'].pct_change()
    return df['returns'].std()

def predict_price(df):
    last_price = df['Adj Close'].iloc[-1]
    last_return = df['Adj Close'].pct_change().iloc[-1]
    return last_price*(1+last_return)

def weighted_predict(asset, last_price):
    asset_lower = asset.lower()
    if asset in WEIGHTS:
        factor_weights = WEIGHTS[asset_lower]
        factor_values = {k: random.uniform(-0.05,0.05) for k in factor_weights.keys()}
        adjustment = sum(factor_weights[f]*factor_values[f] for f in factor_weights.keys())
        return last_price*(1+adjustment)
    return last_price

def compute_risk(vol, threshold):
    if vol<threshold: return "Low"
    elif vol<threshold*2: return "Medium"
    else: return "High"

def ensure_csv_exists(file_path):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["timestamp","asset","predicted_price","volatility","risk"])
        df.to_csv(file_path,index=False)

def generate_placeholder(asset):
    return {"timestamp":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "asset":asset,
            "predicted_price":round(random.uniform(100,1000),2),
            "volatility":round(random.uniform(0,0.05),4),
            "risk":"Placeholder"}

# ------------------------------
def main():
    ensure_csv_exists(CSV_FILE)
    results=[]
    for asset,symbol in ASSETS.items():
        df = fetch_data(symbol)
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if df is not None:
            try:
                vol=compute_volatility(df)
                predicted_price=weighted_predict(asset,predict_price(df))
                risk=compute_risk(vol,VOL_THRESHOLDS[asset])
                results.append({"timestamp":timestamp,"asset":asset,"predicted_price":round(predicted_price,2),
                                "volatility":round(vol,4),"risk":risk})
            except:
                results.append(generate_placeholder(asset))
        else:
            results.append(generate_placeholder(asset))
    pd.DataFrame(results).to_csv(CSV_FILE,mode='a',index=False,header=False)

if __name__=="__main__":
    main()
