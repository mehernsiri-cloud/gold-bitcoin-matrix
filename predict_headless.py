# predict_headless.py
import yfinance as yf
import pandas as pd
import datetime
import os
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

assets = {
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
    "France_RE": "BNP.PA",
    "Dubai_RE": "EMAAR.DU"
}

log_file = "predictions_log.csv"
if os.path.exists(log_file):
    df_log = pd.read_csv(log_file)
else:
    df_log = pd.DataFrame(columns=["date", "asset", "actual", "prediction", "risk"])

today = datetime.date.today()

for name, ticker in assets.items():
    data = yf.download(ticker, period="6mo", interval="1d")
    if not data.empty:
        actual_price = data["Close"].iloc[-1]

        # Prediction with ARIMA
        try:
            series = data["Close"].dropna()
            model = ARIMA(series, order=(2,1,2))
            model_fit = model.fit()
            prediction = float(model_fit.forecast(steps=1).iloc[0])
        except:
            prediction = series.rolling(window=5).mean().iloc[-1]

        # Compute risk based on 14-day rolling volatility
        returns = series.pct_change().dropna()
        vol = returns.rolling(window=14).std().iloc[-1]
        if vol < 0.005:
            risk = "Low"
        elif vol < 0.015:
            risk = "Medium"
        else:
            risk = "High"

        new_row = {
            "date": today,
            "asset": name,
            "actual": round(actual_price,2),
            "prediction": round(prediction,2),
            "risk": risk
        }
        df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)

df_log.to_csv(log_file, index=False)
print("✅ Predictions with risk updated:", today)
