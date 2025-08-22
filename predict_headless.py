# predict_headless.py
import os
import pandas as pd
from datetime import datetime
import yaml

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = "data"
PRED_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
WEIGHT_FILE = "weight.yaml"
ASSETS = ["Gold", "Bitcoin"]

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# LOAD WEIGHTS
# ------------------------------
with open(WEIGHT_FILE, "r") as f:
    weights = yaml.safe_load(f)

# ------------------------------
# LOAD ACTUAL DATA
# ------------------------------
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
if os.path.exists(ACTUAL_FILE):
    actual_df = pd.read_csv(ACTUAL_FILE)
else:
    actual_df = pd.DataFrame(columns=["timestamp","gold_actual","bitcoin_actual"])

# Get latest actual prices
latest_actual = actual_df.iloc[-1] if not actual_df.empty else {"gold_actual": None, "bitcoin_actual": None}

# ------------------------------
# SIMPLE PREDICTION FUNCTION
# ------------------------------
def predict_price(latest_price, asset):
    if latest_price is None:
        return None
    # Sum of weights as simple linear score
    score = sum(weights[asset.lower()].values())
    predicted_price = latest_price * (1 + score)
    return round(predicted_price, 2)

# ------------------------------
# SAVE PREDICTIONS
# ------------------------------
def save_predictions():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for asset in ASSETS:
        latest_price = latest_actual[f"{asset.lower()}_actual"]
        pred_price = predict_price(latest_price, asset)
        rows.append({
            "timestamp": now,
            "asset": asset,
            "predicted_price": pred_price,
            "volatility": 0.01,  # placeholder, can add real volatility later
            "risk": "Medium"
        })

    df_new = pd.DataFrame(rows)

    # Initialize CSV if missing
    if not os.path.exists(PRED_FILE):
        df_new.to_csv(PRED_FILE, index=False)
    else:
        df_new.to_csv(PRED_FILE, mode="a", index=False, header=False)
    print(f"Saved predictions at {now}")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    save_predictions()
