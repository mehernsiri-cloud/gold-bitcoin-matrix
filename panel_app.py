# panel_app.py
import pandas as pd
import yaml
import panel as pn
from ai_predictor import predict_next_n, load_historical_prices, load_macro_indicators

pn.extension('tabulator', sizing_mode="stretch_width")

def get_dashboard():
    """Returns a Panel layout for Binder-ready dashboard"""

    # --- Load historical data ---
    try:
        gold_hist = load_historical_prices("gold")
        bitcoin_hist = load_historical_prices("bitcoin")
    except Exception as e:
        gold_hist = pd.Series(dtype=float)
        bitcoin_hist = pd.Series(dtype=float)
        print(f"Error loading historical data: {e}")

    # --- Load macro indicators ---
    try:
        gold_macro = load_macro_indicators("gold")
        bitcoin_macro = load_macro_indicators("bitcoin")
    except Exception as e:
        gold_macro = {}
        bitcoin_macro = {}
        print(f"Error loading macro indicators: {e}")

    # --- Generate AI predictions ---
    try:
        df_gold_pred = predict_next_n("gold", n_steps=5)
        df_bitcoin_pred = predict_next_n("bitcoin", n_steps=5)
    except Exception as e:
        df_gold_pred = pd.DataFrame(columns=["timestamp", "predicted_price"])
        df_bitcoin_pred = pd.DataFrame(columns=["timestamp", "predicted_price"])
        print(f"Error generating predictions: {e}")

    # --- Panels ---
    gold_table = pn.widgets.Tabulator(
        pd.DataFrame({"Historical": gold_hist, "Macro": pd.Series(gold_macro)}),
        name="Gold Historical & Macro",
        layout="fit_data"
    )
    bitcoin_table = pn.widgets.Tabulator(
        pd.DataFrame({"Historical": bitcoin_hist, "Macro": pd.Series(bitcoin_macro)}),
        name="Bitcoin Historical & Macro",
        layout="fit_data"
    )

    gold_pred_table = pn.widgets.Tabulator(df_gold_pred, name="Gold Predictions")
    bitcoin_pred_table = pn.widgets.Tabulator(df_bitcoin_pred, name="Bitcoin Predictions")

    # --- Layout ---
    tabs = pn.Tabs(
        ("Gold Data", gold_table),
        ("Bitcoin Data", bitcoin_table),
        ("Gold Predictions", gold_pred_table),
        ("Bitcoin Predictions", bitcoin_pred_table)
    )

    return tabs
