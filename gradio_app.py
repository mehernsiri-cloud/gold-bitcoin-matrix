# gradio_app.py
import gradio as gr
import pandas as pd
from ai_predictor import predict_next_n, load_historical_prices, load_macro_indicators


def get_gold_data():
    """Return Gold historical & macro data as DataFrame"""
    try:
        gold_hist = load_historical_prices("gold")
    except Exception as e:
        print(f"[gradio_app] Error loading gold history: {e}")
        gold_hist = pd.Series(dtype=float)

    try:
        gold_macro = load_macro_indicators("gold")
    except Exception as e:
        print(f"[gradio_app] Error loading gold macro: {e}")
        gold_macro = {}

    df = pd.DataFrame({
        "Historical": gold_hist,
        "Macro": pd.Series(gold_macro)
    })
    return df


def get_bitcoin_data():
    """Return Bitcoin historical & macro data as DataFrame"""
    try:
        bitcoin_hist = load_historical_prices("bitcoin")
    except Exception as e:
        print(f"[gradio_app] Error loading bitcoin history: {e}")
        bitcoin_hist = pd.Series(dtype=float)

    try:
        bitcoin_macro = load_macro_indicators("bitcoin")
    except Exception as e:
        print(f"[gradio_app] Error loading bitcoin macro: {e}")
        bitcoin_macro = {}

    df = pd.DataFrame({
        "Historical": bitcoin_hist,
        "Macro": pd.Series(bitcoin_macro)
    })
    return df


def get_gold_predictions():
    """Return Gold predictions as DataFrame"""
    try:
        df_gold_pred = predict_next_n("gold", n_steps=5)
    except Exception as e:
        print(f"[gradio_app] Error generating gold predictions: {e}")
        df_gold_pred = pd.DataFrame(columns=["timestamp", "predicted_price"])
    return df_gold_pred


def get_bitcoin_predictions():
    """Return Bitcoin predictions as DataFrame"""
    try:
        df_bitcoin_pred = predict_next_n("bitcoin", n_steps=5)
    except Exception as e:
        print(f"[gradio_app] Error generating bitcoin predictions: {e}")
        df_bitcoin_pred = pd.DataFrame(columns=["timestamp", "predicted_price"])
    return df_bitcoin_pred


# -----------------------
# Build Gradio Interface
# -----------------------
with gr.Blocks(title="Gold & Bitcoin Dashboard") as demo:
    gr.Markdown("# 📊 Gold & Bitcoin Dashboard")

    with gr.Tab("Gold Data"):
        gr.Dataframe(get_gold_data, headers="keys", datatype="auto", label="Gold Historical & Macro")

    with gr.Tab("Bitcoin Data"):
        gr.Dataframe(get_bitcoin_data, headers="keys", datatype="auto", label="Bitcoin Historical & Macro")

    with gr.Tab("Gold Predictions"):
        gr.Dataframe(get_gold_predictions, headers="keys", datatype="auto", label="Gold Predictions")

    with gr.Tab("Bitcoin Predictions"):
        gr.Dataframe(get_bitcoin_predictions, headers="keys", datatype="auto", label="Bitcoin Predictions")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
