# run_ai_predictions.py
from ai_predictor import predict_next_n, backtest_ai

# Generate AI predictions for Gold and Bitcoin
predict_next_n(None, None, "Gold")
predict_next_n(None, None, "Bitcoin")

# Run backtest to fill historical AI log
backtest_ai("Gold")
backtest_ai("Bitcoin")

print("✅ AI predictions and backtests completed.")
