# run_ai_predictions.py
from ai_predictor import predict_next_n

# AI-driven predictions only
predict_next_n(asset_name="Gold", n_steps=5)
predict_next_n(asset_name="Bitcoin", n_steps=5)


print("✅ AI predictions and backtests completed.")
