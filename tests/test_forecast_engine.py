import numpy as np
import pandas as pd

from forecast_engine import MODEL_VERSION, build_adaptive_forecast


def sample_history(n=260, drift=0.0008, volatility=0.012):
    rng = np.random.default_rng(42)
    returns = rng.normal(drift, volatility, n)
    prices = 2000 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="B"),
        "close": prices,
    })


def test_forecast_has_all_horizons_and_valid_bands():
    result = build_adaptive_forecast("Gold", 2300, sample_history(), {})
    assert result["model_version"] == MODEL_VERSION
    assert [row["horizon_days"] for row in result["forecasts"]] == [1, 3, 7, 14, 30]
    for row in result["forecasts"]:
        assert np.isfinite(row["predicted_price"])
        assert 0 < row["p10"] < row["predicted_price"] < row["p90"]


def test_assets_use_different_forecast_dynamics():
    history = sample_history(drift=0.001, volatility=0.025)
    gold = build_adaptive_forecast("Gold", 2300, history, {})
    bitcoin = build_adaptive_forecast("Bitcoin", 90000, history, {})
    assert gold["regime"] == bitcoin["regime"]
    assert gold["forecasts"][2]["expected_return_pct"] != bitcoin["forecasts"][2]["expected_return_pct"]


def test_calibration_reduces_persistent_overprediction():
    history = sample_history(drift=0.001, volatility=0.01)
    raw = build_adaptive_forecast("Bitcoin", 90000, history, {})
    calibrated = build_adaptive_forecast(
        "Bitcoin", 90000, history, {},
        {7: {"bias_pct": 2.0, "sample": 12, "residuals_pct": []}},
    )
    assert calibrated["forecasts"][2]["predicted_price"] < raw["forecasts"][2]["predicted_price"]


def test_calibration_is_ignored_below_minimum_sample():
    history = sample_history()
    raw = build_adaptive_forecast("Gold", 2300, history, {})
    ignored = build_adaptive_forecast(
        "Gold", 2300, history, {},
        {7: {"bias_pct": 2.0, "sample": 3, "residuals_pct": []}},
    )
    assert ignored["forecasts"][2]["predicted_price"] == raw["forecasts"][2]["predicted_price"]
