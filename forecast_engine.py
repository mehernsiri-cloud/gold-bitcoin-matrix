"""Adaptive, asset-specific forecasting engine for Gold and Bitcoin.

The engine is deliberately deterministic and explainable. It combines robust
momentum, trend, mean reversion, volatility regime and a deliberately weak
macro prior. Calibration only uses outcomes from this model version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 7, 14, 30)
MODEL_VERSION = "adaptive_ensemble_v2"


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _robust_returns(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 5:
        return values
    low, high = values.quantile([0.025, 0.975])
    return values.clip(lower=low, upper=high)


def _slope(log_prices: np.ndarray) -> float:
    if len(log_prices) < 5:
        return 0.0
    x = np.arange(len(log_prices), dtype=float)
    return float(np.polyfit(x, log_prices, 1)[0])


def _calibration_row(calibration: dict, horizon: int) -> dict:
    row = calibration.get(horizon, {}) if isinstance(calibration, dict) else {}
    return {
        "bias_pct": float(row.get("bias_pct", 0.0) or 0.0),
        "sample": int(row.get("sample", 0) or 0),
        "residuals_pct": list(row.get("residuals_pct", []) or []),
    }


def build_adaptive_forecast(
    asset: str,
    current_price: float,
    history: pd.DataFrame,
    macro: dict | None = None,
    calibration: dict | None = None,
) -> dict:
    """Build horizon-specific forecasts using an asset-specific regime ensemble."""
    if asset not in ("Gold", "Bitcoin"):
        raise ValueError(f"Unsupported asset: {asset}")
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("current_price must be a positive finite number")

    hist = history.copy().sort_values("timestamp")
    close = pd.to_numeric(hist["close"], errors="coerce").dropna()
    close = close[close > 0]
    if len(close) < 60:
        raise RuntimeError(f"Historique insuffisant pour {asset}: {len(close)} séances, 60 minimum recommandées.")

    log_close = np.log(close)
    returns = _robust_returns(log_close.diff())
    if len(returns) < 40:
        raise RuntimeError(f"Historique de rendements insuffisant pour {asset}.")

    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())

    gains = close.diff().clip(lower=0).rolling(14).mean().iloc[-1]
    losses = (-close.diff().clip(upper=0)).rolling(14).mean().iloc[-1]
    if pd.isna(gains) or pd.isna(losses):
        rsi = 50.0
    elif losses == 0:
        rsi = 100.0 if gains > 0 else 50.0
    else:
        rsi = float(100 - 100 / (1 + gains / losses))

    vol20 = float(returns.tail(20).std())
    vol60 = float(returns.tail(60).std())
    vol20 = vol20 if np.isfinite(vol20) else 0.0
    vol60 = vol60 if np.isfinite(vol60) else vol20
    daily_vol = max(float(np.sqrt(0.65 * vol20**2 + 0.35 * vol60**2)), 0.002)

    momentum5 = float(returns.tail(5).mean())
    momentum10 = float(returns.tail(10).mean())
    momentum20 = float(returns.tail(20).mean())
    momentum_signal = 0.20 * momentum5 + 0.45 * momentum10 + 0.35 * momentum20
    trend20 = _slope(log_close.tail(20).to_numpy())
    trend60 = _slope(log_close.tail(60).to_numpy())
    trend_signal = 0.70 * trend20 + 0.30 * trend60

    deviation = float(np.log(max(current_price, 1e-12) / max(sma20, 1e-12)))
    z_score = deviation / max(daily_vol * np.sqrt(20), 1e-6)
    reversion_signal = _clip(-z_score * daily_vol * 0.035, -daily_vol * 0.10, daily_vol * 0.10)

    trend_strength = abs(trend20) / max(daily_vol, 1e-6)
    vol_ratio = vol20 / max(vol60, 1e-6)
    if vol_ratio >= 1.35:
        regime = "high_volatility"
    elif sma20 > sma50 and trend20 > 0 and trend_strength >= 0.20:
        regime = "uptrend"
    elif sma20 < sma50 and trend20 < 0 and trend_strength >= 0.20:
        regime = "downtrend"
    else:
        regime = "range"

    if asset == "Gold":
        weights = {"momentum": 0.28, "trend": 0.36, "reversion": 0.24, "macro": 0.12}
        drift_cap = min(0.008, max(0.0012, daily_vol * 0.85))
        decay_days = 45.0
        calendar_factor = 5.0 / 7.0
        fat_tail = 1.05
    else:
        weights = {"momentum": 0.40, "trend": 0.35, "reversion": 0.15, "macro": 0.10}
        drift_cap = min(0.020, max(0.0020, daily_vol * 0.85))
        decay_days = 30.0
        calendar_factor = 1.0
        fat_tail = 1.25

    if regime in ("uptrend", "downtrend"):
        weights["reversion"] *= 0.55
        weights["momentum"] *= 1.10
        weights["trend"] *= 1.10
    elif regime == "range":
        weights["reversion"] *= 1.45
        weights["trend"] *= 0.75
    else:
        weights["momentum"] *= 0.65
        weights["trend"] *= 0.80
        weights["reversion"] *= 0.80

    total_weight = sum(weights.values())
    weights = {key: value / total_weight for key, value in weights.items()}

    macro_values = macro or {}
    macro_total = float(sum(float(v) for v in macro_values.values() if isinstance(v, (int, float))))
    # Existing macro scores are heuristic proxies, not returns. Keep their influence small.
    macro_daily = _clip(macro_total * 0.00012, -0.00025, 0.00025)

    raw_drift = (
        weights["momentum"] * momentum_signal
        + weights["trend"] * trend_signal
        + weights["reversion"] * reversion_signal
        + weights["macro"] * macro_daily
    )
    daily_drift = _clip(raw_drift, -drift_cap, drift_cap)

    factors = {
        "Momentum robuste 5/10/20": momentum_signal,
        "Tendance multi-fenêtre 20/60": trend_signal,
        "Retour à la moyenne": reversion_signal,
        "Contribution macro (faible)": macro_daily,
        "RSI 14 normalisé": (rsi - 50.0) / 50.0,
        "Volatilité réalisée 20/60": daily_vol,
        "Régime de marché": regime,
    }
    for name, value in macro_values.items():
        if isinstance(value, (int, float)):
            factors[f"Macro · {name}"] = float(value)

    forecasts = []
    calibration = calibration or {}
    for horizon in HORIZONS:
        effective_days = horizon * calendar_factor
        decay = float(np.exp(-horizon / decay_days))
        horizon_drift = daily_drift * decay + macro_daily * (1.0 - decay) * 0.35
        raw_price = float(current_price * np.exp(horizon_drift * effective_days))

        cal = _calibration_row(calibration, horizon)
        bias_fraction = 0.0
        if cal["sample"] >= 8 and np.isfinite(cal["bias_pct"]):
            bias_fraction = _clip(cal["bias_pct"] / 100.0, -0.03, 0.03)
        predicted = raw_price / max(1.0 + bias_fraction, 0.5)

        residuals = np.asarray(cal["residuals_pct"], dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) >= 20:
            q10, q90 = np.quantile(residuals, [0.10, 0.90]) / 100.0
            lower = predicted * max(0.01, 1.0 + float(q10))
            upper = predicted * max(1.01, 1.0 + float(q90))
        else:
            band = 1.2816 * daily_vol * np.sqrt(max(effective_days, 1.0)) * fat_tail
            lower = float(predicted * np.exp(-band))
            upper = float(predicted * np.exp(band))

        lower = min(lower, predicted * 0.999)
        upper = max(upper, predicted * 1.001)
        forecasts.append({
            "horizon_days": horizon,
            "target_date": (pd.Timestamp.utcnow().date() + pd.Timedelta(days=horizon)).isoformat(),
            "predicted_price": float(predicted),
            "p10": float(lower),
            "p90": float(upper),
            "expected_return_pct": (float(predicted) / current_price - 1.0) * 100.0,
            "calibration_bias_pct": float(cal["bias_pct"]),
            "calibration_sample": int(cal["sample"]),
            "horizon_daily_drift": float(horizon_drift),
        })

    return {
        "asset": asset,
        "current_price": float(current_price),
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi,
        "momentum10_pct": (float(np.exp(momentum10)) - 1.0) * 100.0,
        "momentum20_pct": (float(np.exp(momentum20 * 20)) - 1.0) * 100.0,
        "volatility20_pct": daily_vol * 100.0,
        "volatility60_pct": vol60 * 100.0,
        "daily_drift": daily_drift,
        "macro_total": macro_total,
        "factors": factors,
        "regime": regime,
        "model_version": MODEL_VERSION,
        "forecasts": forecasts,
        "history": hist,
    }
