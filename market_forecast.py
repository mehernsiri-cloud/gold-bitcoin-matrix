# market_forecast.py
"""Unified, explainable Gold & Bitcoin market monitor and forecast engine.

Forecasts are deterministic, multi-factor scenarios, not guaranteed prices.
Every forecast is stored with its issue time and is never overwritten.
Outcomes are resolved against the first observed daily close on/after the target date.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml
import yfinance as yf

DATA_DIR = Path("data")
FORECAST_FILE = DATA_DIR / "market_forecasts.csv"
WEIGHT_FILE = Path("weight.yaml")
HORIZONS = (1, 3, 7, 14, 30)
ASSETS = {
    "Gold": {"ticker": "GC=F", "history_ticker": "GC=F", "unit": "USD/oz · futures GC=F", "color": "#D6A84B"},
    "Bitcoin": {"ticker": "BTC-USD", "history_ticker": "BTC-USD", "unit": "USD · spot", "color": "#E58B35"},
}
FORECAST_COLUMNS = [
    "forecast_id", "asset", "issued_at", "target_date", "horizon_days",
    "model_version", "current_price", "predicted_price", "p10", "p90",
    "expected_return_pct", "daily_drift", "daily_volatility", "calibration_bias_pct", "calibration_sample", "factor_json",
    "status", "actual_price", "resolved_at", "error_pct", "direction_correct",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _clean_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "close"])
    out = df.reset_index()
    time_col = "Date" if "Date" in out.columns else "Datetime" if "Datetime" in out.columns else out.columns[0]
    out = out.rename(columns={time_col: "timestamp", "Close": "close"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"])
    out = out[out["close"] > 0].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out[["timestamp", "close"]].reset_index(drop=True)


def fetch_asset_market_data(asset: str) -> dict:
    """Fetch a live/intraday quote plus up to two years of daily market history."""
    cfg = ASSETS[asset]
    ticker = yf.Ticker(cfg["history_ticker"])
    hist_raw = ticker.history(period="2y", interval="1d", auto_adjust=True)
    history = _clean_history(hist_raw)

    intraday = pd.DataFrame()
    try:
        intraday = ticker.history(period="5d", interval="5m", auto_adjust=True)
    except Exception:
        pass

    quote = None
    quote_time = None
    quote_source = "Yahoo Finance"
    change_24h = None
    if asset == "Bitcoin":
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bitcoin", "vs_currencies": "usd", "include_24hr_change": "true", "include_last_updated_at": "true"},
                timeout=8,
                headers={"accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()["bitcoin"]
            quote = float(payload["usd"])
            change_24h = float(payload.get("usd_24h_change")) if payload.get("usd_24h_change") is not None else None
            quote_time = datetime.fromtimestamp(int(payload.get("last_updated_at", _utc_now().timestamp())), timezone.utc)
            quote_source = "CoinGecko spot"
        except Exception:
            quote = None

    if quote is None and intraday is not None and not intraday.empty:
        valid = pd.to_numeric(intraday["Close"], errors="coerce").dropna()
        if not valid.empty:
            quote = float(valid.iloc[-1])
            idx = intraday.index[-1]
            quote_time = pd.Timestamp(idx).to_pydatetime()
            if quote_time.tzinfo is None:
                quote_time = quote_time.replace(tzinfo=timezone.utc)

    if quote is None and not history.empty:
        quote = float(history["close"].iloc[-1])
        quote_time = pd.Timestamp(history["timestamp"].iloc[-1]).to_pydatetime().replace(tzinfo=timezone.utc)
        quote_source = "Dernière clôture (fallback)"

    if quote is None or history.empty:
        raise RuntimeError(f"Données de marché indisponibles pour {asset}.")

    try:
        if change_24h is None and intraday is not None and len(intraday) > 1:
            intraday_close = pd.to_numeric(intraday["Close"], errors="coerce").dropna()
            if len(intraday_close) > 1:
                cutoff = pd.Timestamp(intraday_close.index[-1]) - pd.Timedelta(hours=24)
                recent = intraday_close[intraday_close.index >= cutoff]
                if len(recent) > 1:
                    change_24h = (float(recent.iloc[-1]) / float(recent.iloc[0]) - 1) * 100
    except Exception:
        pass
    if change_24h is None and len(history) > 1:
        change_24h = (float(history["close"].iloc[-1]) / float(history["close"].iloc[-2]) - 1) * 100

    return {
        "asset": asset, "price": float(quote), "quote_time": quote_time,
        "quote_source": quote_source, "change_24h": change_24h,
        "history": history, "unit": cfg["unit"], "color": cfg["color"],
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_snapshot() -> dict:
    result = {}
    errors = {}
    for asset in ASSETS:
        try:
            result[asset] = fetch_asset_market_data(asset)
        except Exception as exc:
            errors[asset] = str(exc)
    return {"assets": result, "errors": errors, "fetched_at": _utc_now().isoformat()}


def load_macro_scores(asset: str) -> tuple[dict, str | None]:
    """Read the hourly macro contribution scores generated by update_weights.py."""
    if not WEIGHT_FILE.exists():
        return {}, None
    try:
        with WEIGHT_FILE.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        values = payload.get(asset.lower(), {}) or {}
        scores = {str(k): float(v) for k, v in values.items() if isinstance(v, (int, float))}
        modified = datetime.fromtimestamp(WEIGHT_FILE.stat().st_mtime, timezone.utc).isoformat()
        return scores, modified
    except Exception:
        return {}, None


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _daily_resolved_sample(asset: str) -> pd.DataFrame:
    """Use one independent observation per UTC issue day and forecast horizon."""
    df = _read_forecasts()
    if df.empty:
        return pd.DataFrame()
    done = df[(df["asset"] == asset) & (df["status"] == "resolved")].copy()
    if done.empty:
        return done
    done["issued_at"] = pd.to_datetime(done["issued_at"], errors="coerce", utc=True)
    done["issue_day"] = done["issued_at"].dt.date
    return done.dropna(subset=["issue_day"]).sort_values("issued_at").drop_duplicates(
        subset=["horizon_days", "issue_day"], keep="first"
    )


def _calibration_for_asset(asset: str) -> dict:
    """Estimate historical directional bias by horizon; require 20 distinct days."""
    sample = _daily_resolved_sample(asset)
    result = {}
    if sample.empty:
        return result
    sample["error_pct"] = pd.to_numeric(sample["error_pct"], errors="coerce")
    for horizon, group in sample.groupby("horizon_days"):
        errors = group["error_pct"].dropna()
        bias = float(errors.mean()) / 100 if not errors.empty else 0.0
        result[int(horizon)] = {"bias": bias, "sample": int(len(errors))}
    return result


def build_forecast(asset: str, current_price: float, history: pd.DataFrame, macro: dict) -> dict:
    """Explainable ensemble: momentum, trend, mean reversion and macro score."""
    hist = history.copy().sort_values("timestamp")
    close = pd.to_numeric(hist["close"], errors="coerce").dropna()
    returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(close) < 60 or len(returns) < 40:
        raise RuntimeError(f"Historique insuffisant pour {asset}: {len(close)} séances, 60 minimum recommandées.")

    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
    rsi_delta = close.diff()
    gains = rsi_delta.clip(lower=0).rolling(14).mean().iloc[-1]
    losses = (-rsi_delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 50.0 if pd.isna(gains) or pd.isna(losses) else (100.0 if losses == 0 and gains > 0 else 50.0 if losses == 0 else float(100 - 100 / (1 + gains / losses)))

    momentum_10 = float(returns.tail(10).mean())
    momentum_20 = float(returns.tail(20).mean())
    trend_20_50 = float(np.log(max(sma20, 1e-9) / max(sma50, 1e-9)) / 20)
    mean_reversion = _clip(-np.log(max(current_price, 1e-9) / max(sma20, 1e-9)) * 0.04 / 20, -0.002, 0.002)
    vol20 = float(returns.tail(20).std())
    vol60 = float(returns.tail(60).std())
    daily_vol = max(vol20 if np.isfinite(vol20) else 0.0, 0.002)

    # Macro values are pre-normalized, asset-weighted contributions refreshed by update_weights.py.
    macro_total = float(sum(macro.values())) if macro else 0.0
    macro_daily = _clip(macro_total * 0.0015, -0.001, 0.001)
    momentum_component = _clip(0.55 * momentum_10 + 0.45 * momentum_20, -0.004, 0.004)
    trend_component = _clip(trend_20_50, -0.003, 0.003)
    drift = _clip(
        0.35 * momentum_component + 0.30 * trend_component
        + 0.20 * mean_reversion + 0.15 * macro_daily,
        -0.004, 0.004,
    )

    factors = {
        "Momentum 10–20 séances": momentum_component,
        "Tendance SMA20/SMA50": trend_component,
        "Retour vers moyenne SMA20": mean_reversion,
        "Facteurs macro pondérés": macro_daily,
        "RSI 14": (rsi - 50.0) / 50.0,
        "Volatilité 20 séances": daily_vol,
    }
    for macro_name, macro_value in macro.items():
        factors[f"Macro · {macro_name}"] = float(macro_value)
    calibration = _calibration_for_asset(asset)
    forecasts = []
    for horizon in HORIZONS:
        expected = float(current_price * np.exp(drift * horizon))
        band = 1.2816 * daily_vol * np.sqrt(horizon)
        calibration_row = calibration.get(horizon, {"bias": 0.0, "sample": 0})
        # Learn from the full archived error history; start cautiously after 5 resolved days.
        calibration_correction = _clip(-calibration_row["bias"], -0.05, 0.05) if calibration_row["sample"] >= 5 else 0.0
        expected *= (1.0 + calibration_correction)
        lower = float(current_price * np.exp(drift * horizon - band)) * (1.0 + calibration_correction)
        upper = float(current_price * np.exp(drift * horizon + band)) * (1.0 + calibration_correction)
        forecasts.append({
            "horizon_days": horizon,
            "target_date": (_utc_now().date() + timedelta(days=horizon)).isoformat(),
            "predicted_price": expected,
            "p10": lower,
            "p90": upper,
            "expected_return_pct": (expected / current_price - 1) * 100,
            "calibration_bias_pct": calibration_row["bias"] * 100,
            "calibration_sample": calibration_row["sample"],
        })

    return {
        "asset": asset, "current_price": float(current_price), "sma20": sma20,
        "sma50": sma50, "sma200": sma200, "rsi14": rsi,
        "momentum10_pct": (np.exp(momentum_10) - 1) * 100,
        "momentum20_pct": (np.exp(momentum_20 * 20) - 1) * 100,
        "volatility20_pct": daily_vol * 100, "volatility60_pct": vol60 * 100 if np.isfinite(vol60) else None,
        "daily_drift": drift, "macro_total": macro_total, "factors": factors,
        "forecasts": forecasts, "history": hist,
    }


def _read_forecasts() -> pd.DataFrame:
    if not FORECAST_FILE.exists() or FORECAST_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    try:
        df = pd.read_csv(FORECAST_FILE)
        for col in ["issued_at", "target_date", "resolved_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        # Older journal rows may have blank date columns even though forecast_id
        # still contains: asset|UTC issue timestamp|horizon_days.
        repaired_dates = False
        if "forecast_id" in df.columns and "horizon_days" in df.columns:
            parts = df["forecast_id"].astype(str).str.extract(r"^[^|]+\|([^|]+)\|(\d+)$")
            issue_from_id = pd.to_datetime(parts[0], errors="coerce", utc=True)
            if "issued_at" in df.columns:
                missing_issue = df["issued_at"].isna() & issue_from_id.notna()
                if missing_issue.any():
                    df.loc[missing_issue, "issued_at"] = issue_from_id.loc[missing_issue]
                    repaired_dates = True
            else:
                df["issued_at"] = issue_from_id
                repaired_dates = True

            horizon = pd.to_numeric(df["horizon_days"], errors="coerce")
            target_from_issue = df["issued_at"].dt.normalize() + pd.to_timedelta(horizon, unit="D")
            if "target_date" in df.columns:
                missing_target = df["target_date"].isna() & target_from_issue.notna()
                if missing_target.any():
                    df.loc[missing_target, "target_date"] = target_from_issue.loc[missing_target]
                    repaired_dates = True
            else:
                df["target_date"] = target_from_issue
                repaired_dates = True

        df.attrs["repaired_dates"] = repaired_dates
        return df
    except Exception:
        return pd.DataFrame(columns=FORECAST_COLUMNS)


def save_forecasts(asset: str, forecast: dict, issued_at: datetime | None = None) -> int:
    """Append one immutable forecast cycle per asset/horizon; never replace an old call."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    issued_at = issued_at or _utc_now()
    # Exactly one immutable forecast cycle per asset and UTC calendar day.
    issue_day = issued_at.date().isoformat()
    existing = _read_forecasts()
    rows = []
    for item in forecast["forecasts"]:
        forecast_id = f"{asset.lower()}|{issue_day}|{item['horizon_days']}"
        if not existing.empty and forecast_id in set(existing["forecast_id"].astype(str)):
            continue
        rows.append({
            "forecast_id": forecast_id, "asset": asset, "issued_at": issued_at.isoformat(),
            "target_date": item["target_date"], "horizon_days": item["horizon_days"],
            "model_version": "ensemble_v1", "current_price": forecast["current_price"],
            "predicted_price": item["predicted_price"], "p10": item["p10"], "p90": item["p90"],
            "expected_return_pct": item["expected_return_pct"], "daily_drift": forecast["daily_drift"],
            "daily_volatility": forecast["volatility20_pct"] / 100,
            "calibration_bias_pct": item.get("calibration_bias_pct", 0.0),
            "calibration_sample": item.get("calibration_sample", 0),
            "factor_json": json.dumps(forecast["factors"], ensure_ascii=False),
            "status": "pending", "actual_price": np.nan, "resolved_at": "",
            "error_pct": np.nan, "direction_correct": "",
        })
    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["forecast_id"], keep="first")
        combined.to_csv(FORECAST_FILE, index=False)
    return len(rows)


def _apply_archived_daily_forecast(result: dict, asset: str) -> dict:
    """Use today's first published forecast in the UI; never display a recalculated replacement."""
    log = _read_forecasts()
    if log.empty:
        return result
    issued = pd.to_datetime(log["issued_at"], errors="coerce", utc=True)
    today = pd.Timestamp.now(tz="UTC").date()
    mask = (log["asset"].astype(str) == asset) & (issued.dt.date == today)
    rows = log.loc[mask].copy()
    if rows.empty:
        return result
    rows["horizon_days"] = pd.to_numeric(rows["horizon_days"], errors="coerce")
    rows = rows.sort_values("issued_at").drop_duplicates("horizon_days", keep="first")
    result = dict(result)
    result["forecast_origin_price"] = float(rows.iloc[0]["current_price"])
    result["forecasts"] = [{
        "horizon_days": int(row["horizon_days"]),
        "target_date": pd.to_datetime(row["target_date"], utc=True).date().isoformat(),
        "predicted_price": float(row["predicted_price"]),
        "p10": float(row["p10"]),
        "p90": float(row["p90"]),
        "expected_return_pct": float(row["expected_return_pct"]),
        "calibration_bias_pct": float(row.get("calibration_bias_pct", 0) or 0),
        "calibration_sample": int(row.get("calibration_sample", 0) or 0),
    } for _, row in rows.iterrows()]
    return result


def resolve_forecasts(market_data: dict) -> int:
    """Resolve matured forecasts with the first daily close on/after the target date."""
    df = _read_forecasts()
    if df.empty:
        return 0
    changed = 0
    repaired_dates = bool(df.attrs.get("repaired_dates", False))
    now = _utc_now()
    for asset, payload in market_data.items():
        hist = payload["history"].copy()
        hist["day"] = pd.to_datetime(hist["timestamp"], utc=True).dt.date
        hist = hist[hist["day"] < now.date()].sort_values("day")
        pending = df[(df["asset"] == asset) & (df["status"] == "pending")]
        for idx, row in pending.iterrows():
            target = pd.to_datetime(row["target_date"], utc=True).date()
            candidates = hist[hist["day"] >= target]
            if candidates.empty:
                continue
            actual_row = candidates.iloc[0]
            # Gold is closed on weekends; cap the matching delay to three calendar days.
            if (actual_row["day"] - target).days > 3:
                continue
            actual = float(actual_row["close"])
            predicted = float(row["predicted_price"])
            origin = float(row["current_price"])
            direction_correct = ((predicted >= origin) == (actual >= origin))
            df.at[idx, "status"] = "resolved"
            df.at[idx, "actual_price"] = actual
            df.at[idx, "resolved_at"] = now.isoformat()
            df.at[idx, "error_pct"] = (predicted / actual - 1) * 100 if actual else np.nan
            df.at[idx, "direction_correct"] = bool(direction_correct)
            changed += 1
    if changed or repaired_dates:
        df.to_csv(FORECAST_FILE, index=False)
    return changed


def _metrics_for_asset(asset: str) -> pd.DataFrame:
    df = _read_forecasts()
    if df.empty:
        return pd.DataFrame()
    done = _daily_resolved_sample(asset)
    if done.empty:
        return pd.DataFrame()
    done["abs_error"] = pd.to_numeric(done["error_pct"], errors="coerce").abs()
    done["direction_correct"] = done["direction_correct"].astype(str).str.lower().isin(["true", "1"])
    return done.groupby("horizon_days").agg(
        forecasts=("forecast_id", "count"),
        mape_pct=("abs_error", "mean"),
        directional_accuracy=("direction_correct", "mean"),
    ).reset_index()


def _recent_forecast_comparison(result: dict, days: int | None = None) -> pd.DataFrame:
    """Compare the latest archived one-day forecast with the first observed close."""
    columns = ["Date cible", "Date réelle", "Prévision", "Réel", "Écart (%)"]
    log = _read_forecasts()
    if log.empty or "asset" not in log.columns:
        return pd.DataFrame(columns=columns)

    asset = result.get("asset")
    forecasts = log[log["asset"].astype(str).str.lower() == str(asset).lower()].copy()
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    forecasts["horizon_days"] = pd.to_numeric(forecasts["horizon_days"], errors="coerce")
    forecasts = forecasts[forecasts["horizon_days"] == 1].copy()
    forecasts["issued_at"] = pd.to_datetime(forecasts["issued_at"], errors="coerce", utc=True)
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"], errors="coerce", utc=True)
    forecasts["target_day"] = forecasts["target_date"].dt.tz_convert(None).dt.normalize()
    forecasts["issued_day"] = forecasts["issued_at"].dt.tz_convert(None).dt.normalize()
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    valid = (
        forecasts["issued_at"].notna()
        & forecasts["target_day"].notna()
        & (forecasts["issued_day"] < forecasts["target_day"])
        & (forecasts["target_day"] <= today)
    )
    if days is not None:
        start_day = today - pd.Timedelta(days=max(days - 1, 0))
        valid &= forecasts["target_day"] >= start_day
    forecasts = forecasts[valid].copy()
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    # Use the last forecast issued before each target day, not a forecast
    # recalculated after the outcome was already known.
    forecasts = forecasts.sort_values("issued_at").drop_duplicates("target_day", keep="last")

    history = result["history"].copy()
    history["day"] = pd.to_datetime(history["timestamp"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history.dropna(subset=["day", "close"]).sort_values("day")
    if history.empty:
        return pd.DataFrame(columns=columns)

    fx = result.get("fx_history")
    if isinstance(fx, pd.DataFrame) and not fx.empty:
        fx = fx.copy()
        fx["day"] = pd.to_datetime(fx["timestamp"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        fx["eurusd"] = pd.to_numeric(fx["eurusd"], errors="coerce")
        fx = fx.dropna(subset=["day", "eurusd"]).sort_values("day")

    rows = []
    for row in forecasts.itertuples():
        target_day = row.target_day
        observed = history[history["day"] >= target_day]
        if observed.empty:
            continue
        actual_row = observed.iloc[0]
        actual_day = actual_row["day"]
        if (actual_day - target_day).days > 3:
            continue
        actual_price = float(actual_row["close"])
        if actual_day == today:
            actual_price = float(result["current_price"])

        predicted = float(row.predicted_price)
        # Archived gold forecasts are stored in USD/troy oz; convert them
        # with the EUR/USD rate corresponding to the observed session.
        if asset == "Gold":
            eurusd = None
            if isinstance(fx, pd.DataFrame) and not fx.empty:
                fx_row = fx[fx["day"] <= actual_day]
                if not fx_row.empty:
                    eurusd = float(fx_row.iloc[-1]["eurusd"])
            if eurusd is None or eurusd <= 0:
                eurusd = result.get("fx_rate")
            if eurusd and eurusd > 0:
                predicted = predicted / eurusd / 31.1034768

        rows.append({
            "Date cible": target_day.strftime("%d/%m/%Y"),
            "Date réelle": actual_day.strftime("%d/%m/%Y"),
            "_day": actual_day,
            "Prévision": predicted,
            "Réel": actual_price,
            "Écart (%)": (predicted / actual_price - 1) * 100 if actual_price else np.nan,
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    comparison = pd.DataFrame(rows).sort_values("_day")
    # Weekend gold targets can resolve against the same next trading session.
    comparison = comparison.drop_duplicates("_day", keep="last")
    return comparison.drop(columns="_day").reset_index(drop=True)


def _forecast_chart(result: dict) -> go.Figure:
    hist = result["history"].copy()
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist = hist.dropna(subset=["timestamp"]).sort_values("timestamp")
    # Display window: November 2026 onward. Older data remains available to
    # the forecasting and calibration engine; only this chart is cropped.
    chart_start = pd.Timestamp("2026-11-01")
    hist = hist[
        (hist["timestamp"].dt.normalize() >= chart_start)
        & (hist["timestamp"].dt.normalize() < today)
    ]
    hist = pd.concat([
        hist[["timestamp", "close"]],
        pd.DataFrame([{"timestamp": today, "close": result["current_price"]}]),
    ], ignore_index=True)

    comparison = _recent_forecast_comparison(result)
    if not comparison.empty:
        comparison_days_all = pd.to_datetime(comparison["Date réelle"], format="%d/%m/%Y")
        comparison = comparison.loc[comparison_days_all >= chart_start].copy()
    forecast_color = result.get("color", "#D6A84B")

    # Build the future path from the same 1/3/7/14/30-day values shown in
    # the forecast table, so the chart and table cannot contradict each other.
    future_rows = sorted(result.get("forecasts", []), key=lambda item: int(item["horizon_days"]))
    anchor_days = [0] + [int(item["horizon_days"]) for item in future_rows]
    forecast_origin = float(result.get("forecast_origin_price", result["current_price"]))
    anchor_values = [forecast_origin] + [float(item["predicted_price"]) for item in future_rows]
    anchor_low = [forecast_origin] + [float(item["p10"]) for item in future_rows]
    anchor_high = [forecast_origin] + [float(item["p90"]) for item in future_rows]
    future_day_numbers = list(range(0, 31))
    values = np.interp(future_day_numbers, anchor_days, anchor_values)
    low = np.interp(future_day_numbers, anchor_days, anchor_low)
    high = np.interp(future_day_numbers, anchor_days, anchor_high)
    future_dates = [today + pd.Timedelta(days=i) for i in future_day_numbers]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["close"], name="Cours réel",
        mode="lines+markers", line=dict(color="#64748B", width=2.5),
        marker=dict(size=5),
    ))

    if not comparison.empty:
        comparison_days = pd.to_datetime(comparison["Date réelle"], format="%d/%m/%Y")
        fig.add_trace(go.Scatter(
            x=comparison_days, y=comparison["Prévision"],
            name="Prévision enregistrée (passé)", mode="lines+markers",
            line=dict(color=forecast_color, width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            customdata=np.column_stack([comparison["Date cible"], comparison["Écart (%)"]]),
            hovertemplate="Date réelle : %{x|%d/%m/%Y}<br>Prévision : %{y:,.2f}<br>Date cible : %{customdata[0]}<br>Écart : %{customdata[1]:+.2f}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=future_dates, y=high, line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=low, line=dict(width=0), fill="tonexty",
        fillcolor="rgba(84,153,211,0.16)", name="Fourchette future P10–P90",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=values, name="Prévision future",
        mode="lines", line=dict(color=forecast_color, width=3, dash="dash"),
        showlegend=False,
    ))
    fig.add_vline(
        x=today, line_dash="dot", line_color="#94A3B8",
        annotation_text="Aujourd'hui", annotation_position="top",
    )
    fig.update_layout(
        title=f"{result['asset']} — prévisions passées vs cours réel et projection future",
        height=490, template="plotly_white", hovermode="x unified",
        xaxis_title="Date", yaxis_title=result.get("unit", "USD"),
        margin=dict(l=20, r=20, t=65, b=25),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def _gold_usd_to_eur_per_gram() -> tuple[float, pd.DataFrame]:
    """Return latest USD->EUR rate and aligned daily FX history for gold conversion."""
    fx_raw = yf.Ticker("EURUSD=X").history(period="2y", interval="1d", auto_adjust=True)
    fx = _clean_history(fx_raw).rename(columns={"close": "eurusd"})
    if fx.empty:
        raise RuntimeError("Taux EUR/USD indisponible : conversion de l'or en EUR/g impossible.")
    fx = fx.sort_values("timestamp")
    rate = float(fx["eurusd"].iloc[-1])
    if not np.isfinite(rate) or rate <= 0:
        raise RuntimeError("Taux EUR/USD invalide.")
    return rate, fx


def _gold_display_result(result: dict, eurusd: float, fx_history: pd.DataFrame) -> dict:
    """Convert display-only gold values; persisted model data remains USD/oz."""
    out = dict(result)
    hist = result["history"].copy().sort_values("timestamp")
    fx = fx_history.copy().sort_values("timestamp")
    hist = pd.merge_asof(hist, fx, on="timestamp", direction="backward", tolerance=pd.Timedelta(days=5))
    hist = hist.dropna(subset=["eurusd"])
    hist["close"] = hist["close"] / hist["eurusd"] / 31.1034768
    out["history"] = hist[["timestamp", "close"]].copy()
    scale = 1.0 / eurusd / 31.1034768
    out["current_price"] = result["current_price"] * scale
    if result.get("forecast_origin_price") is not None:
        out["forecast_origin_price"] = result["forecast_origin_price"] * scale
    out["sma20"] = result["sma20"] * scale
    out["sma50"] = result["sma50"] * scale
    out["sma200"] = result["sma200"] * scale
    out["forecasts"] = [
        {**row, "predicted_price": row["predicted_price"] * scale,
         "p10": row["p10"] * scale, "p90": row["p90"] * scale}
        for row in result["forecasts"]
    ]
    out["unit"] = "EUR/g"
    out["fx_rate"] = eurusd
    out["fx_history"] = fx_history[["timestamp", "eurusd"]].copy()
    out["color"] = "#D6A84B"
    return out


def render_market_forecast_dashboard() -> None:
    st.title("🪙 Gold & Bitcoin | Market Intelligence & Forecast")
    st.caption("Une prévision immuable par jour · historique complet des prévisions archivées comparé aux cours réels · calibration sur les erreurs passées.")
    snapshot = fetch_market_snapshot()
    assets = snapshot["assets"]
    if not assets:
        st.error("Impossible de récupérer les cours actuels. Les données indisponibles ne sont pas remplacées par des valeurs fictives.")
        for name, error in snapshot["errors"].items():
            st.caption(f"{name} : {error}")
        return

    market_for_resolution = {name: item for name, item in assets.items()}
    resolve_forecasts(market_for_resolution)

    tabs = st.tabs(list(assets.keys()))
    for tab, asset in zip(tabs, assets.keys()):
        with tab:
            market = assets[asset]
            macro, macro_updated = load_macro_scores(asset)
            try:
                result = build_forecast(asset, market["price"], market["history"], macro)
                save_forecasts(asset, result)
                result = _apply_archived_daily_forecast(result, asset)
                if asset == "Gold":
                    eurusd, fx_history = _gold_usd_to_eur_per_gram()
                    display_result = _gold_display_result(result, eurusd, fx_history)
                    display_market_price = display_result["current_price"]
                    display_unit = "EUR/g"
                else:
                    display_result = result
                    display_market_price = market["price"]
                    display_unit = market["unit"]
            except Exception as exc:
                st.error(f"Prévision indisponible pour {asset} : {exc}")
                continue

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cours actuel", f"{display_market_price:,.2f} {display_unit}", f"{market['change_24h']:+.2f}%" if market["change_24h"] is not None else None)
            c2.metric("RSI 14", f"{result['rsi14']:.1f}", "Surachat" if result["rsi14"] >= 70 else "Survente" if result["rsi14"] <= 30 else "Zone neutre")
            c3.metric("Volatilité 20 séances", f"{result['volatility20_pct']:.2f}%")
            c4.metric("Tendance", "Haussière" if result["daily_drift"] > 0.00015 else "Baissière" if result["daily_drift"] < -0.00015 else "Neutre")

            quote_time = market["quote_time"]
            if isinstance(quote_time, pd.Timestamp):
                quote_time = quote_time.to_pydatetime()
            if quote_time and quote_time.tzinfo is None:
                quote_time = quote_time.replace(tzinfo=timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - quote_time).total_seconds() / 60 if quote_time else None
            freshness = f"Cours reçu : {quote_time.astimezone().strftime('%d/%m/%Y %H:%M:%S %Z')}" if quote_time else "Horodatage indisponible"
            st.caption(f"{freshness} · Source : {market['quote_source']} · {display_unit}" + (f" · Conversion EUR/USD : {eurusd:.5f} · 1 oz troy = 31,1034768 g" if asset == "Gold" else ""))
            if age_minutes is not None and age_minutes > 60:
                st.warning(f"Le dernier cours exploitable date de {age_minutes/60:.1f} h : interpréter la projection avec prudence.")

            st.plotly_chart(_forecast_chart({**display_result, "asset": asset, "unit": display_unit, "color": market["color"]}), use_container_width=True)


    st.divider()
    from newsletter_app import fx_chf_page
    fx_chf_page()


def run_batch() -> None:
    """Hourly maintenance: resolve matured forecasts; publish at most one immutable cycle per UTC day."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    market = {}
    for asset in ASSETS:
        try:
            market[asset] = fetch_asset_market_data(asset)
        except Exception as exc:
            print(f"[market_forecast] {asset}: {exc}")
    if not market:
        raise RuntimeError("Aucun cours de marché disponible; aucun chiffre de remplacement n'est généré.")
    resolve_forecasts(market)
    journal = _read_forecasts()
    today = _utc_now().date()
    for asset, payload in market.items():
        if not journal.empty:
            issued = pd.to_datetime(journal["issued_at"], errors="coerce", utc=True)
            already_published = (
                (journal["asset"].astype(str) == asset)
                & (issued.dt.date == today)
            ).any()
            if already_published:
                print(f"[market_forecast] {asset}: prévision du {today} déjà publiée; aucune réécriture.")
                continue
        macro, _ = load_macro_scores(asset)
        result = build_forecast(asset, payload["price"], payload["history"], macro)
        count = save_forecasts(asset, result)
        print(f"[market_forecast] {asset}: {payload['price']:.2f} USD; {count} nouvelles prévisions sauvegardées.")
    print(f"[market_forecast] Journal immuable: {FORECAST_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Generate and persist forecasts for scheduled runs.")
    args = parser.parse_args()
    if args.batch:
        run_batch()
