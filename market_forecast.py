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
    "Gold": {"ticker": "GC=F", "history_ticker": "GC=F", "unit": "USD/oz", "color": "#D6A84B"},
    "Bitcoin": {"ticker": "BTC-USD", "history_ticker": "BTC-USD", "unit": "USD", "color": "#E58B35"},
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
        calibration_correction = _clip(-calibration_row["bias"], -0.025, 0.025) if calibration_row["sample"] >= 20 else 0.0
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
        return df
    except Exception:
        return pd.DataFrame(columns=FORECAST_COLUMNS)


def save_forecasts(asset: str, forecast: dict, issued_at: datetime | None = None) -> int:
    """Append one immutable forecast cycle per asset/horizon; never replace an old call."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    issued_at = issued_at or _utc_now()
    cycle = issued_at.replace(minute=0, second=0, microsecond=0)
    existing = _read_forecasts()
    rows = []
    for item in forecast["forecasts"]:
        forecast_id = f"{asset.lower()}|{cycle.isoformat()}|{item['horizon_days']}"
        if not existing.empty and forecast_id in set(existing["forecast_id"].astype(str)):
            continue
        rows.append({
            "forecast_id": forecast_id, "asset": asset, "issued_at": cycle.isoformat(),
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


def resolve_forecasts(market_data: dict) -> int:
    """Resolve matured forecasts with the first daily close on/after the target date."""
    df = _read_forecasts()
    if df.empty:
        return 0
    changed = 0
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
    if changed:
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


def _forecast_chart(result: dict) -> go.Figure:
    hist = result["history"].tail(180)
    current = result["current_price"]
    drift = result["daily_drift"]
    vol = result["volatility20_pct"] / 100
    dates = [pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) + pd.Timedelta(days=i) for i in range(1, 31)]
    values = [current * np.exp(drift * i) for i in range(1, 31)]
    bands = [1.2816 * vol * np.sqrt(i) for i in range(1, 31)]
    low = [current * np.exp(drift * i - bands[i-1]) for i in range(1, 31)]
    high = [current * np.exp(drift * i + bands[i-1]) for i in range(1, 31)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["close"], name="Historique réel", line=dict(color="#718096", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=low, line=dict(width=0), fill="tonexty", fillcolor="rgba(84,153,211,0.16)", name="Zone indicative P10–P90"))
    fig.add_trace(go.Scatter(x=dates, y=values, name="Projection centrale", line=dict(color=result.get("color", "#D6A84B"), width=3, dash="dash")))
    fig.add_hline(y=current, line_dash="dot", line_color="#9AA5B1", annotation_text="Cours de référence")
    fig.update_layout(
        title=f"{result['asset']} — historique et projection 30 jours",
        height=470, template="plotly_white", hovermode="x unified",
        xaxis_title="Date", yaxis_title=result.get("unit", "USD"),
        margin=dict(l=20, r=20, t=55, b=25), legend=dict(orientation="h", y=1.02),
    )
    return fig


def render_market_forecast_dashboard() -> None:
    st.title("🪙 Gold & Bitcoin | Market Intelligence & Forecast")
    st.caption("Cours de marché, facteurs explicatifs, projections probabilistes et suivi objectif des erreurs.")
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
            except Exception as exc:
                st.error(f"Prévision indisponible pour {asset} : {exc}")
                continue

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cours actuel", f"{market['price']:,.2f} {market['unit']}", f"{market['change_24h']:+.2f}%" if market["change_24h"] is not None else None)
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
            st.caption(f"{freshness} · Source : {market['quote_source']} · {market['unit']}")
            if age_minutes is not None and age_minutes > 60:
                st.warning(f"Le dernier cours exploitable date de {age_minutes/60:.1f} h : interpréter la projection avec prudence.")

            st.plotly_chart(_forecast_chart({**result, "unit": market["unit"], "color": market["color"]}), use_container_width=True)

            st.subheader("Projections par échéance")
            table = pd.DataFrame(result["forecasts"])
            table["predicted_price"] = table["predicted_price"].map(lambda x: f"{x:,.2f}")
            table["p10"] = table["p10"].map(lambda x: f"{x:,.2f}")
            table["p90"] = table["p90"].map(lambda x: f"{x:,.2f}")
            table["expected_return_pct"] = table["expected_return_pct"].map(lambda x: f"{x:+.2f}%")
            table["calibration"] = table.apply(
                lambda row: f"{row['calibration_bias_pct']:+.2f}% (n={int(row['calibration_sample'])})"
                if int(row["calibration_sample"]) >= 20 else "En apprentissage",
                axis=1,
            )
            table = table.rename(columns={
                "horizon_days": "Horizon (jours)", "target_date": "Date cible",
                "predicted_price": "Projection centrale", "p10": "Borne basse P10",
                "p90": "Borne haute P90", "expected_return_pct": "Variation estimée",
                "calibration": "Biais historique détecté",
            })
            st.dataframe(table, use_container_width=True, hide_index=True)

            left, right = st.columns([1.1, 1])
            with left:
                st.subheader("Facteurs qui influencent le modèle")
                factor_rows = []
                for label, value in result["factors"].items():
                    if "Volatilité" in label:
                        detail = f"{value*100:.2f}% / jour"
                    elif "RSI" in label:
                        detail = f"{50 + value*50:.1f}/100"
                    elif label.startswith("Macro ·"):
                        detail = f"{value:+.3f} (score pondéré)"
                    else:
                        detail = f"{value*100:+.3f}% / jour"
                    factor_rows.append({"Facteur": label, "Signal": detail})
                st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
                if macro:
                    st.caption(f"Scores macro pondérés issus de weight.yaml · dernière modification : {macro_updated or 'inconnue'}. Les rubriques adoption, réglementation et géopolitique restent des proxys, pas des mesures directes d'événements.")
                else:
                    st.warning("Aucun score macro disponible : la projection repose principalement sur les facteurs de marché.")
            with right:
                st.subheader("Suivi des performances")
                metrics = _metrics_for_asset(asset)
                if metrics.empty:
                    st.info("Pas encore assez de prévisions arrivées à échéance pour calculer la précision. Les prévisions restent enregistrées.")
                else:
                    st.dataframe(metrics.rename(columns={
                        "horizon_days": "Horizon (j)", "forecasts": "Échantillon",
                        "mape_pct": "Erreur absolue moyenne (%)",
                        "directional_accuracy": "Direction correcte",
                    }), use_container_width=True, hide_index=True)
                    st.caption("Précision calculée sur une seule prévision par jour et par échéance. Après 20 journées résolues, le modèle corrige automatiquement son biais historique par horizon (correction plafonnée à ±2,5 %).")

            st.caption("Méthode ensemble v1 : momentum, tendance SMA20/SMA50, retour vers la moyenne et score macro pondéré. Les intervalles P10–P90 représentent une plage indicative issue de la volatilité historique, pas une garantie.")
    st.divider()
    st.subheader("Journal des prévisions")
    log = _read_forecasts()
    if log.empty:
        st.info("Le journal sera alimenté dès le premier cycle de prévision.")
    else:
        display = log.sort_values("issued_at", ascending=False).head(40).copy()
        cols = ["asset", "issued_at", "target_date", "horizon_days", "current_price", "predicted_price", "actual_price", "error_pct", "status"]
        display = display[[c for c in cols if c in display.columns]]
        st.dataframe(display, use_container_width=True, hide_index=True)


def run_batch() -> None:
    """Hourly GitHub Actions entry point: persist forecasts and resolve matured ones."""
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
    for asset, payload in market.items():
        macro, _ = load_macro_scores(asset)
        result = build_forecast(asset, payload["price"], payload["history"], macro)
        count = save_forecasts(asset, result)
        print(f"[market_forecast] {asset}: {payload['price']:.2f} USD; {count} nouvelles prévisions sauvegardées.")
    print(f"[market_forecast] Journal: {FORECAST_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Generate and persist forecasts for scheduled runs.")
    args = parser.parse_args()
    if args.batch:
        run_batch()
