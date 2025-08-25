#!/usr/bin/env python3
"""
update_weights.py
Public-sources-only dynamic weights updater for Gold & Bitcoin.

- No API keys required.
- Robust to format changes: multiple fallbacks, schema detection.
- All outputs are in [-1, 1] (directional impact) and plain Python floats.
- Will not fail the workflow: on error uses last-good or neutral 0.0.

Indicators written for each asset:
    inflation, real_rates, bond_yields, energy_prices,
    usd_strength, liquidity, equity_flows, regulation,
    adoption, currency_instability, recession_probability,
    tail_risk_event, geopolitics

Notes:
- Some indicators are “shared macro states” mapped to both assets.
- Signs reflect *impact* on asset (e.g., stronger USD => negative for both).
"""

import os
import io
import re
import csv
import math
import time
import json
import yaml
import random
import zipfile
import datetime as dt
from typing import Dict, Optional, Tuple, List

import requests
import pandas as pd
from xml.etree import ElementTree as ET

# -----------------------------
# File paths
# -----------------------------
WEIGHT_FILE = "weight.yaml"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "weights-updater/1.0 (public-sources)"})
REQ_TIMEOUT = 15


# -----------------------------
# Small utilities
# -----------------------------
def to_float(x, default=0.0) -> float:
    try:
        # Ensure plain python float (no numpy)
        return float(x)
    except Exception:
        return float(default)


def clamp(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))


def pct_change(new: float, old: float) -> float:
    if old is None or old == 0 or pd.isna(old):
        return 0.0
    return (new - old) / abs(old)


def rolling_vol(series: List[float]) -> float:
    s = pd.Series(series).dropna()
    if s.size < 2:
        return 0.0
    return float(s.pct_change().dropna().std())


def soft_normalize(value: float, center: float = 0.0, scale: float = 1.0, max_abs: float = 1.0) -> float:
    """
    Simple squashing to [-max_abs, max_abs].
    """
    if scale == 0:
        return 0.0
    z = (value - center) / scale
    return clamp(z, -max_abs, max_abs)


def robust_get(url: str, as_bytes=False) -> Optional[bytes]:
    try:
        r = SESSION.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200 or not r.content:
            return None
        return r.content if as_bytes else r.content
    except Exception:
        return None


def load_existing_weights(path=WEIGHT_FILE) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return data
        except Exception:
            return {}
    return {}


def safe_dump_yaml(data: Dict, path=WEIGHT_FILE):
    # Convert all numerics to plain python floats
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (int, float)):
            return float(obj)
        return obj

    clean = sanitize(data)
    with open(path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)


# -----------------------------
# 1) Inflation (HICP YoY) – ECB SDW (no key)
# -----------------------------
def fetch_inflation_yoy() -> Optional[float]:
    """
    ECB SDW: Euro area HICP, annual rate YoY
    Series: ICP.M.U2.N.000000.4.ANR
    """
    url = "https://sdw-wsrest.ecb.europa.eu/service/data/ICP/M.U2.N.000000.4.ANR?detail=dataonly"
    try:
        content = robust_get(url, as_bytes=True)
        if not content:
            return None
        # Parse SDMX-ML XML
        root = ET.fromstring(content)
        # Find last Obs value
        ns = {"generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}
        obs = root.findall(".//generic:Obs", ns)
        if not obs:
            return None
        last = obs[-1]
        val = last.find(".//generic:ObsValue", ns)
        if val is None:
            return None
        v = val.attrib.get("value") or val.attrib.get("OBS_VALUE")
        if v is None:
            return None
        return to_float(v)
    except Exception:
        return None


# -----------------------------
# 2) USD strength – ECB eurofxref-hist.csv
# DXY-like proxy using (EUR, JPY, GBP, CAD, SEK, CHF) weights
# -----------------------------
def fetch_usd_strength_proxy() -> Optional[float]:
    """
    Build a DXY-like proxy from ECB daily FX (EUR base):
    We convert to USD index where higher = stronger USD.
    """
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
    try:
        content = robust_get(url, as_bytes=True)
        if not content:
            return None
        df = pd.read_csv(io.BytesIO(content))
        # Ensure we have needed currencies
        needed = ["USD", "JPY", "GBP", "CAD", "SEK", "CHF"]
        for c in needed:
            if c not in df.columns:
                return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        latest = df.iloc[-1]
        prev = df.iloc[-6] if len(df) >= 7 else df.iloc[0]

        # Convert EUR base to USD-based rates (USD per currency) approximations
        # ECB CSV gives X per EUR; USD column is USD per EUR directly.
        # DXY approx (weights): EUR 57.6, JPY 13.6, GBP 11.9, CAD 9.1, SEK 4.2, CHF 3.6
        # We'll form USD strength: higher USD per EUR (USD column) => stronger USD.
        def index_from_row(row):
            try:
                eur_usd = to_float(row["USD"])
                eur_jpy = to_float(row["JPY"])
                eur_gbp = to_float(row["GBP"])
                eur_cad = to_float(row["CAD"])
                eur_sek = to_float(row["SEK"])
                eur_chf = to_float(row["CHF"])
                # Normalize each leg relative to a baseline to avoid magnitude dominance
                # We use log-sum of weighted legs
                comp = (
                    0.576 * math.log(eur_usd) +
                    0.136 * math.log(eur_jpy) +
                    0.119 * math.log(eur_gbp) +
                    0.091 * math.log(eur_cad) +
                    0.042 * math.log(eur_sek) +
                    0.036 * math.log(eur_chf)
                )
                return math.exp(comp)
            except Exception:
                return None

        idx_latest = index_from_row(latest)
        idx_prev = index_from_row(prev)
        if idx_latest is None or idx_prev is None:
            return None

        # Normalize to directional [-1,1] from 1-week change
        ch = pct_change(idx_latest, idx_prev)
        # 1% weekly change ~ 0.5 impact
        return clamp(ch / 0.02, -1, 1)
    except Exception:
        return None


# -----------------------------
# 3) U.S. Treasury yields – curve & real rates proxy
# -----------------------------
def fetch_treasury_yields() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (ten_year, three_month, real_rate_proxy)
    Data: Treasury daily yield curve CSV (no key)
    """
    # All daily CSV with many years
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/Datasets/yield.csv?type=daily_treasury_yield_curve&field_tdr_date_value=all"
    try:
        content = robust_get(url, as_bytes=True)
        if not content:
            return (None, None, None)
        df = pd.read_csv(io.BytesIO(content))
        # Expected columns like: "Date","BC_3MONTH","BC_10YEAR"
        # Normalize names to upper
        cols = {c.upper(): c for c in df.columns}
        def get_col(*cands):
            for c in cands:
                if c in cols:
                    return cols[c]
            return None

        date_col = get_col("DATE")
        m3_col = get_col("BC_3MONTH", "3 MO")
        y10_col = get_col("BC_10YEAR", "10 YR")

        if not date_col or not m3_col or not y10_col:
            return (None, None, None)

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        last = df.iloc[-1]
        y10 = to_float(last[y10_col]) / 100.0  # convert % to decimal
        m3 = to_float(last[m3_col]) / 100.0

        # real rate proxy ≈ 10Y nominal – latest inflation (YoY approximated later, but we’ll fill here as nominal)
        # We'll return nominal here; final real rate computed after we fetch inflation.
        return (y10, m3, None)
    except Exception:
        return (None, None, None)


# -----------------------------
# 4) Energy prices – EIA Brent & WTI (XLS, no key)
# -----------------------------
def fetch_energy_prices() -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (brent, wti) latest daily close.
    EIA XLS public files.
    """
    urls = {
        "brent": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",
        "wti":   "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls",
    }
    out = {}
    for k, url in urls.items():
        try:
            content = robust_get(url, as_bytes=True)
            if not content:
                out[k] = None
                continue
            # Read excel (last sheet)
            xls = pd.ExcelFile(io.BytesIO(content))
            sheet = xls.sheet_names[-1]
            df = xls.parse(sheet_name=sheet, header=None)
            # Find last row with numeric price
            df.columns = ["date", "price"] + list(df.columns[2:])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[["date", "price"]].dropna()
            df = df[df["price"].apply(lambda z: isinstance(z, (int, float)))]
            if df.empty:
                out[k] = None
                continue
            last = df.dropna().iloc[-1]
            out[k] = to_float(last["price"])
        except Exception:
            out[k] = None
    return (out.get("brent"), out.get("wti"))


# -----------------------------
# 5) Stooq S&P 500 (liquidity & equity flows proxy)
# -----------------------------
def fetch_spx_from_stooq(days: int = 60) -> Optional[pd.DataFrame]:
    url = "https://stooq.com/q/d/l/?s=^spx&i=d"
    try:
        content = robust_get(url, as_bytes=True)
        if not content:
            return None
        df = pd.read_csv(io.BytesIO(content))
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").tail(days)
        return df
    except Exception:
        return None


# -----------------------------
# 6) CoinGecko (adoption/liquidity for crypto)
# -----------------------------
def fetch_coingecko_global() -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (btc_dominance_percent, total_market_cap_usd)
    """
    url = "https://api.coingecko.com/api/v3/global"
    try:
        content = robust_get(url, as_bytes=True)
        if not content:
            return (None, None)
        data = json.loads(content.decode("utf-8"))
        mkt = data.get("data", {})
        btc_dom = mkt.get("market_cap_percentage", {}).get("btc")
        total_cap = mkt.get("total_market_cap", {}).get("usd")
        return (to_float(btc_dom), to_float(total_cap))
    except Exception:
        return (None, None)


# -----------------------------
# 7) News sentiment (Reuters/BBC/Coindesk RSS)
# -----------------------------
RSS_FEEDS = {
    "geopolitics": [
        "https://feeds.reuters.com/Reuters/worldNews",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
    ],
    "regulation": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ],
    "tail_risk_event": [
        "https://feeds.reuters.com/reuters/marketsNews",
        "http://feeds.bbci.co.uk/news/rss.xml",
    ],
    "currency_instability": [
        "https://www.reutersagency.com/feed/?best-topics=foreign-exchange&post_type=best",
        "https://www.reuters.com/markets/currencies/rss",
    ],
}

POS_WORDS = set("""
approve approved approvals bullish growth recovery surge upside resilient
win breakthrough peace ceasefire stabilize stable easing relief record
""".split())

NEG_WORDS = set("""
ban bans crackdown restrict restrictive lawsuit probe investigation
conflict war escalation sanctions terror strike crash slump plunge
default downgrade turmoil crisis instability volatility shock recession
""".split())

def simple_sentiment_score(titles: List[str]) -> float:
    """
    Tiny lexicon-based sentiment in [-1,1].
    """
    if not titles:
        return 0.0
    pos = neg = 0
    for t in titles:
        s = (t or "").lower()
        pos += sum(1 for w in POS_WORDS if w in s)
        neg += sum(1 for w in NEG_WORDS if w in s)
    if pos == 0 and neg == 0:
        return 0.0
    raw = (pos - neg) / max(1, (pos + neg))
    return clamp(raw, -1, 1)


def fetch_rss_titles(url: str, limit: int = 30) -> List[str]:
    try:
        content = robust_get(url, as_bytes=True)
        if not content:
            return []
        root = ET.fromstring(content)
        titles = []
        for item in root.findall(".//item"):
            t = item.findtext("title")
            if t:
                titles.append(t.strip())
            if len(titles) >= limit:
                break
        if titles:
            return titles
        # Atom fallback
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            t = entry.findtext("{http://www.w3.org/2005/Atom}title")
            if t:
                titles.append(t.strip())
            if len(titles) >= limit:
                break
        return titles
    except Exception:
        return []


def topic_sentiment(topic_key: str) -> float:
    feeds = RSS_FEEDS.get(topic_key, [])
    titles = []
    for u in feeds:
        titles.extend(fetch_rss_titles(u, limit=25))
    return simple_sentiment_score(titles)


# -----------------------------
# Compose all indicators
# -----------------------------
def build_indicators() -> Dict[str, float]:
    # 1) Inflation (YoY %)
    inflation_yoy = fetch_inflation_yoy()  # e.g., ~2-10
    # Normalize: 2% ~ neutral; +/-3% away => +/-1
    inflation = soft_normalize((inflation_yoy or 0.0) - 2.0, center=0.0, scale=3.0, max_abs=1.0)

    # 2) Treasury yields / curve
    ten_y, three_m, _ = fetch_treasury_yields()
    # If any missing, set safe neutral
    if ten_y is None or three_m is None:
        ten_y = ten_y or 0.03
        three_m = three_m or 0.02
    curve = ten_y - three_m
    # Recession probability proxy: more inversion => more recession risk
    recession_probability = clamp(-soft_normalize(curve, center=0.01, scale=0.015, max_abs=1.0), -1, 1)

    # Real rates proxy = 10y nominal – inflation_yoy
    rr = (ten_y or 0.0) - (inflation_yoy or 0.0) / 100.0
    # Normalize real rates around 0
    real_rates = soft_normalize(rr, center=0.0, scale=0.01, max_abs=1.0)

    # Bond yields impact ~ magnitude of 10y level
    bond_yields = soft_normalize((ten_y or 0.0) - 0.025, center=0.0, scale=0.01, max_abs=1.0)

    # 3) USD strength (weekly change)
    usd_strength = fetch_usd_strength_proxy()
    if usd_strength is None:
        usd_strength = 0.0

    # 4) Energy prices (Brent & WTI)
    brent, wti = fetch_energy_prices()
    # Use pct change vs 30d median if we can form something from Stooq SPX as clock, else just relative to static center
    if brent is not None and wti is not None:
        # Higher energy => positive for Gold (inflation hedge), mixed/negative for BTC mining costs,
        # but here we produce a generic macro intensity in [-1,1] where higher crude => positive impact measure.
        # Normalize around Brent $80, scale $20.
        energy_prices = soft_normalize(((brent + wti) / 2.0) - 80.0, center=0.0, scale=20.0, max_abs=1.0)
    else:
        energy_prices = 0.0

    # 5) Liquidity & equity flows – from S&P momentum/volatility
    spx = fetch_spx_from_stooq(days=60)
    if spx is not None and not spx.empty:
        spx["ret"] = spx["Close"].pct_change()
        # Liquidity proxy: low realized vol => higher liquidity
        vol = float(spx["ret"].rolling(20).std().iloc[-1] or 0.0)
        # Map 1% daily vol -> -1 (illiquid), 0.3% -> +1 (liquid)
        liquidity = clamp(soft_normalize(0.003 - vol, center=0.0, scale=0.005, max_abs=1.0), -1, 1)

        # Equity flows proxy: 20-day momentum
        mom = float((spx["Close"].iloc[-1] - spx["Close"].iloc[-21]) / spx["Close"].iloc[-21]) if len(spx) > 21 else 0.0
        # +10% 20d -> +1, -10% -> -1
        equity_flows = clamp(mom / 0.10, -1, 1)
    else:
        liquidity = 0.0
        equity_flows = 0.0

    # 6) Crypto adoption & liquidity – CoinGecko
    btc_dom, total_cap = fetch_coingecko_global()
    # Adoption: BTC dominance rising => stronger adoption signal for BTC narrative
    adoption = clamp(((btc_dom or 45.0) - 45.0) / 10.0, -1, 1)
    # Tie liquidity slightly to crypto total market cap relative move, if we can’t form delta, use neutral 0
    # (We avoid storing previous state on disk; keeping neutral if unavailable)
    crypto_liq_boost = 0.0 if total_cap in (None, 0.0) else 0.1 * clamp(math.log10(total_cap) - 12.0, -1, 1)
    liquidity = clamp(liquidity + crypto_liq_boost, -1, 1)

    # 7) Sentiment-based indicators
    geopolitics = topic_sentiment("geopolitics")
    regulation = topic_sentiment("regulation")
    tail_risk_event = topic_sentiment("tail_risk_event")
    fx_sent = topic_sentiment("currency_instability")

    # Currency instability from FX volatility (ECB)
    # Compute 20d std of EURUSD, EURGBP, EURJPY from ECB csv
    try:
        fx_content = robust_get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv", as_bytes=True)
        if fx_content:
            fxd = pd.read_csv(io.BytesIO(fx_content))
            fxd["Date"] = pd.to_datetime(fxd["Date"])
            fxd = fxd.sort_values("Date").tail(40)
            vols = []
            for c in ["USD", "GBP", "JPY"]:
                if c in fxd.columns:
                    vols.append(fxd[c].pct_change().rolling(20).std().iloc[-1])
            fx_vol = float(pd.Series(vols).mean()) if vols else None
        else:
            fx_vol = None
    except Exception:
        fx_vol = None

    # Map FX vol to [-1,1]
    # 1% daily 20d vol -> high instability (~ +1), 0.2% -> low (~ -1)
    if fx_vol is not None and not math.isnan(fx_vol):
        currency_instability = clamp((fx_vol - 0.002) / 0.008, -1, 1)
    else:
        currency_instability = fx_sent  # fallback to sentiment

    # Recession probability already computed from curve inversion
    # Tail risk: combine sentiment with equity vol spike (if present)
    if spx is not None and "ret" in spx.columns and not spx["ret"].isna().all():
        realized_vol = float(spx["ret"].rolling(10).std().iloc[-1] or 0.0)
        vol_spike = clamp((realized_vol - 0.01) / 0.02, -1, 1)  # >1% daily std is stressy
        tail_risk_event = clamp(0.5 * tail_risk_event + 0.5 * vol_spike, -1, 1)

    # Pack indicators as *macro state* values
    return {
        "inflation": to_float(inflation),
        "real_rates": to_float(real_rates),
        "bond_yields": to_float(bond_yields),
        "energy_prices": to_float(energy_prices),
        "usd_strength": to_float(usd_strength),
        "liquidity": to_float(liquidity),
        "equity_flows": to_float(equity_flows),
        "regulation": to_float(regulation),
        "adoption": to_float(adoption),
        "currency_instability": to_float(currency_instability),
        "recession_probability": to_float(recession_probability),
        "tail_risk_event": to_float(tail_risk_event),
        "geopolitics": to_float(geopolitics),
    }


# -----------------------------
# Map macro state to asset weights (directional impact)
# If you want different signs per asset, adjust here.
# -----------------------------
ASSET_SIGNS = {
    "gold": {
        # Positive inflation/energy/liquidity often good for Gold (hedge)
        "inflation": +1,
        "real_rates": -1,         # higher real rates negative for gold
        "bond_yields": -1,        # higher yields weigh on gold
        "energy_prices": +0.5,    # energy up -> inflation hedge demand
        "usd_strength": -1,       # stronger USD negative for gold
        "liquidity": +0.3,        # modestly positive (risk appetite can lift commodities)
        "equity_flows": -0.3,     # strong equity flows may divert from gold
        "regulation": +0.0,
        "adoption": +0.1,         # small positive if general adoption/institutional interest rises
        "currency_instability": +0.5,
        "recession_probability": +0.5,
        "tail_risk_event": +0.5,
        "geopolitics": +0.4,
    },
    "bitcoin": {
        "inflation": +0.4,        # inflation supports BTC narrative
        "real_rates": -0.5,
        "bond_yields": -0.3,
        "energy_prices": -0.2,    # mining cost & risk appetite drag
        "usd_strength": -0.6,
        "liquidity": +0.7,        # risk-on/liquidity benefits BTC
        "equity_flows": +0.5,
        "regulation": +0.2,       # positive if headlines are favorable; negative if not (handled in value sign)
        "adoption": +0.8,
        "currency_instability": +0.5,
        "recession_probability": -0.2,  # risk-off may hurt BTC
        "tail_risk_event": -0.1,  # sudden shocks can be mixed/negative for BTC
        "geopolitics": +0.1,      # mild positive if it boosts BTC narrative
    },
}


def apply_asset_signs(macro: Dict[str, float], signs: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in macro.items():
        s = signs.get(k, 1.0)
        out[k] = clamp(to_float(v) * to_float(s), -1, 1)
    return out


# -----------------------------
# Main
# -----------------------------
def update_weights():
    existing = load_existing_weights()

    macro = build_indicators()

    gold = apply_asset_signs(macro, ASSET_SIGNS["gold"])
    btc = apply_asset_signs(macro, ASSET_SIGNS["bitcoin"])

    # If any indicator failed (None), keep last-good or set 0.0
    def merge_with_fallback(asset_key: str, new_vals: Dict[str, float]) -> Dict[str, float]:
        prev = (existing.get(asset_key) or {}).copy()
        out = {}
        for k in ASSET_SIGNS[asset_key].keys():
            val = new_vals.get(k, None)
            if val is None or pd.isna(val):
                # keep previous or neutral
                out[k] = to_float(prev.get(k, 0.0))
            else:
                out[k] = to_float(val)
        return out

    updated = {
        "gold": merge_with_fallback("gold", gold),
        "bitcoin": merge_with_fallback("bitcoin", btc),
    }

    safe_dump_yaml(updated, WEIGHT_FILE)
    print("weight.yaml updated successfully.")


if __name__ == "__main__":
    update_weights()
