"""
Gold & Bitcoin Predictive Matrix - Streamlit App (User-friendly UI)

Improved features in this version:
- Modern, friendly layout with colored metric cards and Plotly charts
- Manual and Auto modes with clearly separated UX
- Price charts for Gold & Bitcoin
- Colored expected price (green/red) and confidence indicators
- Preset scenario buttons and quick-reset

How to use:
1) Replace your app.py with this file in your GitHub repo for Streamlit Cloud.
2) Ensure weights.yaml is available in the repo or upload it via the sidebar.
3) Add API keys to Streamlit Secrets when using Auto mode (FRED_KEY, ALPHAVANTAGE_KEY).
4) requirements.txt should include: streamlit, pandas, pyyaml, requests, yfinance, plotly

"""

import streamlit as st
import pandas as pd
import yaml
import requests
import os
import yfinance as yf
import plotly.express as px
from datetime import datetime

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="Gold & Bitcoin Predictor", layout="wide")
st.title("📊 Gold & Bitcoin — Predictive Matrix")

# Drivers list
DRIVERS = [
    "geopolitics", "inflation", "real_rates", "usd_strength", "liquidity",
    "equity_flows", "bond_yields", "regulation", "adoption",
    "currency_instability", "recession_probability",
    "energy_prices", "tail_risk_event"
]

# ------------------------------
# Helpers: load weights
# ------------------------------
@st.cache_data
def load_weights(path="weights.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        # friendly default weights (simple, not optimized)
        default = {
            "gold": {k: (1.0 if k in ["inflation","real_rates","usd_strength","geopolitics","liquidity"] else 0.5) for k in DRIVERS},
            "bitcoin": {k: (1.2 if k in ["adoption","equity_flows","regulation","tail_risk_event"] else 0.6) for k in DRIVERS}
        }
        return default

# ------------------------------
# Market price fetchers
# ------------------------------
@st.cache_data(ttl=300)
def fetch_bitcoin_price():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_change=true")
        j = r.json()
        price = j["bitcoin"]["usd"]
        change24h = j["bitcoin"].get("usd_24h_change", None)
        return price, change24h
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def fetch_gold_price():
    try:
        t = yf.Ticker("GC=F")
        df = t.history(period="30d")
        if df.empty:
            return None, None, None
        last_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_close
        change = (last_close - prev_close) / prev_close * 100 if prev_close != 0 else None
        # return historical series for chart
        hist = df.reset_index()[['Date','Close']].rename(columns={'Close':'Gold_Close'})
        return last_close, change, hist
    except Exception:
        return None, None, None

@st.cache_data(ttl=300)
def fetch_btc_history(days=30):
    try:
        # Use CoinGecko simple range via market_chart? fallback to price only
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}")
        j = r.json()
        prices = j.get('prices', [])
        df = pd.DataFrame(prices, columns=['timestamp','price']).assign(
            Date=lambda d: pd.to_datetime(d['timestamp'], unit='ms')
        )[['Date','price']].rename(columns={'price':'BTC_Price'})
        return df
    except Exception:
        return pd.DataFrame()

# ------------------------------
# Auto fetch with real APIs (friendly and safe)
# ------------------------------

def auto_fetch():
    params = {}
    fred_key = st.secrets.get("FRED_KEY")

    # --- Inflation (US CPI from FRED) ---
    try:
        if fred_key:
            cpi = requests.get(
                f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={fred_key}&file_type=json"
            ).json()
            latest_cpi = float(cpi["observations"][-1]["value"])
            params["inflation"] = 2 if latest_cpi > 3 else (1 if latest_cpi > 2 else 0)
        else:
            params["inflation"] = 0
    except Exception:
        params["inflation"] = 0

    # --- Real Rates (10y nominal - 10y TIPS) ---
    try:
        if fred_key:
            t10 = requests.get(
                f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={fred_key}&file_type=json"
            ).json()
            tips10 = requests.get(
                f"https://api.stlouisfed.org/fred/series/observations?series_id=DFII10&api_key={fred_key}&file_type=json"
            ).json()
            real_rate = float(t10["observations"][-1]["value"]) - float(tips10["observations"][-1]["value"])
            params["real_rates"] = -2 if real_rate > 2 else (-1 if real_rate > 1 else 0)
        else:
            params["real_rates"] = 0
    except Exception:
        params["real_rates"] = 0

    # --- USD Strength (EURUSD proxy via Alpha Vantage) ---
    try:
        av_key = st.secrets.get("ALPHAVANTAGE_KEY")
        if av_key:
            dxy = requests.get(
                f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=EUR&apikey={av_key}"
            ).json()
            rate = float(dxy["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
            params["usd_strength"] = 2 if rate < 0.9 else (1 if rate < 0.95 else 0)
        else:
            params["usd_strength"] = 0
    except Exception:
        params["usd_strength"] = 0

    # --- Bitcoin adoption ---
    try:
        cg = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin").json()
        followers = cg.get("community_data", {}).get("twitter_followers", 0) or 0
        params["adoption"] = 2 if followers > 6000000 else (1 if followers > 5000000 else 0)
    except Exception:
        params["adoption"] = 0

    # Neutral placeholders for geopolitics & tail risk (can be wired to GDELT later)
    params["geopolitics"] = 0
    params["tail_risk_event"] = 0

    # Fill rest neutral
    for k in DRIVERS:
        if k not in params:
            params[k] = 0

    return params

# ------------------------------
# Scoring
# ------------------------------

def score_assets(params, weights):
    results = {}
    for asset in ["gold", "bitcoin"]:
        score = 0.0
        for k in params:
            w = weights.get(asset, {}).get(k, 0)
            score += params[k] * w
        # Normalize and cap
        score = max(min(score, 100), -100)
        direction = "Bullish" if score > 20 else "Bearish" if score < -20 else "Neutral"
        expected_return = round((score / 100) * (15 if asset == "gold" else 30), 2)
        results[asset] = {
            "score": round(score, 2),
            "direction": direction,
            "expected_return_pct": expected_return,
            "confidence_pct": int(min(abs(round(score)), 100))
        }
    return results

# ------------------------------
# Preset scenarios
# ------------------------------
PRESETS = {
    "Risk-On": {k: (1 if k in ['equity_flows','adoption'] else 0) for k in DRIVERS},
    "Risk-Off": {k: (1 if k in ['geopolitics','liquidity','tail_risk_event'] else 0) for k in DRIVERS},
    "High Inflation": {k: (2 if k=='inflation' else 0) for k in DRIVERS},
    "War/Geopolitical": {k: (2 if k=='geopolitics' else 0) for k in DRIVERS}
}

# ------------------------------
# UI Layout
# ------------------------------
weights = None
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Manual", "Auto (with APIs)"], index=0)
    uploaded = st.file_uploader("Upload custom weights.yaml", type=["yaml","yml"])
    if uploaded:
        try:
            weights = yaml.safe_load(uploaded)
            st.success("Weights loaded from upload.")
        except Exception:
            st.error("Failed to parse weights.yaml")
    else:
        try:
            weights = load_weights()
        except Exception:
            st.warning("Using default weights.")

    st.markdown("---")
    st.write("**Presets**")
    for p in PRESETS:
        if st.button(p):
            st.session_state['preset'] = p
    if st.button("Reset sliders"):
        st.session_state['reset'] = True

# Main
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Inputs & Scenario")
    if mode == "Manual":
        # Use sliders in two columns
        sliders = {}
        cols = st.columns(2)
        for i, d in enumerate(DRIVERS):
            default = 0
            if 'preset' in st.session_state and st.session_state['preset'] in PRESETS:
                default = PRESETS[st.session_state['preset']].get(d, 0)
            if 'reset' in st.session_state and st.session_state['reset']:
                default = 0
            with cols[i % 2]:
                sliders[d] = st.slider(d, -2, 2, default, key=f"s_{d}")
        params = sliders
    else:
        st.write("Auto mode — pulling live indicators (requires secrets for best results)")
        with st.spinner("Fetching live inputs..."):
            params = auto_fetch()
        st.json(params)

    st.markdown("---")
    st.write("**Driver notes**: values are on a -2..+2 scale (negative to positive impact on asset). Use presets to quickly test common scenarios.")

with col2:
    st.subheader("Market Snapshot")
    btc_price, btc_change = fetch_bitcoin_price()
    gold_price, gold_change, gold_hist = fetch_gold_price()

    g1, g2 = st.columns(2)
    if btc_price is not None:
        sign = "▲" if (btc_change or 0) > 0 else ("▼" if (btc_change or 0) < 0 else "—")
        g1.metric(label="Bitcoin (USD)", value=f"${btc_price:,.0f}", delta=f"{(btc_change or 0):.2f}% {sign}")
    else:
        g1.write("Bitcoin price: N/A")

    if gold_price is not None:
        signg = "▲" if (gold_change or 0) > 0 else ("▼" if (gold_change or 0) < 0 else "—")
        g2.metric(label="Gold Futures (GC=F)", value=f"${gold_price:,.2f}", delta=f"{(gold_change or 0):.2f}% {signg}")
    else:
        g2.write("Gold price: N/A")

# Compute score and expected prices
results = score_assets(params, weights)
for asset in results:
    cur = None
    if asset == 'bitcoin':
        cur = btc_price
    else:
        cur = gold_price
    if cur is not None:
        er = results[asset]['expected_return_pct']
        results[asset]['current_price_usd'] = round(cur,2)
        results[asset]['expected_price_usd'] = round(cur * (1 + er/100), 2)
    else:
        results[asset]['current_price_usd'] = None
        results[asset]['expected_price_usd'] = None

# Display results in beautiful cards
st.markdown("---")
st.subheader("Prediction Cards")
rc1, rc2 = st.columns(2)
with rc1:
    r = results['gold']
    color = "#2ecc71" if r['direction']=="Bullish" else ("#e74c3c" if r['direction']=="Bearish" else "#3498db")
    st.markdown(f"<div style='border-radius:12px;padding:16px;background:#f9f9fb;'>"
                f"<h3 style='color:{color}'>Gold — {r['direction']}</h3>"
                f"<p><b>Score:</b> {r['score']} &nbsp;&nbsp; <b>Confidence:</b> {r['confidence_pct']}%</p>"
                f"<p><b>Expected move (30d):</b> {r['expected_return_pct']}%</p>"
                f"<p><b>Current price:</b> ${r['current_price_usd'] if r['current_price_usd'] else 'N/A'}</p>"
                f"<h2 style='margin-top:6px;color:{color}'>Expected: ${r['expected_price_usd'] if r['expected_price_usd'] else 'N/A'}</h2>"
                f"</div>", unsafe_allow_html=True)

with rc2:
    r = results['bitcoin']
    color = "#2ecc71" if r['direction']=="Bullish" else ("#e74c3c" if r['direction']=="Bearish" else "#3498db")
    st.markdown(f"<div style='border-radius:12px;padding:16px;background:#f9f9fb;'>"
                f"<h3 style='color:{color}'>Bitcoin — {r['direction']}</h3>"
                f"<p><b>Score:</b> {r['score']} &nbsp;&nbsp; <b>Confidence:</b> {r['confidence_pct']}%</p>"
                f"<p><b>Expected move (30d):</b> {r['expected_return_pct']}%</p>"
                f"<p><b>Current price:</b> ${r['current_price_usd'] if r['current_price_usd'] else 'N/A'}</p>"
                f"<h2 style='margin-top:6px;color:{color}'>Expected: ${r['expected_price_usd'] if r['expected_price_usd'] else 'N/A'}</h2>"
                f"</div>", unsafe_allow_html=True)

# Charts area
st.markdown("---")
st.subheader("Charts & Visuals")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**Price history (30 days)**")
    # BTC history
    btc_hist = fetch_btc_history(30)
    if not btc_hist.empty:
        fig = px.line(btc_hist, x='Date', y='BTC_Price', title='Bitcoin (30d)', labels={'BTC_Price':'Price (USD)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Bitcoin history not available.")

with chart_col2:
    st.markdown("**Gold history (30 days)**")
    if gold_hist is not None and not gold_hist.empty:
        fig2 = px.line(gold_hist, x='Date', y='Gold_Close', title='Gold Futures (30d)', labels={'Gold_Close':'Price (USD)'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("Gold history not available.")

# Scenario bar chart of expected returns
st.markdown("---")
st.subheader("Scenario outcome")
sc_df = pd.DataFrame([{
    'asset': 'Gold',
    'expected_return_pct': results['gold']['expected_return_pct']
}, {
    'asset': 'Bitcoin',
    'expected_return_pct': results['bitcoin']['expected_return_pct']
}])
bar = px.bar(sc_df, x='asset', y='expected_return_pct', text='expected_return_pct', title='Expected 30-day return (%)')
st.plotly_chart(bar, use_container_width=True)

# Driver contributions table
with st.expander("Show driver inputs & contributions"):
    st.write(pd.Series(params, name='Input (-2..+2)'))
    # contribution = param * weight
    contrib = {}
    for asset in ['gold','bitcoin']:
        ws = weights.get(asset, {})
        contrib[asset] = {k: round(params.get(k,0) * ws.get(k,0),3) for k in DRIVERS}
    contrib_df = pd.DataFrame(contrib)
    st.dataframe(contrib_df)

# Download results
out_df = pd.DataFrame(results).T
st.download_button("Download prediction (CSV)", out_df.to_csv().encode(), "predictions.csv")

st.markdown("---")
st.caption(f"App updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} — Built for quick what-if analysis. Modify weights.yaml to tune behaviour.")
