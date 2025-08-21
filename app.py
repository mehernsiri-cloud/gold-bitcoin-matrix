
import streamlit as st
import pandas as pd
import yaml
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Gold & Bitcoin Predictive Matrix", layout="wide")

st.title("🧠 Gold & Bitcoin Predictive Matrix")
st.caption("No-install web app — score macro drivers → get 30-day directional bias for Gold & Bitcoin")

# -----------------------------
# Defaults (you can swap via sidebar upload)
# -----------------------------
DEFAULT_WEIGHTS = {
    "gold": {
        "geopolitics": 0.18,
        "inflation": 0.12,
        "real_rates": -0.22,
        "usd_strength": -0.14,
        "liquidity": 0.10,
        "equity_flows": -0.06,
        "bond_yields": -0.08,
        "regulation": 0.00,
        "adoption": 0.02,
        "currency_instability": 0.12,
        "recession_probability": 0.12,
        "energy_prices": 0.02,
        "tail_risk_event": 0.20
    },
    "bitcoin": {
        "geopolitics": 0.08,
        "inflation": 0.10,
        "real_rates": -0.10,
        "usd_strength": -0.06,
        "liquidity": 0.18,
        "equity_flows": 0.12,
        "bond_yields": -0.06,
        "regulation": 0.18,
        "adoption": 0.20,
        "currency_instability": 0.10,
        "recession_probability": -0.04,
        "energy_prices": -0.02,
        "tail_risk_event": 0.06
    }
}

DRIVERS = [
    "geopolitics",
    "inflation",
    "real_rates",
    "usd_strength",
    "liquidity",
    "equity_flows",
    "bond_yields",
    "regulation",
    "adoption",
    "currency_instability",
    "recession_probability",
    "energy_prices",
    "tail_risk_event"
]

def compute_score(params, weights, asset):
    w = weights[asset]
    max_abs = sum(abs(v) for v in w.values()) * 2.0  # all drivers at +/-2
    raw = sum(w.get(k,0.0) * params.get(k,0) for k in DRIVERS)
    score = (raw / max_abs) * 100.0 if max_abs > 0 else 0.0
    return score

def proj_from_score(score, asset):
    alpha = 0.15 if asset == "gold" else 0.30  # % per score point over 30 days
    exp_30d = alpha * score
    direction = "Bullish" if score > 10 else ("Bearish" if score < -10 else "Neutral")
    confidence = round(0.50 + 0.50 * min(1.0, abs(score)/100.0), 2)
    return direction, exp_30d, confidence

def placeholder_auto_fetch():
    # Replace this with real APIs later.
    return {
        "geopolitics": 0,
        "inflation": 0,
        "real_rates": 0,
        "usd_strength": 0,
        "liquidity": 1,
        "equity_flows": 1,
        "bond_yields": 0,
        "regulation": 0,
        "adoption": 1,
        "currency_instability": 0,
        "recession_probability": -1,
        "energy_prices": 0,
        "tail_risk_event": 0
    }

# Sidebar: weights & mode
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Input mode", ["Manual (sliders)", "Auto-placeholder"], index=0)
    st.markdown("---")
    st.subheader("Weights")
    upl = st.file_uploader("Upload custom weights.yaml (optional)", type=["yaml", "yml"])
    if upl:
        try:
            custom_weights = yaml.safe_load(upl.read())
            weights = custom_weights
            st.success("Custom weights loaded.")
        except Exception as e:
            st.error(f"Failed to parse YAML: {e}")
            weights = DEFAULT_WEIGHTS
    else:
        weights = DEFAULT_WEIGHTS

    st.download_button("Download current weights.yaml",
                       data=yaml.safe_dump(weights).encode("utf-8"),
                       file_name="weights.yaml",
                       mime="text/yaml")

# Inputs
st.subheader("📥 Inputs (−2 to +2)")
if mode.startswith("Auto"):
    params = placeholder_auto_fetch()
    st.info("Auto mode: using placeholder inputs. Replace with real API logic in the code.")
else:
    cols = st.columns(4)
    params = {}
    for i, drv in enumerate(DRIVERS):
        with cols[i % 4]:
            params[drv] = st.slider(drv.replace("_", " ").title(), -2, 2, 0)

# Compute
gold_score = compute_score(params, weights, "gold")
btc_score  = compute_score(params, weights, "bitcoin")

gold_dir, gold_move, gold_conf = proj_from_score(gold_score, "gold")
btc_dir,  btc_move,  btc_conf  = proj_from_score(btc_score, "bitcoin")

# Display results
st.subheader("📊 Predictions (30 days)")
c1, c2 = st.columns(2)
with c1:
    st.metric("Gold", f"{gold_dir}", f"{gold_move:.2f}%")
    st.caption(f"Score: {gold_score:.1f} | Confidence: {gold_conf}")
with c2:
    st.metric("Bitcoin", f"{btc_dir}", f"{btc_move:.2f}%")
    st.caption(f"Score: {btc_score:.1f} | Confidence: {btc_conf}")

# Breakdown table
def contributions(params, weights, asset):
    rows = []
    for k in DRIVERS:
        rows.append({
            "driver": k,
            "input(-2..+2)": params.get(k, 0),
            "weight": weights[asset].get(k, 0.0),
            "contribution": weights[asset].get(k, 0.0) * params.get(k, 0)
        })
    df = pd.DataFrame(rows).sort_values("contribution", ascending=True)
    return df

st.markdown("---")
st.subheader("🔎 Driver-by-driver contributions")
gdf = contributions(params, weights, "gold")
bdf = contributions(params, weights, "bitcoin")

t1, t2 = st.columns(2)
with t1:
    st.write("**Gold**")
    st.dataframe(gdf, use_container_width=True)
with t2:
    st.write("**Bitcoin**")
    st.dataframe(bdf, use_container_width=True)

st.markdown("---")
st.caption("Tip: Click the sidebar to upload custom weights or switch to Auto mode. Replace placeholder fetch with real APIs inside the code.")
st.caption(f"Last updated: {datetime.utcnow().isoformat()}Z")
