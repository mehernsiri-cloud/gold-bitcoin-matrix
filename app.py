# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import yaml
import requests

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gold, Bitcoin & Jobs Dashboard", layout="wide")

# ------------------------------
# DATA FILES
# ------------------------------
DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# LOAD DATA SAFELY
# ------------------------------
def load_csv_safe(path, default_cols):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=True)
    else:
        df = pd.DataFrame(columns=default_cols)
    return df

df_pred = load_csv_safe(PREDICTION_FILE, ["timestamp", "asset", "predicted_price", "volatility", "risk"])
df_actual = load_csv_safe(ACTUAL_FILE, ["timestamp", "gold_actual", "bitcoin_actual"])

# ------------------------------
# LOAD WEIGHTS / ASSUMPTIONS
# ------------------------------
if os.path.exists(WEIGHT_FILE):
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
else:
    weights = {"gold": {}, "bitcoin": {}}

# ------------------------------
# EMOJI MAPPING
# ------------------------------
INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

# ------------------------------
# MERGE PREDICTIONS WITH ACTUALS
# ------------------------------
def merge_actual_pred(asset_name, actual_col):
    asset_pred = df_pred[df_pred["asset"] == asset_name].copy()
    if asset_pred.empty:
        return asset_pred

    if actual_col in df_actual.columns:
        asset_pred = pd.merge_asof(
            asset_pred.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        asset_pred["actual"] = None

    asset_pred["predicted_price"] = pd.to_numeric(asset_pred["predicted_price"], errors='coerce')
    asset_pred["actual"] = pd.to_numeric(asset_pred["actual"], errors='coerce')

    asset_pred["signal"] = asset_pred.apply(
        lambda row: "Buy" if row["predicted_price"] > row["actual"] else ("Sell" if row["predicted_price"] < row["actual"] else "Hold"),
        axis=1
    )

    asset_pred["trend"] = ""
    if len(asset_pred) >= 3:
        last3 = asset_pred["predicted_price"].tail(3)
        if last3.is_monotonic_increasing:
            asset_pred["trend"] = "Bullish 📈"
        elif last3.is_monotonic_decreasing:
            asset_pred["trend"] = "Bearish 📉"
        else:
            asset_pred["trend"] = "Neutral ⚖️"

    asset_pred["assumptions"] = str(weights.get(asset_name.lower(), {}))
    asset_pred["target_horizon"] = "Days"

    asset_pred["target_price"] = asset_pred.apply(
        lambda row: row["actual"] if row["signal"]=="Buy" else (row["predicted_price"] if row["signal"]=="Sell" else row["actual"]),
        axis=1
    )

    return asset_pred

gold_df = merge_actual_pred("Gold", "gold_actual")
btc_df = merge_actual_pred("Bitcoin", "bitcoin_actual")

# ------------------------------
# UTILS
# ------------------------------
def color_signal(val):
    if val == "Buy":
        color = "#1f77b4"
    elif val == "Sell":
        color = "#ff7f0e"
    else:
        color = "gray"
    return f'color: {color}; font-weight:bold; text-align:center'

def alert_badge(signal):
    if signal == "Buy":
        return f'<div style="background-color:#1f77b4;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">BUY</div>'
    elif signal == "Sell":
        return f'<div style="background-color:#ff7f0e;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">SELL</div>'
    else:
        return f'<div style="background-color:gray;color:white;padding:8px;font-size:20px;text-align:center;border-radius:5px">HOLD</div>'

def target_price_card(price, asset_name):
    st.markdown(f"""
        <div style='background-color:#ffd700;color:black;padding:12px;font-size:22px;text-align:center;border-radius:8px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price}
        </div>
        """, unsafe_allow_html=True)

def assumptions_card(asset_df, asset_name):
    if asset_df.empty:
        st.info(f"No assumptions available for {asset_name}")
        return
    assumptions_str = asset_df["assumptions"].iloc[-1]
    target_horizon = asset_df["target_horizon"].iloc[-1]
    try:
        assumptions = eval(assumptions_str) if assumptions_str else {}
    except:
        assumptions = {}

    if not assumptions:
        st.info(f"No assumptions available for {asset_name}")
        return

    indicators = list(assumptions.keys())
    values = [assumptions[k] for k in indicators]
    icons = [INDICATOR_ICONS.get(k,"❔") for k in indicators]
    colors = ["#1f77b4" if v>0 else "#ff7f0e" if v<0 else "gray" for v in values]

    fig = go.Figure()
    for ind, val, icon, color in zip(indicators, values, icons, colors):
        fig.add_trace(go.Bar(
            x=[f"{icon} {ind}"],
            y=[val],
            marker_color=color,
            text=[f"{val:.2f}"],
            textposition='auto'
        ))
    fig.update_layout(title=f"{asset_name} Assumptions & Target ({target_horizon})",
                      yaxis_title="Weight / Impact")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# WHAT-IF SLIDERS
# ------------------------------
st.sidebar.header("🔧 What-If Scenario")

inflation_adj = st.sidebar.slider("Inflation 💹 (%)", 0.0, 10.0, 2.5, 0.1)
usd_adj = st.sidebar.slider("USD Strength 💵 (%)", -10.0, 10.0, 0.0, 0.1)
oil_adj = st.sidebar.slider("Oil Price 🛢️ (%)", -50.0, 50.0, 0.0, 0.1)
vix_adj = st.sidebar.slider("VIX / Volatility 🚨", 0.0, 100.0, 20.0, 1.0)

def apply_what_if(df):
    if df.empty:
        return df
    adj = 1 + inflation_adj*0.01 - usd_adj*0.01 + oil_adj*0.01 - vix_adj*0.005
    df = df.copy()
    df["predicted_price"] = df["predicted_price"] * adj
    df["target_price"] = df["predicted_price"]
    return df

gold_df_adj = apply_what_if(gold_df)
btc_df_adj = apply_what_if(btc_df)

def generate_summary(asset_df, asset_name):
    if asset_df.empty:
        return f"No data for {asset_name}"
    last_row = asset_df.iloc[-1]
    summary = f"**{asset_name} Market Summary:** "
    summary += f"Signal: {last_row['signal']} | Trend: {last_row['trend']} | Target Price: {last_row['target_price']}"
    return summary

# ------------------------------
# JOB DASHBOARD CONFIG (Adzuna)
# ------------------------------
CATEGORIES = {
    "HR": ["HCM", "HR", "Workday", "SAP SuccessFactors", "HRIS", "RH", "SIRH"],
    "Project Management": ["Project manager", "Chef de projet", "Program Manager"],
    "Supply": ["WMS", "Manhattan Associate", "Supply chain project manager"]
}

LOCATIONS = {
    "France": "fr",
    "Dubai": "ae",
    "Luxembourg": "lu",
    "Switzerland": "ch",
    "Worldwide": "gb"  # fallback
}

ADZUNA_APP_ID = "2c269bb8"
ADZUNA_APP_KEY = "39be78e26991e138d40ce4313620aebb"

def fetch_jobs_adzuna(keyword, country_code, location, max_results=5, remote_only=False, company_filter=None):
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": max_results,
        "what": keyword,
        "where": location if location != "Worldwide" else "",
        "content-type": "application/json"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        jobs = []
        for job in data.get("results", []):
            job_location = job.get("location", {}).get("display_name", "")
            company = job.get("company", {}).get("display_name", "")
            if remote_only and "remote" not in job_location.lower():
                continue
            if company_filter and company_filter.lower() not in company.lower():
                continue
            jobs.append({
                "title": job.get("title"),
                "company": company,
                "location": job_location,
                "link": job.get("redirect_url")
            })
        return pd.DataFrame(jobs)
    except:
        return pd.DataFrame([])

def jobs_dashboard():
    st.title("💼 Jobs Dashboard (Trello Style)")

    # Sidebar filters
    st.sidebar.header("Job Filters")
    location_choice = st.sidebar.selectbox("🌐 Select Location", list(LOCATIONS.keys()))
    remote_only = st.sidebar.checkbox("🏠 Only remote jobs", value=False)
    company_filter = st.sidebar.text_input("🏢 Filter by company (optional)", "")

    st.markdown(f"### 🌍 Showing jobs in **{location_choice}**")

    # Trello-style columns for categories
    cols = st.columns(len(CATEGORIES))
    for col, (cat, kws) in zip(cols, CATEGORIES.items()):
        with col:
            st.markdown(
                f"<div style='background-color:#004080;color:white;padding:10px;border-radius:8px;text-align:center;font-weight:bold'>📌 {cat}</div>",
                unsafe_allow_html=True
            )
            found_any = False
            for kw in kws:
                df_jobs = fetch_jobs_adzuna(
                    kw,
                    LOCATIONS[location_choice],
                    location_choice,
                    max_results=5,
                    remote_only=remote_only,
                    company_filter=company_filter if company_filter else None
                )
                if not df_jobs.empty:
                    found_any = True
                    for _, job in df_jobs.iterrows():
                        st.markdown(f"""
                            <div style='background-color:#f8f9fa;padding:10px;border-radius:8px;margin-top:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1)'>
                                <b><a href="{job['link']}" target="_blank" style="text-decoration:none;color:#004080">{job['title']}</a></b><br>
                                <span style='color:gray'>{job['company']} | {job['location']}</span>
                            </div>
                        """, unsafe_allow_html=True)
            if not found_any:
                st.info(f"No jobs found for {cat} in {location_choice}.")


# ------------------------------
# MAIN MENU
# ------------------------------
menu = st.sidebar.radio("📊 Choose Dashboard", ["Gold & Bitcoin", "Jobs"])

if menu == "Gold & Bitcoin":
    st.title("📊 Gold & Bitcoin Market Dashboard")
    col1, col2 = st.columns(2)
    for col, df, name in zip([col1, col2], [gold_df_adj, btc_df_adj], ["Gold", "Bitcoin"]):
        with col:
            st.subheader(name)
            if not df.empty:
                last_signal = df["signal"].iloc[-1]
                st.markdown(alert_badge(last_signal), unsafe_allow_html=True)
                last_trend = df["trend"].iloc[-1] if df["trend"].iloc[-1] else "Neutral ⚖️"
                st.markdown(f"**Market Trend:** {last_trend}")
                st.markdown(generate_summary(df, name))
                target_price_card(df["target_price"].iloc[-1], name)
                display_df = df[["timestamp","actual","predicted_price","volatility","risk","signal"]].tail(2)
                st.dataframe(display_df.style.applymap(color_signal, subset=["signal"]))
                assumptions_card(df, name)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual"], mode="lines+markers", name="Actual"))
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_price"], mode="lines+markers", name="Predicted"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {name} data available yet.")
elif menu == "Jobs":
    jobs_dashboard()
