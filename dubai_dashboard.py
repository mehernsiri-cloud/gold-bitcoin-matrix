# dubai_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.express as px
from datetime import datetime
from io import StringIO

st.set_page_config(page_title="Dubai Real Estate Dashboard", layout="wide")

USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}

# ---------------- Utility Functions ----------------
@st.cache_data(ttl=3600)
def safe_get_text(url, timeout=10):
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=timeout)
        r.raise_for_status()
        return r.text
    except:
        return ""

# ---------------- DXBInteract Scraper ----------------
@st.cache_data(ttl=3600)
def fetch_dxbinteract():
    """Scrape DXBInteract for live real estate data"""
    url = "https://dxbinteract.com/?ref=sitesmm"
    html = safe_get_text(url)
    if not html:
        return pd.DataFrame()

    try:
        soup = BeautifulSoup(html, "html.parser")
        rows = []

        # Try tables
        tables = soup.find_all("table")
        if tables:
            for table in tables:
                trs = table.find_all("tr")
                for tr in trs[1:]:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(tds) >= 3:
                        area = tds[0]
                        prop_type = tds[1] if len(tds) > 3 else "All"
                        avg_price = tds[2].replace(",", "").replace("AED", "")
                        tx = tds[3] if len(tds) > 3 else np.nan
                        rows.append([area, prop_type, float(avg_price), int(tx) if tx else np.nan, datetime.utcnow().date()])
            if rows:
                return pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])

        # Fallback: cards
        cards = soup.find_all("div", class_=lambda x: x and ("card" in x or "stat" in x.lower()))
        for card in cards:
            text = card.get_text(" ", strip=True)
            import re
            price_match = re.search(r"([A-Za-z\s]+)\s.*?([\d\.,]+)\s*(AED)?", text)
            tx_match = re.search(r"(\d+)\s+transaction", text, flags=re.I)
            if price_match:
                area = price_match.group(1).strip()
                prop_type = "All"
                price = float(price_match.group(2).replace(",", ""))
                tx = int(tx_match.group(1)) if tx_match else np.nan
                rows.append([area, prop_type, price, tx, datetime.utcnow().date()])

        if rows:
            return pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])
    except:
        pass

    # Failed scrape
    return pd.DataFrame()

# ---------------- Synthetic Fallback ----------------
@st.cache_data(ttl=3600)
def generate_synthetic_data():
    areas = ["Dubai Marina", "Downtown Dubai", "Jumeirah Village Circle", "Deira", "Business Bay"]
    types = ["Studio", "1BR", "2BR", "3BR"]
    rows = []
    today = datetime.utcnow().date()
    for a in areas:
        for t in types:
            avg_price = np.round(np.random.uniform(200000, 1200000), 2)
            tx = int(np.random.uniform(5, 200))
            rows.append([a, t, avg_price, tx, today])
    return pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])

# ---------------- Population Data ----------------
@st.cache_data(ttl=24*60*60)
def fetch_population():
    years = list(range(2010, datetime.utcnow().year+1))
    pop = [2000000 + (i-2010)*80000 for i in years]
    return pd.DataFrame({"Year": years, "Population": pop})

# ---------------- Data Merge ----------------
@st.cache_data(ttl=3600)
def build_market_data():
    df = fetch_dxbinteract()
    if df.empty:
        df = generate_synthetic_data()
    return df

# ---------------- Dashboard ----------------
def show_dashboard():
    st.title("🏙️ Dubai Real Estate Dashboard")
    st.write("Automatic, public-data only. No API keys required.")
    st.markdown("---")

    df_market = build_market_data()
    df_pop = fetch_population()

    # Sidebar filters
    min_price = int(df_market["avg_price"].min())
    max_price = int(df_market["avg_price"].max())
    price_range = st.sidebar.slider("Average price range", min_price, max_price, (min_price, max_price))
    area_list = ["All Areas"] + sorted(df_market["area"].dropna().unique().tolist())
    area_choice = st.sidebar.selectbox("Area", area_list)
    type_list = ["All Types"] + sorted(df_market["property_type"].dropna().unique().tolist())
    type_choice = st.sidebar.selectbox("Property Type", type_list)
    menu = st.sidebar.radio("View", ["Market Overview", "Prices", "Sales", "Market Health", "Population"])

    # Filter data
    dff = df_market.copy()
    if area_choice != "All Areas":
        dff = dff[dff["area"]==area_choice]
    if type_choice != "All Types":
        dff = dff[dff["property_type"]==type_choice]
    dff = dff[(dff["avg_price"]>=price_range[0]) & (dff["avg_price"]<=price_range[1])]

    # ---------------- Views ----------------
    if menu=="Market Overview":
        st.header("Market Overview")
        st.metric("Avg Price", f"{dff['avg_price'].mean():,.0f}")
        st.metric("Transactions", f"{dff['transactions'].sum():,.0f}")
        st.dataframe(dff)

    elif menu=="Prices":
        st.header("Property Prices")
        st.dataframe(dff)
        fig = px.bar(dff, x="area", y="avg_price", color="property_type", title="Avg Price by Area")
        st.plotly_chart(fig, use_container_width=True)

    elif menu=="Sales":
        st.header("Transactions by Area")
        agg = dff.groupby("area", as_index=False)["transactions"].sum()
        st.dataframe(agg)
        fig = px.bar(agg, x="area", y="transactions", title="Transactions by Area")
        st.plotly_chart(fig, use_container_width=True)

    elif menu=="Market Health":
        st.header("Market Health")
        # compute simple changes (MoM)
        if dff["date"].nunique() > 1:
            latest = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-1]).reset_index(drop=True)
            prev = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-2] if len(g)>=2 else None).dropna().reset_index(drop=True)
            if not prev.empty:
                merged = pd.merge(latest, prev, on="area", suffixes=("_latest","_prev"))
                merged["mom_change_pct"] = (merged["avg_price_latest"] - merged["avg_price_prev"])/merged["avg_price_prev"]*100
                merged["tx_change_pct"] = (merged["transactions_latest"] - merged["transactions_prev"])/merged["transactions_prev"].replace(0,np.nan)*100
                merged = merged.replace([np.inf,-np.inf],0).fillna(0)
                st.dataframe(merged[["area","avg_price_prev","avg_price_latest","mom_change_pct","transactions_prev","transactions_latest","tx_change_pct"]])
            else:
                st.info("Not enough historical data for MoM change.")
        else:
            st.info("No time series data. Snapshot only.")

    elif menu=="Population":
        st.header("Population & Demand Proxy")
        st.dataframe(df_pop)
        fig = px.line(df_pop, x="Year", y="Population", title="Dubai Population (Synthetic)")
        st.plotly_chart(fig, use_container_width=True)
        if "transactions" in df_market.columns and not df_market["transactions"].dropna().empty:
            total_tx = df_market["transactions"].sum()
            demand_proxy = df_pop.iloc[-1]["Population"] / max(1,total_tx)
            st.metric("Population per Transaction", f"{int(demand_proxy):,}")
        else:
            st.info("Transactions not available to compute demand proxy.")

    st.markdown("---")
    st.caption("Data source priority: DXBInteract → synthetic fallback. Fully public pages, no API keys.")

if __name__=="__main__":
    show_dashboard()
