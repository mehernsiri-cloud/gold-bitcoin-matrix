# dubai_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import StringIO
import requests

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Dubai Real Estate Dashboard", layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0; +https://example.com/bot)"}

# Official Dubai Open Data (CSV/Excel links)
# These are published monthly, updated on the portal
DUBAI_REAL_ESTATE_URL = "https://opendata.dubailand.gov.ae/datastore/odata3.0/bf1f11c3-96a1-42d0-80c5-6ff92e6d4f13?format=csv"

# ---------------- DATA FETCHERS ----------------
@st.cache_data(ttl=24*60*60, show_spinner=True)
def fetch_real_estate_data():
    """Fetch Dubai real estate transactions from open data portal."""
    try:
        r = requests.get(DUBAI_REAL_ESTATE_URL, headers=USER_AGENT, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    # Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Typical columns: "transaction_date", "community", "property_type", "transaction_value"
    mapping = {}
    for c in df.columns:
        if "community" in c or "area" in c:
            mapping[c] = "area"
        if "property" in c and "type" in c:
            mapping[c] = "property_type"
        if "transaction" in c and "value" in c:
            mapping[c] = "avg_price"
        if "transaction" in c and ("count" in c or "number" in c):
            mapping[c] = "transactions"
        if "date" in c:
            mapping[c] = "date"

    df = df.rename(columns=mapping)

    # Keep only relevant columns
    keep = [c for c in ["area", "property_type", "avg_price", "transactions", "date"] if c in df.columns]
    df = df[keep].copy()

    # Clean data
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "avg_price" in df.columns:
        df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    if "transactions" in df.columns:
        df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")

    return df


@st.cache_data(ttl=24*60*60)
def fetch_population_data():
    """Dubai population (synthetic fallback if not available)."""
    try:
        # Example Dubai Statistics Center dataset could go here
        # For now we simulate
        years = list(range(2010, datetime.utcnow().year + 1))
        pop = [2000000 + (i - 2010) * 80000 for i in years]
        return pd.DataFrame({"Year": years, "Population": pop})
    except Exception:
        return pd.DataFrame()


def build_market_data():
    df = fetch_real_estate_data()
    if not df.empty:
        return df

    # fallback synthetic dataset
    areas = ["Dubai Marina", "Downtown Dubai", "JVC", "Deira", "Business Bay"]
    types = ["Studio", "1BR", "2BR", "3BR"]
    rows = []
    today = datetime.utcnow().date()
    for a in areas:
        for t in types:
            avg_price = np.round(np.random.uniform(200000, 1200000), 2)
            tx = int(np.random.uniform(5, 200))
            rows.append([a, t, avg_price, tx, today])
    return pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])


# ---------------- DASHBOARD ----------------
def show_dashboard():
    st.title("🏙️ Dubai Real Estate Dashboard")
    st.write("Automatic dashboard using **Dubai Open Data Portal**. No API keys required.")
    st.markdown("---")

    df_market = build_market_data()
    df_pop = fetch_population_data()

    # Sidebar
    st.sidebar.header("Controls")
    min_price = int(df_market["avg_price"].min() or 0)
    max_price = int(df_market["avg_price"].max() or 2_000_000)
    price_range = st.sidebar.slider("Average price range", min_price, max_price, (min_price, max_price))

    area_list = ["All Areas"] + sorted(df_market["area"].dropna().unique().tolist())
    area_choice = st.sidebar.selectbox("Area", area_list)

    prop_list = ["All Types"] + sorted(df_market["property_type"].dropna().unique().tolist())
    type_choice = st.sidebar.selectbox("Property Type", prop_list)

    menu = st.sidebar.radio("View", ["Market Overview", "Prices", "Sales", "Market Health", "Population"])

    # Filters
    dff = df_market.copy()
    if area_choice != "All Areas":
        dff = dff[dff["area"] == area_choice]
    if type_choice != "All Types":
        dff = dff[dff["property_type"] == type_choice]
    dff = dff[(dff["avg_price"].fillna(0) >= price_range[0]) & (dff["avg_price"].fillna(max_price+1) <= price_range[1])]

    # Views
    if menu == "Market Overview":
        st.header("Market Overview")
        avg_price = dff["avg_price"].mean()
        total_tx = dff["transactions"].sum() if "transactions" in dff else 0
        last_date = dff["date"].max() if "date" in dff else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Price", f"{avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")
        col2.metric("Transactions", f"{int(total_tx):,}")
        col3.metric("Last Update", str(last_date))

        st.subheader("Top Areas")
        top = dff.sort_values("avg_price", ascending=False).head(10)
        st.dataframe(top)
        st.plotly_chart(px.bar(top, x="area", y="avg_price", color="property_type"), use_container_width=True)

    elif menu == "Prices":
        st.header("Prices by Area & Type")
        if dff.empty:
            st.info("No price data available.")
        else:
            agg = dff.groupby(["area", "property_type"], as_index=False).agg({"avg_price": "mean", "transactions": "sum"})
            st.dataframe(agg.head(50))
            st.plotly_chart(px.bar(agg, x="area", y="avg_price", color="property_type"), use_container_width=True)

    elif menu == "Sales":
        st.header("Transactions by Area")
        if dff.empty:
            st.info("No sales data available.")
        else:
            agg = dff.groupby("area", as_index=False).transactions.sum()
            st.plotly_chart(px.bar(agg, x="area", y="transactions"), use_container_width=True)

    elif menu == "Market Health":
        st.header("Market Health Indicators")
        if "date" in dff.columns and dff["date"].nunique() > 1:
            latest = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-1]).reset_index(drop=True)
            prev = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-2] if len(g) >= 2 else None).dropna()
            merged = pd.merge(latest, prev, on="area", suffixes=("_latest", "_prev"))
            merged["mom_change_pct"] = (merged["avg_price_latest"] - merged["avg_price_prev"]) / merged["avg_price_prev"] * 100
            st.dataframe(merged[["area", "avg_price_prev", "avg_price_latest", "mom_change_pct"]])
            st.plotly_chart(px.bar(merged, x="area", y="mom_change_pct"), use_container_width=True)
        else:
            st.info("Not enough time series data to compute health.")

    elif menu == "Population":
        st.header("Dubai Population")
        st.dataframe(df_pop)
        st.plotly_chart(px.line(df_pop, x="Year", y="Population"), use_container_width=True)

        if "transactions" in df_market.columns and not df_market.empty:
            total_tx = df_market.groupby("date", as_index=False).transactions.sum().iloc[-1]["transactions"]
            latest_pop = df_pop.iloc[-1]["Population"]
            demand_proxy = latest_pop / max(1, total_tx)
            st.metric("Population per transaction", f"{int(demand_proxy):,}")

    st.markdown("---")
    st.caption("Sources: Dubai Open Data Portal (DLD), synthetic fallback if data unavailable.")


if __name__ == "__main__":
    show_dashboard()
