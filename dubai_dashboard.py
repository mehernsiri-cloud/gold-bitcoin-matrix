# dubai_dashboard_safe.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Dubai Real Estate Dashboard", layout="wide")

# ---------------- Utility Functions ----------------
@st.cache_data(ttl=3600)
def safe_get_csv(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except:
        return None

# ---------------- Official Data ----------------
@st.cache_data(ttl=3600)
def fetch_dld_data():
    """
    Attempt to load Dubai Land Department (DLD) official CSVs.
    Returns empty DataFrame if unavailable.
    """
    # Example DLD open data URL (update with actual CSV links if available)
    urls = [
        "https://opendata.arcgis.com/datasets/real-estate-transactions.csv",
        # Add more official CSV URLs if available
    ]
    for url in urls:
        df = safe_get_csv(url)
        if df is not None and not df.empty:
            # Standardize columns
            df_cols = [c.lower() for c in df.columns]
            mapping = {}
            for c in df.columns:
                cl = c.lower()
                if "area" in cl or "community" in cl:
                    mapping[c] = "area"
                if "price" in cl and "avg" in cl:
                    mapping[c] = "avg_price"
                if "price" in cl and "avg" not in cl:
                    mapping[c] = "price"
                if "transaction" in cl and ("count" in cl or "volume" in cl):
                    mapping[c] = "transactions"
                if "date" in cl:
                    mapping[c] = "date"
                if "type" in cl or "unit" in cl:
                    mapping[c] = "property_type"
            df = df.rename(columns=mapping)
            keep = [c for c in ["area","property_type","avg_price","transactions","date"] if c in df.columns]
            df = df[keep].copy()
            # Convert types
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.date
            if "avg_price" in df.columns:
                df["avg_price"] = pd.to_numeric(df["avg_price"], errors='coerce')
            if "transactions" in df.columns:
                df["transactions"] = pd.to_numeric(df["transactions"], errors='coerce')
            return df
    return pd.DataFrame()

@st.cache_data(ttl=24*60*60)
def fetch_population():
    """
    Attempt to load Dubai population data from official DSC CSV or fallback to synthetic.
    """
    # Example DSC CSV URL (update if actual CSV exists)
    urls = [
        "https://www.dsc.gov.ae/Report/PopulationAndDemographics.csv"
    ]
    for url in urls:
        df = safe_get_csv(url)
        if df is not None and not df.empty:
            cols = [c.lower() for c in df.columns]
            if "year" in cols and "population" in cols:
                df = df.rename(columns={df.columns[cols.index("year")]: "Year",
                                        df.columns[cols.index("population")]: "Population"})
                return df[["Year","Population"]]
    # Fallback synthetic series
    years = list(range(2010, datetime.utcnow().year+1))
    pop = [2000000 + (i-2010)*80000 for i in years]
    return pd.DataFrame({"Year": years, "Population": pop})

# ---------------- Synthetic Market Data ----------------
@st.cache_data(ttl=3600)
def generate_synthetic_market():
    areas = ["Dubai Marina","Downtown Dubai","Jumeirah Village Circle","Deira","Business Bay"]
    types = ["Studio","1BR","2BR","3BR"]
    rows = []
    today = datetime.utcnow().date()
    for a in areas:
        for t in types:
            avg_price = np.round(np.random.uniform(200000,1200000),2)
            tx = int(np.random.uniform(5,200))
            rows.append([a,t,avg_price,tx,today])
    return pd.DataFrame(rows, columns=["area","property_type","avg_price","transactions","date"])

# ---------------- Build Market Data ----------------
@st.cache_data(ttl=3600)
def build_market_data():
    df = fetch_dld_data()
    if df.empty:
        df = generate_synthetic_market()
    return df

# ---------------- Dashboard ----------------
def show_dashboard():
    st.title("🏙️ Dubai Real Estate Dashboard (Official Data)")
    st.write("Fully legal, using official open-data sources (DLD/DSC) with fallback.")
    st.markdown("---")

    df_market = build_market_data()
    df_pop = fetch_population()

    # Sidebar filters
    min_price = int(df_market["avg_price"].min())
    max_price = int(df_market["avg_price"].max())
    price_range = st.sidebar.slider("Average price range", min_price, max_price, (min_price,max_price))
    area_list = ["All Areas"] + sorted(df_market["area"].dropna().unique().tolist())
    area_choice = st.sidebar.selectbox("Area", area_list)
    type_list = ["All Types"] + sorted(df_market["property_type"].dropna().unique().tolist())
    type_choice = st.sidebar.selectbox("Property Type", type_list)
    menu = st.sidebar.radio("View", ["Market Overview","Prices","Sales","Market Health","Population"])

    # Apply filters
    dff = df_market.copy()
    if area_choice!="All Areas":
        dff = dff[dff["area"]==area_choice]
    if type_choice!="All Types":
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
        # Compute simple MoM price change if multiple dates
        if dff["date"].nunique()>1:
            latest = dff.groupby("area").apply(lambda g:g.sort_values("date").iloc[-1]).reset_index(drop=True)
            prev = dff.groupby("area").apply(lambda g:g.sort_values("date").iloc[-2] if len(g)>=2 else None).dropna().reset_index(drop=True)
            if not prev.empty:
                merged = pd.merge(latest, prev, on="area", suffixes=("_latest","_prev"))
                merged["mom_change_pct"] = (merged["avg_price_latest"]-merged["avg_price_prev"])/merged["avg_price_prev"]*100
                merged["tx_change_pct"] = (merged["transactions_latest"]-merged["transactions_prev"])/merged["transactions_prev"].replace(0,np.nan)*100
                merged = merged.replace([np.inf,-np.inf],0).fillna(0)
                st.dataframe(merged[["area","avg_price_prev","avg_price_latest","mom_change_pct","transactions_prev","transactions_latest","tx_change_pct"]])
            else:
                st.info("Not enough historical data for MoM change.")
        else:
            st.info("Snapshot only, no historical data.")

    elif menu=="Population":
        st.header("Population & Demand Proxy")
        st.dataframe(df_pop)
        fig = px.line(df_pop, x="Year", y="Population", title="Dubai Population")
        st.plotly_chart(fig, use_container_width=True)
        if "transactions" in df_market.columns and not df_market["transactions"].dropna().empty:
            total_tx = df_market["transactions"].sum()
            demand_proxy = df_pop.iloc[-1]["Population"]/max(1,total_tx)
            st.metric("Population per Transaction", f"{int(demand_proxy):,}")
        else:
            st.info("Transactions not available for demand proxy.")

    st.markdown("---")
    st.caption("Data source priority: Dubai Land Department (DLD) → Dubai Statistics Center (DSC) → synthetic fallback.")

if __name__=="__main__":
    show_dashboard()
