# dubai_dashboard_synthetic.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Dubai Real Estate Dashboard", layout="wide")

# ---------------- Synthetic Data Generators ----------------
@st.cache_data(ttl=3600)
def generate_synthetic_market(days=30):
    """
    Generate synthetic Dubai real estate market data.
    days: number of daily snapshots to simulate
    """
    areas = ["Dubai Marina","Downtown Dubai","Jumeirah Village Circle","Deira","Business Bay"]
    types = ["Studio","1BR","2BR","3BR"]
    rows = []
    today = datetime.utcnow().date()
    
    for delta in range(days):
        date = today - timedelta(days=delta)
        for area in areas:
            for ptype in types:
                avg_price = np.round(np.random.uniform(200000, 1500000),2)
                transactions = int(np.random.uniform(5,200))
                rows.append([area, ptype, avg_price, transactions, date])
    
    df = pd.DataFrame(rows, columns=["area","property_type","avg_price","transactions","date"])
    return df

@st.cache_data(ttl=24*60*60)
def generate_synthetic_population():
    """
    Generate synthetic Dubai population time series
    """
    years = list(range(2010, datetime.utcnow().year+1))
    pop = [2000000 + (i-2010)*80000 for i in years]
    df = pd.DataFrame({"Year": years, "Population": pop})
    return df

# ---------------- Dashboard ----------------
def show_dashboard():
    st.title("🏙️ Dubai Real Estate Dashboard (Synthetic Data)")
    st.write("Fully legal dashboard using synthetic market & population data for demonstration.")
    st.markdown("---")
    
    df_market = generate_synthetic_market()
    df_pop = generate_synthetic_population()
    
    # ---------------- Sidebar ----------------
    st.sidebar.header("Filters & Controls")
    min_price = int(df_market["avg_price"].min())
    max_price = int(df_market["avg_price"].max())
    price_range = st.sidebar.slider("Average Price Range", min_price, max_price, (min_price,max_price))
    
    area_list = ["All Areas"] + sorted(df_market["area"].unique())
    area_choice = st.sidebar.selectbox("Area", area_list)
    
    type_list = ["All Types"] + sorted(df_market["property_type"].unique())
    type_choice = st.sidebar.selectbox("Property Type", type_list)
    
    menu = st.sidebar.radio("View", ["Market Overview","Prices","Sales","Market Health","Population"])
    
    # ---------------- Apply Filters ----------------
    dff = df_market.copy()
    if area_choice != "All Areas":
        dff = dff[dff["area"]==area_choice]
    if type_choice != "All Types":
        dff = dff[dff["property_type"]==type_choice]
    dff = dff[(dff["avg_price"]>=price_range[0]) & (dff["avg_price"]<=price_range[1])]
    
    # ---------------- Market Overview ----------------
    if menu=="Market Overview":
        st.header("Market Overview")
        avg_price = int(dff["avg_price"].mean())
        total_tx = int(dff["transactions"].sum())
        areas_covered = dff["area"].nunique()
        last_date = dff["date"].max()
        
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Avg Price", f"{avg_price:,}")
        col2.metric("Total Transactions", f"{total_tx:,}")
        col3.metric("Areas Covered", areas_covered)
        col4.metric("Last Snapshot", str(last_date))
        
        st.markdown("#### Top Areas by Average Price")
        top = dff.groupby("area", as_index=False)["avg_price"].mean().sort_values("avg_price", ascending=False)
        st.dataframe(top)
        fig = px.bar(top, x="area", y="avg_price", color="area", title="Top Areas by Avg Price")
        st.plotly_chart(fig, use_container_width=True)
    
    # ---------------- Prices ----------------
    elif menu=="Prices":
        st.header("Property Prices")
        agg = dff.groupby(["area","property_type"], as_index=False).agg({"avg_price":"mean","transactions":"sum"})
        st.dataframe(agg)
        fig = px.bar(agg, x="area", y="avg_price", color="property_type", title="Avg Price by Area & Type")
        st.plotly_chart(fig, use_container_width=True)
        
        # Prices over time
        st.markdown("### Price Trend Over Last 30 Days")
        ts = dff.groupby(["date","area"], as_index=False).avg_price.mean()
        fig2 = px.line(ts, x="date", y="avg_price", color="area", title="Price Trend by Area")
        st.plotly_chart(fig2, use_container_width=True)
    
    # ---------------- Sales ----------------
    elif menu=="Sales":
        st.header("Transactions / Sales Volume")
        sales_agg = dff.groupby("area", as_index=False)["transactions"].sum().sort_values("transactions", ascending=False)
        st.dataframe(sales_agg)
        fig = px.bar(sales_agg, x="area", y="transactions", title="Transactions by Area")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Transactions Trend")
        tx_ts = dff.groupby("date", as_index=False)["transactions"].sum()
        fig2 = px.line(tx_ts, x="date", y="transactions", title="Total Transactions Over Time")
        st.plotly_chart(fig2, use_container_width=True)
    
    # ---------------- Market Health ----------------
    elif menu=="Market Health":
        st.header("Market Health Indicators (Synthetic)")
        # Compute MoM price change
        latest = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-1]).reset_index(drop=True)
        prev = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-2] if len(g)>=2 else None).dropna().reset_index(drop=True)
        if not prev.empty:
            merged = pd.merge(latest, prev, on="area", suffixes=("_latest","_prev"))
            merged["mom_change_pct"] = (merged["avg_price_latest"]-merged["avg_price_prev"])/merged["avg_price_prev"]*100
            merged["tx_change_pct"] = (merged["transactions_latest"]-merged["transactions_prev"])/merged["transactions_prev"].replace(0,np.nan)*100
            merged = merged.fillna(0)
            st.dataframe(merged[["area","avg_price_prev","avg_price_latest","mom_change_pct","transactions_prev","transactions_latest","tx_change_pct"]])
            fig = px.bar(merged, x="area", y="mom_change_pct", title="MoM Price Change (%) by Area")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical snapshots for MoM calculations.")
    
    # ---------------- Population ----------------
    elif menu=="Population":
        st.header("Population & Demand Proxy")
        st.dataframe(df_pop)
        fig = px.line(df_pop, x="Year", y="Population", title="Dubai Population Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        total_tx_latest = df_market.groupby("date")["transactions"].sum().sort_values(ascending=False).iloc[0]
        demand_proxy = df_pop.iloc[-1]["Population"]/max(1,total_tx_latest)
        st.metric("Population per Transaction", f"{int(demand_proxy):,}")
    
    st.markdown("---")
    st.caption("All data is synthetic for demonstration purposes. No live government data is used. Fully legal and safe for Streamlit Cloud deployment.")

# ---------------- Run ----------------
if __name__=="__main__":
    show_dashboard()
