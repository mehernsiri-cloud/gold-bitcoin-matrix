# dubai_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO

# -------- CONFIG --------
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0; +https://example.com/bot)"}

# Known public CSV endpoints to try as fallbacks (DLD / DSC style)
DLD_SAMPLES = [
    # These endpoints may change — functions handle failures gracefully
    "https://dubailand.gov.ae/en/open-data/real-estate-data/",  # landing page (not a CSV) - left as reference
    # Potential CSV exports (try multiple common patterns)
    "https://dubailand.gov.ae/portal/_layouts/15/download.aspx?SourceUrl=/OpenData/RealEstateSales.csv",
]

DSC_POSSIBLE = [
    # Example attempt at a DSC CSV — these will generally fail if unavailable; handled gracefully
    "https://www.dsc.gov.ae/Report/PopulationAndDemographics.csv",
]

# --------- UTIL / CACHING ----------
@st.cache_data(show_spinner=False)
def safe_get_text(url, timeout=10):
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        # do not raise — return empty
        return ""

@st.cache_data(show_spinner=False)
def safe_get_csv(url, timeout=15):
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=timeout)
        r.raise_for_status()
        text = r.text
        # Try to read CSV into pandas
        df = pd.read_csv(StringIO(text))
        return df
    except Exception:
        return None

# --------- DXBInteract SCRAPER ----------
@st.cache_data(ttl=60*60)
def fetch_dxb_interact():
    """
    Try to scrape DXBInteract public pages for price & transaction info.
    Returns a DataFrame with standardized columns:
      ['area', 'property_type', 'avg_price', 'transactions', 'date']
    If scraping fails or page structure changed, returns empty DataFrame.
    """
    base_url = "https://dxbinteract.com/?ref=sitesmm"
    html = safe_get_text(base_url)
    if not html:
        return pd.DataFrame()

    try:
        soup = BeautifulSoup(html, "html.parser")

        # DXBInteract presents multiple widgets — try to parse common table-like blocks.
        # We search for tables first; otherwise look for cards containing area & price info.
        tables = soup.find_all("table")
        if tables:
            # Attempt to parse first meaningful table
            rows = []
            for table in tables:
                ths = [th.get_text(strip=True) for th in table.find_all("th")]
                trs = table.find_all("tr")
                for tr in trs[1:]:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(tds) >= 3:
                        rows.append(tds)
                if rows:
                    break
            if rows:
                # Make conservative column mapping
                df = pd.DataFrame(rows)
                # Try to interpret columns flexibly
                if df.shape[1] >= 4:
                    df = df.iloc[:, :4]
                    df.columns = ["area", "property_type", "avg_price", "transactions"]
                elif df.shape[1] == 3:
                    df.columns = ["area", "avg_price", "transactions"]
                    df["property_type"] = "All"
                    df = df[["area", "property_type", "avg_price", "transactions"]]
                else:
                    # Unknown table shape
                    return pd.DataFrame()
                # Clean numeric columns
                df["avg_price"] = df["avg_price"].str.replace(r"[^\d\.\-]", "", regex=True).replace("", np.nan).astype(float, errors='ignore')
                df["transactions"] = df["transactions"].str.replace(r"[^\d\.\-]", "", regex=True).replace("", np.nan)
                # Try convert transactions to numeric
                df["transactions"] = pd.to_numeric(df["transactions"], errors='coerce')
                df["date"] = datetime.utcnow().date()
                # Standardize columns
                df = df[["area", "property_type", "avg_price", "transactions", "date"]]
                return df

        # If no table, try reading cards (common UX pattern)
        cards = soup.find_all("div", class_=lambda x: x and ("card" in x or "stat" in x.lower()))
        extracted = []
        for card in cards:
            text = card.get_text(separator=" ", strip=True)
            # Heuristic: "AreaName - AED 1,200,000 - 15 Transactions"
            parts = text.split()
            if len(parts) < 3:
                continue
            extracted.append(text)
        # Try to parse extracted strings into area/price/transactions
        rows = []
        for s in extracted:
            # Best-effort regex parse
            import re
            price_match = re.search(r"([A-Za-z\s]+)\s.*?([\d\.,]+)\s*(AED|USD|USD)?", s)
            tx_match = re.search(r"(\d+)\s+transaction", s, flags=re.I)
            if price_match:
                area = price_match.group(1).strip()
                price = price_match.group(2).replace(",", "")
                tx = int(tx_match.group(1)) if tx_match else np.nan
                rows.append([area, "All", float(price), tx, datetime.utcnow().date()])

        if rows:
            df = pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])
            return df

    except Exception:
        pass

    # Failed to parse DXBInteract
    return pd.DataFrame()

# --------- DUBAI LAND DEPARTMENT / PUBLIC CSV FALLBACK ----------
@st.cache_data(ttl=60*60)
def fetch_dld_open_data_try():
    """
    Try multiple known or guessed DLD CSV endpoints. If none work, returns None/empty df.
    """
    possible_csvs = [
        # Patterns tried — may or may not exist; safe_get_csv returns None if unreachable.
        "https://opendata.dubailand.gov.ae/datasets/real-estate-transactions.csv",
        "https://dubailand.gov.ae/portal/_layouts/15/download.aspx?SourceUrl=/OpenData/RealEstateSales.csv",
        # Official graph CSV endpoints (often used on fred style)
        # The function will gracefully handle failures.
    ]
    for url in possible_csvs:
        df = safe_get_csv(url)
        if df is not None and not df.empty:
            # Try to standardize a subset of columns
            df_cols = [c.lower() for c in df.columns]
            # Common columns might be: area, price, transaction_date, transactions, unit_type
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
            # Keep relevant columns if present
            keep = [c for c in ["area", "property_type", "avg_price", "transactions", "date"] if c in df.columns]
            df = df[keep].copy()
            # Basic cleaning
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.date
            if "avg_price" in df.columns:
                df["avg_price"] = pd.to_numeric(df["avg_price"], errors='coerce')
            if "transactions" in df.columns:
                df["transactions"] = pd.to_numeric(df["transactions"], errors='coerce')
            return df
    return pd.DataFrame()

# --------- DUBAI STATISTICS POPULATION ----------
@st.cache_data(ttl=24*60*60)
def fetch_dubai_population():
    """
    Try to fetch Dubai population time series from known public sources or return sample.
    """
    # Try DSC CSV variants
    for url in DSC_POSSIBLE:
        df = safe_get_csv(url)
        if df is not None and not df.empty:
            # try to find Year/Population columns
            cols = [c.lower() for c in df.columns]
            if "year" in cols and "population" in cols:
                df = df.rename(columns={df.columns[cols.index("year")]: "Year",
                                        df.columns[cols.index("population")]: "Population"})
                return df[["Year", "Population"]]
    # Fallback: small synthetic series (keeps UX working)
    years = list(range(2010, datetime.utcnow().year + 1))
    pop = [2000000 + (i - 2010) * 80000 for i in years]  # synthetic but plausible growth
    return pd.DataFrame({"Year": years, "Population": pop})

# --------- DATA MERGE / PREP ----------
def build_unified_market_df():
    """
    Build a unified DataFrame combining DXBInteract + DLD fallback.
    Columns: ['area','property_type','avg_price','transactions','date']
    """
    # Try DXBInteract first
    df1 = fetch_dxb_interact()
    if df1 is not None and not df1.empty:
        return df1

    # Fallback to DLD open CSV attempts
    df2 = fetch_dld_open_data_try()
    if df2 is not None and not df2.empty:
        return df2

    # Final fallback: generate a small synthetic dataset so UI remains functional
    areas = ["Dubai Marina", "Downtown Dubai", "Jumeirah Village Circle", "Deira", "Business Bay"]
    types = ["Studio", "1BR", "2BR", "3BR"]
    rows = []
    today = datetime.utcnow().date()
    for a in areas:
        for t in types:
            avg_price = np.round(np.random.uniform(200000, 1200000) if "Dubai Marina" not in a else np.random.uniform(300000, 2000000), 2)
            tx = int(np.random.uniform(5, 200))
            rows.append([a, t, avg_price, tx, today])
    df = pd.DataFrame(rows, columns=["area", "property_type", "avg_price", "transactions", "date"])
    return df

# --------- DASHBOARD UI ----------
def show_dubai_dashboard():
    st.title("🏙️ Dubai Real Estate Dashboard")
    st.write("Automatic, public-data only. No API keys required. Data refreshes on page load (cached for short periods).")
    st.markdown("---")

    df_market = build_unified_market_df()
    df_pop = fetch_dubai_population()

    # Sidebar filters
    st.sidebar.header("Dubai Dashboard Controls")
    min_price = int(float(df_market["avg_price"].min()) if not df_market["avg_price"].isnull().all() else 0)
    max_price = int(float(df_market["avg_price"].max()) if not df_market["avg_price"].isnull().all() else 2_000_000)
    price_range = st.sidebar.slider("Average price range", min_price, max_price, (min_price, max_price))

    area_list = sorted(df_market["area"].dropna().unique().tolist())
    area_list.insert(0, "All Areas")
    area_choice = st.sidebar.selectbox("Area", area_list)

    prop_types = sorted(df_market["property_type"].dropna().unique().tolist())
    prop_types.insert(0, "All Types")
    type_choice = st.sidebar.selectbox("Property Type", prop_types)

    # Menu
    menu = st.sidebar.radio("View", ["Market Overview", "Prices", "Sales", "Market Health", "Population"])
    st.sidebar.markdown("Data source priority: DXBInteract → DLD open data → local synthetic fallback")

    # Apply filters
    dff = df_market.copy()
    if area_choice != "All Areas":
        dff = dff[dff["area"] == area_choice]
    if type_choice != "All Types":
        dff = dff[dff["property_type"] == type_choice]
    dff = dff[(dff["avg_price"].fillna(0) >= price_range[0]) & (dff["avg_price"].fillna(max_price+1) <= price_range[1])]

    # MARKET OVERVIEW
    if menu == "Market Overview":
        st.header("Market Overview")
        st.write("Summary metrics across the selected filter.")
        col1, col2, col3, col4 = st.columns(4)
        avg_price = int(dff["avg_price"].mean()) if not dff["avg_price"].dropna().empty else None
        total_tx = int(dff["transactions"].sum()) if "transactions" in dff.columns and not dff["transactions"].dropna().empty else 0
        areas_covered = dff["area"].nunique()
        last_date = dff["date"].max() if "date" in dff.columns else None

        col1.metric("Avg Price", f"{avg_price:,}" if avg_price else "N/A")
        col2.metric("Transactions (sum)", f"{total_tx:,}")
        col3.metric("Areas", areas_covered)
        col4.metric("Last Update", str(last_date))

        st.markdown("#### Top areas by average price")
        if not dff.empty:
            top = dff.sort_values("avg_price", ascending=False).dropna(subset=["avg_price"]).head(10)
            st.dataframe(top[["area", "property_type", "avg_price", "transactions", "date"]])
            fig = px.bar(top, x="area", y="avg_price", color="property_type", title="Top areas by Avg Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No market data available to show.")

    # PRICES
    elif menu == "Prices":
        st.header("Property Prices (by Area & Type)")
        if dff.empty:
            st.info("No price data available for selected filters.")
        else:
            # Aggregate by area/type
            agg = dff.groupby(["area", "property_type"], as_index=False).agg({
                "avg_price": "mean",
                "transactions": "sum"
            }).sort_values("avg_price", ascending=False)
            st.dataframe(agg.head(200))

            st.markdown("### Prices over time (per area)")
            # If date exists, show time series; otherwise show current snapshot
            if "date" in dff.columns and dff["date"].nunique() > 1:
                ts = dff.groupby(["date","area"], as_index=False).avg_price.mean()
                fig = px.line(ts, x="date", y="avg_price", color="area", title="Avg Price over time by area")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(agg, x="area", y="avg_price", color="property_type", title="Average price snapshot")
                st.plotly_chart(fig, use_container_width=True)

    # SALES
    elif menu == "Sales":
        st.header("Sales Volume & Transactions")
        if dff.empty or "transactions" not in dff.columns:
            st.info("No transaction data available.")
        else:
            sales_agg = dff.groupby("area", as_index=False).transactions.sum().sort_values("transactions", ascending=False)
            st.dataframe(sales_agg.head(200))
            fig = px.bar(sales_agg.head(20), x="area", y="transactions", title="Transactions by Area (total)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Transaction trend (if time series available)")
            if "date" in dff.columns and dff["date"].nunique() > 1:
                tx_ts = dff.groupby("date", as_index=False).transactions.sum()
                fig2 = px.line(tx_ts, x="date", y="transactions", title="Transactions over time (all areas)")
                st.plotly_chart(fig2, use_container_width=True)

    # MARKET HEALTH
    elif menu == "Market Health":
        st.header("Market Health Indicators")
        if dff.empty:
            st.info("No data to compute market health.")
        else:
            # Price change proxies - require at least two dates per area
            if "date" in dff.columns and dff["date"].nunique() > 1:
                # compute simple MoM (last two dates) and YoY if data present
                latest = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-1]).reset_index(drop=True)
                prev = dff.groupby("area").apply(lambda g: g.sort_values("date").iloc[-2] if len(g.sort_values("date"))>=2 else None)
                prev = prev.dropna().reset_index(drop=True)
                # merge by area where possible
                if not prev.empty:
                    merged = pd.merge(latest, prev, on="area", suffixes=("_latest", "_prev"))
                    merged["mom_change_pct"] = (merged["avg_price_latest"] - merged["avg_price_prev"]) / merged["avg_price_prev"] * 100
                    merged["tx_change_pct"] = (merged["transactions_latest"] - merged["transactions_prev"]) / merged["transactions_prev"].replace(0, np.nan) * 100
                    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0)
                    st.markdown("### Recent price/transaction changes by area")
                    st.dataframe(merged[["area", "avg_price_prev", "avg_price_latest", "mom_change_pct", "transactions_prev", "transactions_latest", "tx_change_pct"]].sort_values("mom_change_pct", ascending=False).head(50))
                    fig = px.bar(merged.sort_values("mom_change_pct", ascending=False).head(20), x="area", y="mom_change_pct", title="MoM Price Change (%) by Area")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough historical snapshots to compute MoM/YoY changes.")
            else:
                st.info("No time series in filtered data to compute market health changes. Snapshot-only view is available.")

            # Market health score — simple composite
            avg_price = dff["avg_price"].mean() if not dff["avg_price"].isnull().all() else 0
            total_tx = dff["transactions"].sum() if "transactions" in dff.columns else 0
            # Score normalisation: smaller is healthier here (toy example)
            score = max(0, min(100, 50 + (avg_price / (max_price if (max_price:=dff['avg_price'].max()) else 1)) * 10 - np.log1p(total_tx)))
            st.metric("Market Health Score (0 worst → 100 best)", f"{int(score)}")

    # POPULATION
    elif menu == "Population":
        st.header("Population & Demand Proxies")
        st.write("Population figures and derived demand proxy (population per transaction).")
        if df_pop is None or df_pop.empty:
            st.info("Population data not available.")
        else:
            st.dataframe(df_pop)
            fig = px.line(df_pop, x="Year", y="Population", title="Dubai Population (historical)")
            st.plotly_chart(fig, use_container_width=True)

            # Demand proxy: population / transactions (latest)
            if "transactions" in df_market.columns and not df_market["transactions"].dropna().empty:
                total_tx_latest = df_market.groupby("date", as_index=False).transactions.sum().sort_values("date").iloc[-1]["transactions"]
                latest_pop = df_pop.iloc[-1]["Population"]
                demand_proxy = latest_pop / max(1, total_tx_latest)
                st.metric("Population per transaction (latest)",
                          f"{int(demand_proxy):,}",
                          help="Higher means population is large compared to transaction volume — possible higher latent demand.")
            else:
                st.info("Transactions data not available to compute demand proxy.")

    st.markdown("---")
    st.caption("Data sources: DXBInteract (scraped public pages), Dubai Land Department open data (when available), Dubai Statistics Center (when available). The module uses public pages only and falls back to synthetic data if sources are unreachable.")
