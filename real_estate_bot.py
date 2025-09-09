# real_estate_bot.py
"""
AI-driven Real Estate Bot (Dubai) for Streamlit.

Features:
- Streamlit UI (real_estate_dashboard)
- Improved NLU extracting: budget (AED), preference (studio/apartment/villa/etc.), area, phone, email, name
- Optional OpenAI integration (OPENAI_API_KEY) for better NLU/responses
- Listings connector: tries BAYUT_API_KEY or PROPERTYFINDER_API_KEY, otherwise lightweight scraper fallback
- SQLite storage for conversations and leads + CSV export
- Returns top matching live listings (when available) in replies
"""

import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st

# Optional dependencies
try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup  # for scraper fallback
except Exception:
    BeautifulSoup = None

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ------------------------------
# Config / paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_FILE = os.path.join(DATA_DIR, "real_estate_bot.db")
LEADS_CSV = os.path.join(DATA_DIR, "real_estate_leads.csv")
MARKET_CACHE = os.path.join(DATA_DIR, "market_cache.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# DB helpers
# ------------------------------
def _get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        email TEXT,
        budget TEXT,
        preference TEXT,
        area TEXT,
        message TEXT,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------
# Market data fetcher (placeholder)
# ------------------------------
def fetch_market_data(source: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """Load cached market snapshot or return a safe dummy. Replace hook to call a real market API."""
    if source and os.path.exists(source) and not force_refresh:
        try:
            with open(source, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    if os.path.exists(MARKET_CACHE) and not force_refresh:
        try:
            with open(MARKET_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    dummy = {
        "timestamp": datetime.utcnow().isoformat(),
        "avg_yield_pct": 6.8,
        "avg_price_per_sqm": 15000,
        "hot_areas": [
            {"area": "Downtown Dubai", "avg_price_per_sqm": 20000, "typical_roi": 6.5},
            {"area": "Dubai Marina", "avg_price_per_sqm": 18000, "typical_roi": 6.8},
            {"area": "Dubai Hills", "avg_price_per_sqm": 12000, "typical_roi": 7.2}
        ]
    }
    try:
        with open(MARKET_CACHE, "w", encoding="utf-8") as f:
            json.dump(dummy, f, indent=2)
    except Exception:
        pass
    return dummy

# ------------------------------
# NLU / entity extraction
# ------------------------------
AREA_KEYWORDS = [
    "downtown", "business bay", "dubai marina", "palm jumeirah", "jvc", "jlt",
    "dld", "dubai hills", "dlrc", "al barsha", "meer", "emirates hills"
]

PREF_KEYWORDS = ["studio", "apartment", "1br", "2br", "3br", "villa", "penthouse", "townhouse", "office"]

def parse_budget(text: str) -> Optional[int]:
    """
    Parse common budget expressions and return integer AED (e.g. '500k', '500,000', '500K AED', '0.5m').
    """
    txt = text.replace(",", "").lower()
    # look for patterns like 500k, 500k aed, 500000, 0.5m
    m = re.search(r"(\d+\.?\d*)\s*(k|m)?\s*(aed|dirham|dhs)?", txt)
    if not m:
        # sometimes '500k' without whitespace
        m2 = re.search(r"(\d+)\s*k\b", txt)
        if m2:
            return int(float(m2.group(1)) * 1000)
        return None
    num = float(m.group(1))
    mult = m.group(2)
    if mult == "k":
        return int(num * 1000)
    if mult == "m":
        return int(num * 1_000_000)
    return int(num)

def extract_entities(text: str) -> Dict[str, Optional[str]]:
    """
    Extract entities: name, phone, email, budget (int AED), preference, area.
    """
    res = {"name": None, "phone": None, "email": None, "budget": None, "preference": None, "area": None}

    # email
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    if m:
        res["email"] = m.group(0)

    # phone (UAE +971 or local 5/50/52 etc.)
    m = re.search(r"(\+971\s?\d{7,9}|\b05\d{7}\b|\b5\d{6,8}\b|\b[0-9]{7,9}\b)", text)
    if m:
        res["phone"] = m.group(0).strip()

    # budget
    b = parse_budget(text)
    if b:
        res["budget"] = str(b)

    # preference
    for k in PREF_KEYWORDS:
        if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.I):
            res["preference"] = k.lower()
            break

    # area detection (match keywords & common abbreviations)
    for a in AREA_KEYWORDS:
        if a in text.lower():
            res["area"] = a.title()
            break
    # attempt to catch short codes like DLRC
    m = re.search(r"\b(DLRC|DMC|DIFC|JVC|JLT)\b", text, flags=re.I)
    if m and not res["area"]:
        res["area"] = m.group(1).upper()

    # name heuristic
    m = re.search(r"(?:my name is|i am|this is)\s+([A-Z][a-zA-Z\- ]{1,40})", text)
    if m:
        res["name"] = m.group(1).strip()

    return res

# ------------------------------
# Listings connector (Bayut / Property Finder / Scraper)
# ------------------------------
BAYUT_API_KEY = os.environ.get("BAYUT_API_KEY")
PROPERTYFINDER_API_KEY = os.environ.get("PROPERTYFINDER_API_KEY")

def _range_from_budget(budget_aed: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """Given budget number, return a min/max search range (10% tolerance)."""
    if budget_aed is None:
        return None, None
    low = int(budget_aed * 0.9)
    high = int(budget_aed * 1.1)
    return low, high

def fetch_listings_from_bayut(area: Optional[str], min_price: Optional[int], max_price: Optional[int], limit: int = 5):
    """Example Bayut API call - update endpoint and params according to actual Bayut docs."""
    if not BAYUT_API_KEY or requests is None:
        return []
    try:
        endpoint = "https://api.bayut.com/v1/sales/search"  # hypothetical - replace with real endpoint
        headers = {"Authorization": f"Bearer {BAYUT_API_KEY}", "Accept": "application/json"}
        params = {"location": area, "min_price": min_price, "max_price": max_price, "limit": limit}
        r = requests.get(endpoint, headers=headers, params={k: v for k, v in params.items() if v is not None}, timeout=10)
        r.raise_for_status()
        data = r.json()
        # normalize results (depends on actual API)
        listings = []
        for it in data.get("results", [])[:limit]:
            listings.append({
                "title": it.get("title") or it.get("property_name"),
                "area": it.get("location") or area,
                "price": it.get("price"),
                "bedrooms": it.get("beds") or it.get("bedrooms"),
                "url": it.get("url"),
                "provider": "bayut"
            })
        return listings
    except Exception:
        return []

def fetch_listings_from_propertyfinder(area: Optional[str], min_price: Optional[int], max_price: Optional[int], limit: int = 5):
    """Example Property Finder API call - update endpoint and params per docs."""
    if not PROPERTYFINDER_API_KEY or requests is None:
        return []
    try:
        endpoint = "https://api.propertyfinder.ae/v1/listings/search"  # hypothetical
        headers = {"Authorization": f"Bearer {PROPERTYFINDER_API_KEY}", "Accept": "application/json"}
        params = {"location": area, "min_price": min_price, "max_price": max_price, "limit": limit}
        r = requests.get(endpoint, headers=headers, params={k: v for k, v in params.items() if v is not None}, timeout=10)
        r.raise_for_status()
        data = r.json()
        listings = []
        for it in data.get("data", [])[:limit]:
            listings.append({
                "title": it.get("title"),
                "area": it.get("location") or area,
                "price": it.get("price"),
                "bedrooms": it.get("bedrooms"),
                "url": it.get("url"),
                "provider": "propertyfinder"
            })
        return listings
    except Exception:
        return []

def scrape_listings(area: Optional[str], min_price: Optional[int], max_price: Optional[int], limit: int = 5):
    """
    Lightweight scraper fallback. Attempts to scrape a public listing site (e.g., bayut.com/search)
    This is a fragile fallback: if bs4 or requests not installed, returns safe sample results.
    """
    sample = [
        {"title": "Studio - Downtown Dubai", "area": "Downtown Dubai", "price": 480000, "bedrooms": 0, "url": "https://example.com/1", "provider": "sample"},
        {"title": "1BR - Dubai Marina", "area": "Dubai Marina", "price": 650000, "bedrooms": 1, "url": "https://example.com/2", "provider": "sample"}
    ]
    if requests is None or BeautifulSoup is None:
        return sample[:limit]
    try:
        # Very conservative scraper - adapt to an actual listing page if you have permission.
        q_area = area.replace(" ", "+") if area else ""
        url = f"https://www.bayut.com/for-sale/?q={q_area}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        listings = []
        cards = soup.select(".card-listing")[:limit] or soup.select(".ef447dde")[:limit]
        for c in cards:
            # best-effort extraction
            title = c.get_text()[:80]
            price_text = c.select_one(".price") or c.select_one(".property-price")
            price = None
            if price_text:
                p = re.sub(r"[^\d]", "", price_text.get_text())
                price = int(p) if p else None
            listings.append({"title": title.strip(), "area": area or "Unknown", "price": price or 0, "bedrooms": None, "url": url, "provider": "scrape"})
        if not listings:
            return sample[:limit]
        return listings
    except Exception:
        return sample[:limit]

def fetch_listings(area: Optional[str], budget_aed: Optional[int], limit: int = 5):
    """
    Top-level listings fetcher: tries Bayut -> PropertyFinder -> Scraper -> sample.
    Budget is used to compute min/max.
    """
    min_p, max_p = _range_from_budget(budget_aed) if budget_aed else (None, None)
    # 1) Bayut
    if BAYUT_API_KEY:
        res = fetch_listings_from_bayut(area, min_p, max_p, limit=limit)
        if res:
            return res
    # 2) PropertyFinder
    if PROPERTYFINDER_API_KEY:
        res = fetch_listings_from_propertyfinder(area, min_p, max_p, limit=limit)
        if res:
            return res
    # 3) Scraper fallback
    return scrape_listings(area, min_p, max_p, limit=limit)

# ------------------------------
# OpenAI helper (synchronous)
# ------------------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def openai_extract_and_reply(user_text: str) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper to call OpenAI for extraction+reply. Returns dict or None."""
    if not OPENAI_AVAILABLE or not OPENAI_KEY:
        return None
    try:
        system = (
            "You are a Dubai Real Estate assistant. Extract JSON fields: name, phone, email, budget, preference, area. "
            "Also return a friendly 'reply' string. Output only valid JSON."
        )
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text}
            ],
            temperature=0.0,
            max_tokens=400
        )
        text = resp.choices[0].message.get("content", "")
        parsed = json.loads(text)
        return parsed
    except Exception:
        return None

# ------------------------------
# Lead storage / export
# ------------------------------
def save_lead(fields: Dict[str, Optional[str]], note: Optional[str] = None):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO leads (name, phone, email, budget, preference, area, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fields.get("name"), fields.get("phone"), fields.get("email"), fields.get("budget"),
            fields.get("preference"), fields.get("area"), note, now
        )
    )
    conn.commit()
    conn.close()
    # append to csv
    try:
        import csv
        write_header = not os.path.exists(LEADS_CSV)
        with open(LEADS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["name", "phone", "email", "budget", "preference", "area", "message", "created_at"])
            w.writerow([fields.get("name"), fields.get("phone"), fields.get("email"), fields.get("budget"),
                        fields.get("preference"), fields.get("area"), note, now])
    except Exception:
        pass

# ------------------------------
# Bot engine: generate reply, extract, fetch listings
# ------------------------------
def bot_reply(user_text: str, use_openai: bool = True) -> Dict[str, Any]:
    """
    Process a user message:
     - store the message
     - extract entities (regex + optional OpenAI)
     - attempt to fetch listings matching budget/area/preference
     - save lead if contact info found
     - return reply + extracted + listings
    """
    conn = _get_conn()
    c = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    c.execute("INSERT INTO conversations (role, message, timestamp) VALUES (?, ?, ?)", ("user", user_text, timestamp))
    conn.commit()

    extracted = extract_entities(user_text)
    reply_text = None

    # Try OpenAI enhanced extraction+reply if available
    if use_openai and OPENAI_AVAILABLE and OPENAI_KEY:
        parsed = openai_extract_and_reply(user_text)
        if parsed and isinstance(parsed, dict):
            # merge fields if present
            for k in ("name", "phone", "email", "budget", "preference", "area"):
                if k in parsed and parsed[k]:
                    extracted[k] = parsed[k]
            reply_text = parsed.get("reply")

    # If budget extracted textually but not numeric, try parsing inside returned budget string
    if extracted.get("budget") and isinstance(extracted.get("budget"), str) and not extracted["budget"].isdigit():
        maybe = parse_budget(extracted["budget"])
        if maybe:
            extracted["budget"] = str(maybe)

    # If still no reply, craft a rule-based reply and include listings when possible
    if not reply_text:
        txt = user_text.lower()
        if "roi" in txt or "investment" in txt:
            market = fetch_market_data()
            reply_text = f"Estimated average yield in Dubai is ~{market.get('avg_yield_pct')}% (cached). I can prepare a tailored ROI projection for a selected property. Do you want that?"
        elif "meeting" in txt or "schedule" in txt:
            reply_text = "Sure — please provide your preferred date & time and contact details; I will schedule a meeting with an agent."
        else:
            # If we have budget/area/preference, try fetch listings and show top matches
            budget_num = int(extracted["budget"]) if extracted.get("budget") and str(extracted.get("budget")).isdigit() else None
            area = extracted.get("area")
            pref = extracted.get("preference")
            listings = []
            if budget_num or area or pref:
                listings = fetch_listings(area, budget_num, limit=5)
                if listings:
                    # Build a friendly reply with top 3 matches
                    lines = []
                    for it in listings[:3]:
                        price = it.get("price") or "N/A"
                        lines.append(f"- {it.get('title')} ({it.get('area')}) — {price} AED — {it.get('url') or 'link'}")
                    reply_text = "I found some matching properties:\n" + "\n".join(lines) + "\nWould you like me to schedule visits or get floorplans?"
                else:
                    reply_text = "I couldn't find live listings immediately — would you like me to ask an agent to source options and get back to you?"
            else:
                reply_text = "Thanks — could you tell me your budget (e.g., 500K AED), preferred area (e.g., Downtown Dubai), and contact phone or email so I can find matching properties?"

    # Save bot response
    c.execute("INSERT INTO conversations (role, message, timestamp) VALUES (?, ?, ?)", ("bot", reply_text, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    # Save lead if contact details present
    if extracted.get("email") or extracted.get("phone") or extracted.get("name"):
        save_lead(extracted, note=user_text)

    # Also fetch listings for return (if not already)
    budget_num = int(extracted["budget"]) if extracted.get("budget") and str(extracted.get("budget")).isdigit() else None
    listings_for_return = fetch_listings(extracted.get("area"), budget_num, limit=5) if (budget_num or extracted.get("area")) else []

    return {"reply": reply_text, "extracted": extracted, "listings": listings_for_return}

# ------------------------------
# Streamlit UI: real_estate_dashboard()
# ------------------------------
def real_estate_dashboard():
    st.title("🏠 Real Estate Agent Bot — Dubai Sales Assistant")
    st.markdown(
        """
        This assistant helps the sales team by:
        - Qualifying leads (extracts name, phone, email, budget, preference, area)
        - Storing conversations and leads to a local SQLite DB and CSV
        - Showing cached Dubai market indicators (configurable)
        - Fetching live listings from Bayut / PropertyFinder (if API keys provided) or using a scraper fallback
        - Optional OpenAI-based understanding if OPENAI_API_KEY is available
        """
    )

    # Market snapshot
    col_m1, col_m2, col_m3 = st.columns([2, 2, 2])
    market = fetch_market_data()
    col_m1.metric("Avg Yield (%)", market.get("avg_yield_pct", "N/A"))
    col_m2.metric("Avg Price / m²", market.get("avg_price_per_sqm", "N/A"))
    with col_m3:
        st.markdown("**Hot areas:**")
        for a in market.get("hot_areas", [])[:4]:
            st.write(f"- {a.get('area')}: ROI {a.get('typical_roi')}% | {a.get('avg_price_per_sqm')}/m²")

    # Chat + structured capture
    if "re_bot_history" not in st.session_state:
        st.session_state.re_bot_history = []

    st.markdown("### 💬 Client message / question")
    with st.form("re_bot_form", clear_on_submit=False):
        user_text = st.text_area("", height=120, key="re_user_text")
        col1, col2, col3 = st.columns([1, 1, 1])
        submit = col1.form_submit_button("Send")
        extract_only = col2.form_submit_button("Extract & Save Lead")
        clear = col3.form_submit_button("Clear Conversation")

    if clear:
        st.session_state.re_bot_history = []
    if submit or extract_only:
        if not user_text or user_text.strip() == "":
            st.warning("Please enter a message.")
        else:
            result = bot_reply(user_text)
            st.session_state.re_bot_history.append({"user": user_text, "bot": result["reply"], "extracted": result["extracted"], "listings": result["listings"]})

    # Show conversation
    st.markdown("### Conversation (latest)")
    for msg in st.session_state.re_bot_history[-20:]:
        st.markdown(f"**Client:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")
        if msg.get("extracted"):
            st.markdown(f"_Extracted:_ {msg['extracted']}")
        if msg.get("listings"):
            st.markdown("_Top listings:_")
            for it in msg["listings"][:3]:
                st.markdown(f"- **{it.get('title')}** — {it.get('area')} — {it.get('price')} AED — [{it.get('provider')}]({it.get('url')})")

    # Admin tools
    st.markdown("---")
    st.markdown("### Admin / Leads")
    if st.button("Show saved leads (last 50)"):
        conn = _get_conn()
        df = None
        try:
            df = st.experimental_data_editor(pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC LIMIT 50", conn), num_rows="dynamic")
        except Exception:
            # fallback to simple display
            df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC LIMIT 50", conn)
            st.dataframe(df)
        conn.close()

    if st.button("Export leads to CSV now"):
        if os.path.exists(LEADS_CSV):
            with open(LEADS_CSV, "rb") as f:
                st.download_button("Download leads CSV", f, file_name=os.path.basename(LEADS_CSV))
        else:
            st.info("No leads CSV available yet.")

    st.markdown("---")
    st.markdown("_Notes:_ To enable live Bayut/PropertyFinder integration set environment variables BAYUT_API_KEY or PROPERTYFINDER_API_KEY. To enable better NLU/responses set OPENAI_API_KEY._")

# Direct run
if __name__ == "__main__":
    real_estate_dashboard()
