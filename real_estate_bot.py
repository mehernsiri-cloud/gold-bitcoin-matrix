"""
real_estate_bot.py

A self-contained, AI-ready Real Estate Agent Bot (Dubai) that plugs into your existing
Streamlit app as a separate module (like jobs_app.py). It is designed so you DON'T have to
change your `app.py`. Put this file next to your `app.py` and import `real_estate_dashboard`
from it.

Features:
- Streamlit UI for conversational chat + structured lead capture
- Optional OpenAI integration for smarter conversation & entity extraction (uses OPENAI_API_KEY env var)
- SQLite persistent storage (conversation history + leads) + CSV export
- Market data fetcher hooks (configure an API or a data file) and caching
- Simple rule-based fallback when OpenAI not available

Usage:
1. Save this file next to your app.py
2. In your app.py add: from real_estate_bot import real_estate_dashboard
   and add the menu item call just like jobs_app.

Note: This file purposely does NOT call any external APIs at creation time. It provides
clear spots (fetch_market_data) where you can plug your preferred data sources (paid APIs
or web-scrapers) and keeps everything local by default.
"""

import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List

import streamlit as st

# Optional: openai for smarter assistant. Only used if OPENAI_API_KEY is set in env.
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
    """
    Fetch market data for Dubai real estate.

    - source: optional URL or file path. If None, we try to load a cached file.
    - Returns a dictionary with keys you can adapt, e.g. {"average_yield": 6.5, "avg_price_per_sqm": 15000}

    NOTE: Implement your own data source here (property API, MLS, paid provider). This function
    includes a simple cache mechanism to avoid repeated calls.
    """
    # If user provided a local JSON file path
    if source and os.path.exists(source) and not force_refresh:
        try:
            with open(source, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Try cache
    if os.path.exists(MARKET_CACHE) and not force_refresh:
        try:
            with open(MARKET_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Default dummy data (safe fallback)
    dummy = {
        "timestamp": datetime.utcnow().isoformat(),
        "avg_yield_pct": 7.0,
        "avg_price_per_sqm": 14000,
        "hot_areas": [
            {"area": "Downtown Dubai", "avg_price_per_sqm": 20000, "typical_roi": 6.5},
            {"area": "Dubai Marina", "avg_price_per_sqm": 18000, "typical_roi": 6.8},
            {"area": "Dubai Hills", "avg_price_per_sqm": 12000, "typical_roi": 7.2}
        ]
    }

    # Save cache for later
    try:
        with open(MARKET_CACHE, "w", encoding="utf-8") as f:
            json.dump(dummy, f, indent=2)
    except Exception:
        pass

    return dummy

# ------------------------------
# Simple NLU: extract structured fields from free text
# ------------------------------
LEAD_PATTERNS = {
    "phone": r"(\+?\d[\d\s\-]{6,}\d)",
    "email": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
    "budget": r"(budget\s*[:=]?\s*\$?\s*([0-9,.kKmM]+))",
}


def extract_structured(text: str) -> Dict[str, Optional[str]]:
    """Try simple regex-based extraction first. Returns dict with keys: name, phone, email, budget, preference."""
    res = {"name": None, "phone": None, "email": None, "budget": None, "preference": None}

    # email
    m = re.search(LEAD_PATTERNS["email"], text)
    if m:
        res["email"] = m.group(1)

    # phone
    m = re.search(LEAD_PATTERNS["phone"], text)
    if m:
        res["phone"] = m.group(1)

    # budget
    m = re.search(LEAD_PATTERNS["budget"], text, flags=re.IGNORECASE)
    if m:
        res["budget"] = m.group(2)

    # preference: look for keywords
    pref_keys = ["apartment", "villa", "studio", "townhouse", "office", "retail"]
    for k in pref_keys:
        if k in text.lower():
            res["preference"] = k
            break

    # name: heuristic - look for 'my name is X' or 'i am X'
    m = re.search(r"(?:my name is|i am|this is)\s+([A-Z][a-zA-Z\- ]{1,40})", text)
    if m:
        res["name"] = m.group(1).strip()

    return res

# ------------------------------
# OpenAI integration helper (optional)
# ------------------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_KEY:
    openai.api_key = OPENAI_KEY

async def _call_openai_extract(prompt: str) -> Optional[Dict[str, Any]]:
    """Call OpenAI for entity extraction or response generation. Async signature to keep options open.
    NOTE: streamlit is synchronous — this function is kept as a wrapper; we will call it synchronously if needed.
    """
    if not OPENAI_AVAILABLE or not OPENAI_KEY:
        return None
    try:
        # Use a compact ChatCompletion call with JSON instructions for entity extraction
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if os.environ.get("OPENAI_MODEL") is None else os.environ.get("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "You are an assistant that extracts contact leads and property preferences from short user messages. Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )
        text = resp.choices[0].message.get("content", "")
        # Try to load JSON from response
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            # fallback: return raw text
            return {"raw": text}
    except Exception:
        return None

# ------------------------------
# Bot engine: generate reply and optionally extract structured lead
# ------------------------------

def bot_reply(user_text: str, use_openai: bool = True) -> Dict[str, Any]:
    """
    Returns a dict with keys: reply (text), extracted (dict)
    - When OpenAI configured, it will attempt to generate a nicer reply and extract entities.
    - Otherwise fallback to rule-based responses and regex extraction.
    """
    # Store the user message
    conn = _get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO conversations (role, message, timestamp) VALUES (?, ?, ?)", ("user", user_text, datetime.utcnow().isoformat()))
    conn.commit()

    # Try OpenAI extraction if available
    extracted = extract_structured(user_text)
    reply = None

    if OPENAI_AVAILABLE and OPENAI_KEY and use_openai:
        # Build a prompt asking for a friendly reply and a JSON extraction
        prompt = (
            "Message: \n" + user_text + "\n\n"
            "Return a JSON object with fields: name, phone, email, budget, preference, and a friendly reply as 'reply'. "
            "If a field is not present leave it null. Only output JSON."
        )
        try:
            parsed = _call_openai_extract(prompt)
            if parsed and isinstance(parsed, dict):
                # merge fields
                for k in ["name", "phone", "email", "budget", "preference"]:
                    if k in parsed and parsed[k]:
                        extracted[k] = parsed[k]
                reply = parsed.get("reply") or parsed.get("raw")
        except Exception:
            pass

    # If reply still not set, use rule-based
    if not reply:
        txt = user_text.lower()
        if "apartment" in txt:
            reply = "We have several apartments available in Downtown Dubai and Dubai Marina. What's your budget and preferred move-in date?"
        elif "villa" in txt:
            reply = "Villas in Palm Jumeirah and Dubai Hills are available. Can you share your budget and preferred number of bedrooms?"
        elif "roi" in txt or "investment" in txt:
            market = fetch_market_data()
            reply = f"Current estimated average yield in Dubai is {market.get('avg_yield_pct')}% (cached). Would you like a tailored ROI projection?"
        elif "meeting" in txt or "schedule" in txt:
            reply = "Sure — please provide your preferred date/time and contact details, and I will schedule a meeting with an agent."
        else:
            reply = "Thanks — could you tell me your budget, preferred area (e.g., Downtown Dubai), and contact phone or email?"

    # Save bot response
    c.execute("INSERT INTO conversations (role, message, timestamp) VALUES (?, ?, ?)", ("bot", reply, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    # Save lead if we have meaningful fields
    if extracted.get("email") or extracted.get("phone") or extracted.get("name"):
        save_lead(extracted, note=user_text)

    return {"reply": reply, "extracted": extracted}

# ------------------------------
# Lead storage / export
# ------------------------------

def save_lead(fields: Dict[str, Optional[str]], note: Optional[str] = None):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO leads (name, phone, email, budget, preference, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            fields.get("name"), fields.get("phone"), fields.get("email"), fields.get("budget"), fields.get("preference"), note, now
        )
    )
    conn.commit()
    conn.close()

    # Also append to CSV for quick export
    try:
        import csv
        write_header = not os.path.exists(LEADS_CSV)
        with open(LEADS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["name", "phone", "email", "budget", "preference", "message", "created_at"])
            w.writerow([fields.get("name"), fields.get("phone"), fields.get("email"), fields.get("budget"), fields.get("preference"), note, now])
    except Exception:
        pass

# ------------------------------
# Streamlit UI: real_estate_dashboard()
# ------------------------------

def real_estate_dashboard():
    st.title("🏠 Real Estate Agent Bot — Dubai Sales Assistant")
    st.markdown(
        """
        This assistant helps the sales team by: \n
        - Qualifying leads (extracts name, phone, email, budget, preference)\n
        - Storing conversations and leads to a local SQLite DB and CSV\n
        - Showing cached Dubai market indicators (configurable)\n
        - Optional OpenAI-based understanding if OPENAI_API_KEY is available\n
        """
    )

    # Market snapshot
    col_m1, col_m2, col_m3 = st.columns([2, 2, 2])
    market = fetch_market_data()
    col_m1.metric("Avg Yield (%)", market.get("avg_yield_pct", "N/A"))
    col_m2.metric("Avg Price / m²", market.get("avg_price_per_sqm", "N/A"))
    col_m3.text("Hot areas:")
    for a in market.get("hot_areas", [])[:3]:
        col_m3.write(f"- {a['area']}: {a.get('typical_roi')}% | {a.get('avg_price_per_sqm')}/m²")

    # Chat + structured capture
    if "re_bot_history" not in st.session_state:
        st.session_state.re_bot_history = []

    with st.form("re_bot_form", clear_on_submit=False):
        user_text = st.text_area("Client message / question", height=120)
        col1, col2 = st.columns([1, 1])
        submit = col1.form_submit_button("Send")
        extract_only = col2.form_submit_button("Extract & Save Lead")

    if submit or extract_only:
        if not user_text or user_text.strip() == "":
            st.warning("Please enter a message.")
        else:
            # run bot
            result = bot_reply(user_text)
            st.session_state.re_bot_history.append({"user": user_text, "bot": result["reply"], "extracted": result["extracted"]})

    # Display recent conversation
    st.markdown("### Conversation")
    for msg in st.session_state.re_bot_history[-10:]:
        st.markdown(f"**Client:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")
        if msg["extracted"]:
            st.markdown(f"_Extracted:_ {msg['extracted']}")

    # Admin tools: view leads & export
    st.markdown("---")
    st.markdown("### Admin / Leads")
    if st.button("Show saved leads (last 50)"):
        conn = _get_conn()
        df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC LIMIT 50", conn)
        conn.close()
        st.dataframe(df)

    if st.button("Export leads to CSV now"):
        if os.path.exists(LEADS_CSV):
            with open(LEADS_CSV, "rb") as f:
                st.download_button("Download leads CSV", f, file_name=os.path.basename(LEADS_CSV))
        else:
            st.info("No leads CSV available yet.")

    st.markdown("---")
    st.markdown("_Notes:_ This bot can be upgraded to call OpenAI for better NLU and responses. To enable, set OPENAI_API_KEY env var and optionally OPENAI_MODEL.")


# Allow running directly for quick debugging
if __name__ == "__main__":
    # For local debug only
    real_estate_dashboard()
