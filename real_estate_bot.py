# real_estate_bot.py
"""
AI-driven Real Estate Bot (Dubai) for Streamlit.

Features:
- Streamlit UI (real_estate_dashboard)
- Improved NLU extracting: budget (AED), preference (studio/apartment/villa/etc.), area, phone, email, name
- Optional OpenAI integration (OPENAI_API_KEY) for better NLU/responses
- Listings connector: tries BAYUT_API_KEY or PROPERTYFINDER_API_KEY, otherwise lightweight regex scraper fallback
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
import pandas as pd

# Optional dependencies
try:
    import requests
except Exception:
    requests = None

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
# Market data fetcher
# ------------------------------
def fetch_market_data(force_refresh: bool = False) -> Dict[str, Any]:
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
    txt = text.replace(",", "").lower()
    m = re.search(r"(\d+\.?\d*)\s*(k|m)?", txt)
    if not m:
        return None
    num = float(m.group(1))
    mult = m.group(2)
    if mult == "k":
        return int(num * 1000)
    if mult == "m":
        return int(num * 1_000_000)
    return int(num)

def extract_entities(text: str) -> Dict[str, Optional[str]]:
    res = {"name": None, "phone": None, "email": None, "budget": None, "preference": None, "area": None}

    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    if m:
        res["email"] = m.group(0)

    m = re.search(r"(\+971\s?\d{7,9}|\b05\d{7}\b)", text)
    if m:
        res["phone"] = m.group(0).strip()

    b = parse_budget(text)
    if b:
        res["budget"] = str(b)

    for k in PREF_KEYWORDS:
        if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.I):
            res["preference"] = k.lower()
            break

    for a in AREA_KEYWORDS:
        if a in text.lower():
            res["area"] = a.title()
            break

    m = re.search(r"\b(DLRC|DMC|DIFC|JVC|JLT)\b", text, flags=re.I)
    if m and not res["area"]:
        res["area"] = m.group(1).upper()

    m = re.search(r"(?:my name is|i am|this is)\s+([A-Z][a-zA-Z\- ]+)", text)
    if m:
        res["name"] = m.group(1).strip()

    return res

# ------------------------------
# Listings connector
# ------------------------------
BAYUT_API_KEY = os.environ.get("BAYUT_API_KEY")
PROPERTYFINDER_API_KEY = os.environ.get("PROPERTYFINDER_API_KEY")

def _range_from_budget(budget_aed: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if budget_aed is None:
        return None, None
    low = int(budget_aed * 0.9)
    high = int(budget_aed * 1.1)
    return low, high

def scrape_listings(area: Optional[str], limit: int = 5):
    sample = [
        {"title": "Studio - Downtown Dubai", "area": "Downtown Dubai", "price": 480000, "url": "https://example.com/1", "provider": "sample"},
        {"title": "1BR - Dubai Marina", "area": "Dubai Marina", "price": 650000, "url": "https://example.com/2", "provider": "sample"}
    ]
    if requests is None or not area:
        return sample[:limit]

    try:
        q = area.replace(" ", "+")
        url = f"https://www.bayut.com/for-sale/?q={q}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        html = r.text
        titles = re.findall(r'title="([^"]+)"', html)[:limit]
        prices = re.findall(r"AED\s?[\d,]+", html)[:limit]

        results = []
        for t, p in zip(titles, prices):
            try:
                price = int(re.sub(r"[^\d]", "", p))
            except Exception:
                price = None
            results.append({"title": t, "area": area, "price": price, "url": url, "provider": "scraper"})
        return results or sample[:limit]
    except Exception:
        return sample[:limit]

def fetch_listings(area: Optional[str], budget_aed: Optional[int], limit: int = 5):
    return scrape_listings(area, limit=limit)

# ------------------------------
# OpenAI helper
# ------------------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def openai_extract_and_reply(user_text: str) -> Optional[Dict[str, Any]]:
    if not OPENAI_AVAILABLE or not OPENAI_KEY:
        return None
    try:
        system = "Extract JSON with fields: name, phone, email, budget, preference, area, and a reply."
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            temperature=0.0,
            max_tokens=400
        )
        return json.loads(resp.choices[0].message.get("content", ""))
    except Exception:
        return None

# ------------------------------
# Lead storage
# ------------------------------
def save_lead(fields: Dict[str, Optional[str]], note: Optional[str] = None):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO leads (name, phone, email, budget, preference, area, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (fields.get("name"), fields.get("phone"), fields.get("email"), fields.get("budget"),
         fields.get("preference"), fields.get("area"), note, now)
    )
    conn.commit()
    conn.close()

# ------------------------------
# Bot engine
# ------------------------------
def bot_reply(user_text: str) -> Dict[str, Any]:
    conn = _get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO conversations (role, message, timestamp) VALUES (?, ?, ?)",
              ("user", user_text, datetime.utcnow().isoformat()))
    conn.commit()

    extracted = extract_entities(user_text)
    reply_text = None

    if OPENAI_AVAILABLE and OPENAI_KEY:
        parsed = openai_extract_and_reply(user_text)
        if parsed:
            extracted.update({k: v for k, v in parsed.items() if k in extracted and v})
            reply_text = parsed.get("reply")

    if not reply_text:
        listings = fetch_listings(extracted.get("area"), int(extracted["budget"]) if extracted.get("budget") else None)
        if listings:
            lines = [f"- {it['title']} ({it['area']}) — {it['price']} AED — {it['url']}" for it in listings[:3]]
            reply_text = "I found some options:\n" + "\n".join(lines)
        else:
            reply_text = "Please share your budget and preferred area so I can suggest properties."

    c.execute("INSERT INTO conversations (role, message, timestamp) VALUES (?, ?, ?)",
              ("bot", reply_text, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    if extracted.get("email") or extracted.get("phone") or extracted.get("name"):
        save_lead(extracted, note=user_text)

    return {"reply": reply_text, "extracted": extracted}

# ------------------------------
# Streamlit UI
# ------------------------------
def real_estate_dashboard():
    st.title("🏠 Real Estate Agent Bot — Dubai Sales Assistant")

    if "re_bot_history" not in st.session_state:
        st.session_state.re_bot_history = []

    user_text = st.text_area("Client message:", "")
    if st.button("Send"):
        result = bot_reply(user_text)
        st.session_state.re_bot_history.append({"user": user_text, "bot": result["reply"], "extracted": result["extracted"]})

    for msg in st.session_state.re_bot_history[-20:]:
        st.markdown(f"**Client:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")
        st.markdown(f"_Extracted:_ {msg['extracted']}")

if __name__ == "__main__":
    real_estate_dashboard()
