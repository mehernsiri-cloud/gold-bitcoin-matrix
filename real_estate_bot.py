# real_estate_bot.py
"""
Dubai Real Estate Sales Bot (MVP) — Streamlit version with guided flow.

Features:
- Step-by-step conversation: asks for missing info until all required fields are collected
- Extracts name, phone, email, budget, property type, preferred area
- Static market data for ROI & area recommendations
- Saves leads to local CSV (data/real_estate_leads.csv)
- Fully self-contained: no APIs or OpenAI required
"""

import os
import re
import json
from datetime import datetime

import streamlit as st
import pandas as pd

# ------------------------------
# Config / paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LEADS_CSV = os.path.join(DATA_DIR, "real_estate_leads.csv")
MARKET_DATA_JSON = os.path.join(DATA_DIR, "market_data.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# Static market data
# ------------------------------
DEFAULT_MARKET_DATA = {
    "JVC": {"roi": 6.0, "price_range": "400K-500K"},
    "Dubai South": {"roi": 5.8, "price_range": "350K-500K"},
    "Dubai Marina": {"roi": 6.5, "price_range": "500K-900K"},
    "Business Bay": {"roi": 6.3, "price_range": "500K-1M"},
    "Downtown": {"roi": 7.0, "price_range": "1M-3M"},
    "Palm Jumeirah": {"roi": 7.2, "price_range": "1M-5M"}
}

if not os.path.exists(MARKET_DATA_JSON):
    with open(MARKET_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_MARKET_DATA, f, indent=2)

with open(MARKET_DATA_JSON, "r", encoding="utf-8") as f:
    MARKET_DATA = json.load(f)

# ------------------------------
# Keywords for extraction
# ------------------------------
AREA_KEYWORDS = list(MARKET_DATA.keys())
PREF_KEYWORDS = ["studio", "apartment", "1br", "2br", "3br", "villa", "penthouse", "townhouse"]

REQUIRED_FIELDS = ["name", "phone", "email", "budget", "preference", "area"]

# ------------------------------
# Helper functions
# ------------------------------
def parse_budget(text: str):
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

def extract_entities(text: str, current_fields: dict):
    """Update existing fields with any info found in user text."""
    res = current_fields.copy()

    # Email
    if not res.get("email"):
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        if m: res["email"] = m.group(0)

    # Phone (UAE pattern)
    if not res.get("phone"):
        m = re.search(r"(\+971\s?\d{7,9}|\b05\d{7}\b)", text)
        if m: res["phone"] = m.group(0)

    # Budget
    if not res.get("budget"):
        b = parse_budget(text)
        if b: res["budget"] = str(b)

    # Property type
    if not res.get("preference"):
        for k in PREF_KEYWORDS:
            if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.I):
                res["preference"] = k.lower()
                break

    # Area
    if not res.get("area"):
        for a in AREA_KEYWORDS:
            if a.lower() in text.lower():
                res["area"] = a
                break

    # Name (simple pattern)
    if not res.get("name"):
        m = re.search(r"(?:my name is|i am|this is)\s+([A-Z][a-zA-Z\- ]+)", text, flags=re.I)
        if m: res["name"] = m.group(1).strip()

    return res

def save_lead(fields: dict):
    lead = fields.copy()
    lead["timestamp"] = datetime.utcnow().isoformat()
    if os.path.exists(LEADS_CSV):
        df = pd.read_csv(LEADS_CSV)
        df = pd.concat([df, pd.DataFrame([lead])], ignore_index=True)
    else:
        df = pd.DataFrame([lead])
    df.to_csv(LEADS_CSV, index=False)

def recommend_areas(budget_aed: int):
    if budget_aed < 500_000:
        return ["JVC", "Dubai South"]
    elif 500_000 <= budget_aed <= 1_000_000:
        return ["Dubai Marina", "Business Bay"]
    else:
        return ["Downtown", "Palm Jumeirah"]

def build_recommendation_text(budget: int, area: str = None):
    recommended_areas = recommend_areas(budget)
    if area and area in recommended_areas:
        recommended_areas = [area] + [a for a in recommended_areas if a != area]

    lines = []
    for a in recommended_areas:
        roi = MARKET_DATA.get(a, {}).get("roi", "N/A")
        price_range = MARKET_DATA.get(a, {}).get("price_range", "N/A")
        lines.append(f"- {a}: {price_range} AED — ROI: {roi}%")
    return "\n".join(lines)

# ------------------------------
# Streamlit UI
# ------------------------------
def real_estate_dashboard():
    st.title("🏠 Dubai Real Estate Bot — Guided MVP")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "fields" not in st.session_state:
        st.session_state.fields = {f: None for f in REQUIRED_FIELDS}

    user_text = st.text_area("Client message:", "")
    if st.button("Send"):
        # Update collected info
        st.session_state.fields = extract_entities(user_text, st.session_state.fields)

        # Determine missing fields
        missing_fields = [k for k, v in st.session_state.fields.items() if not v]

        if missing_fields:
            reply_text = "Hello! Could you please provide your " + ", ".join(missing_fields) + "?"
        else:
            # All info collected → give recommendations
            budget = int(st.session_state.fields["budget"])
            reply_text = "Thanks for sharing your info!\n\nHere are some suggested areas based on your budget:\n"
            reply_text += build_recommendation_text(budget, st.session_state.fields.get("area"))
            # Save lead
            save_lead(st.session_state.fields)
            # Reset fields for next client
            st.session_state.fields = {f: None for f in REQUIRED_FIELDS}

        # Append to history
        st.session_state.history.append({
            "user": user_text,
            "bot": reply_text,
            "extracted": st.session_state.fields
        })

    # Show last 20 messages
    for msg in st.session_state.history[-20:]:
        st.markdown(f"**Client:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")
        st.markdown(f"_Current collected info:_ {msg['extracted']}")

# ------------------------------
if __name__ == "__main__":
    real_estate_dashboard()
