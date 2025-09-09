# real_estate_bot.py
"""
Dubai Real Estate Sales Bot (MVP) — Streamlit version with guided form.

Features:
- Guided form: user fills fields with format hints
- Mandatory fields enforced: name, phone, email, budget, property type, area
- Static market data for ROI & area recommendations
- Saves leads to local CSV (data/real_estate_leads.csv) with automatic initialization
- Download CSV button available in Streamlit
- Fully self-contained: no APIs or OpenAI required
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import re
import json

# ------------------------------
# Config / paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LEADS_CSV = os.path.join(DATA_DIR, "real_estate_leads.csv")
MARKET_DATA_JSON = os.path.join(DATA_DIR, "market_data.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# Initialize CSV if not exists
# ------------------------------
if not os.path.exists(LEADS_CSV):
    pd.DataFrame(columns=["name","phone","email","budget","preference","area","timestamp"]).to_csv(LEADS_CSV, index=False)

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

PROPERTY_TYPES = ["Studio", "1BR", "2BR", "3BR", "Apartment", "Villa", "Penthouse", "Townhouse"]
AREAS = list(MARKET_DATA.keys())

# ------------------------------
# Helper functions
# ------------------------------
def save_lead(fields: dict):
    """Save lead to CSV with timestamp"""
    lead = fields.copy()
    lead["timestamp"] = datetime.utcnow().isoformat()
    df = pd.read_csv(LEADS_CSV)
    df = pd.concat([df, pd.DataFrame([lead])], ignore_index=True)
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
    st.title("🏠 Dubai Real Estate Bot — Form-Based MVP")
    st.write("Please fill out all mandatory fields (*).")

    with st.form(key="lead_form"):
        name = st.text_input("Full Name *", placeholder="e.g., John Doe")
        phone = st.text_input("Phone (UAE format) *", placeholder="e.g., +971501234567")
        email = st.text_input("Email *", placeholder="e.g., example@gmail.com")
        budget = st.number_input("Budget in AED *", min_value=100_000, step=50_000, value=500_000)
        preference = st.selectbox("Property Type *", PROPERTY_TYPES)
        area = st.selectbox("Preferred Area *", AREAS)

        submit_btn = st.form_submit_button("Submit")

    if submit_btn:
        # Validation
        errors = []
        if not name.strip():
            errors.append("Name is required.")
        if not re.match(r"[\w\.-]+@[\w\.-]+\.\w+$", email):
            errors.append("Invalid email format.")
        if not re.match(r"(\+971\s?\d{7,9}|05\d{7})", phone):
            errors.append("Invalid UAE phone number.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            lead_data = {
                "name": name.strip(),
                "phone": phone.strip(),
                "email": email.strip(),
                "budget": budget,
                "preference": preference,
                "area": area
            }
            save_lead(lead_data)
            st.success("✅ Lead saved successfully!")

            st.subheader("Recommended Areas & ROI:")
            st.text(build_recommendation_text(budget, area))

            # Offer CSV download
            df = pd.read_csv(LEADS_CSV)
            st.download_button(
                label="📥 Download all leads as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="real_estate_leads.csv",
                mime="text/csv"
            )

# ------------------------------
if __name__ == "__main__":
    real_estate_dashboard()
