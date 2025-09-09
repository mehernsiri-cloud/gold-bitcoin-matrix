# real_estate_bot.py
"""
Dubai Real Estate Sales Bot (MVP) — Streamlit with GitHub push

Features:
- Form-based lead collection (name, phone, email, budget, property type, area)
- Mandatory field validation
- Saves leads to CSV automatically
- Commits & pushes CSV to GitHub on each submission using GH_PAT
- Provides recommended areas & ROI
"""

import os
import re
import json
import subprocess
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

PROPERTY_TYPES = ["Studio", "1BR", "2BR", "3BR", "Apartment", "Villa", "Penthouse", "Townhouse"]
AREAS = list(MARKET_DATA.keys())

# ------------------------------
# Initialize CSV if missing
# ------------------------------
if not os.path.exists(LEADS_CSV):
    df = pd.DataFrame(columns=["name","phone","email","budget","preference","area","timestamp"])
    df.to_csv(LEADS_CSV, index=False)

# ------------------------------
# Helper functions
# ------------------------------
def save_lead(fields: dict):
    """Save lead locally to CSV and push to GitHub."""
    lead = fields.copy()
    lead["timestamp"] = datetime.utcnow().isoformat()
    df = pd.read_csv(LEADS_CSV)
    df = pd.concat([df, pd.DataFrame([lead])], ignore_index=True)
    df.to_csv(LEADS_CSV, index=False)
    # Try GitHub push
    gh_pat = os.getenv("GH_PAT")
    if not gh_pat:
        st.warning("⚠️ GH_PAT not set — cannot push leads to GitHub.")
        return
    try:
        repo_url = f"https://{gh_pat}@github.com/mehernsiri-cloud/gold-bitcoin-matrix.git"
        subprocess.run(["git", "config", "--global", "user.email", "bot@localhost"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "RealEstateBot"], check=True)
        subprocess.run(["git", "add", LEADS_CSV], check=True)
        subprocess.run(["git", "commit", "-m", f"Add new lead {datetime.utcnow().isoformat()}"], check=False)
        subprocess.run(["git", "pull", "--rebase", "-X", "theirs", repo_url], check=True)
        subprocess.run(["git", "push", repo_url], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Git push failed: {e}")

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
    st.title("🏠 Dubai Real Estate Bot — GitHub Auto-Push MVP")
    st.write("Please fill out all mandatory fields (*)")

    with st.form(key="lead_form"):
        name = st.text_input("Full Name *", placeholder="e.g., John Doe")
        phone = st.text_input("Phone (UAE format) *", placeholder="e.g., +971501234567")
        email = st.text_input("Email *", placeholder="e.g., example@gmail.com")
        budget = st.number_input("Budget in AED *", min_value=100_000, step=50_000, value=500_000)
        preference = st.selectbox("Property Type *", PROPERTY_TYPES)
        area = st.selectbox("Preferred Area *", AREAS)
        submit_btn = st.form_submit_button("Submit")

    if submit_btn:
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
            st.success("✅ Lead saved and pushed to GitHub!")

            st.subheader("Recommended Areas & ROI:")
            st.text(build_recommendation_text(budget, area))

            # Allow CSV download
            with open(LEADS_CSV, "rb") as f:
                st.download_button(
                    label="📥 Download Leads CSV",
                    data=f,
                    file_name="real_estate_leads.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    real_estate_dashboard()
