# real_estate_bot.py
"""
Dubai Real Estate Sales Bot (MVP) — Streamlit with GitHub REST API push

Features:
- Form-based lead collection (name, phone, email, budget, property type, area)
- Mandatory field validation
- Saves leads to CSV automatically (data/real_estate_leads.csv)
- Pushes CSV to GitHub via REST API on each submission using GH_PAT + GH_REPO
- Provides recommended areas & ROI
Notes:
- On Streamlit Cloud set secrets: GH_PAT (token) and GH_REPO (owner/repo), optional GH_BRANCH
- Add `requests` to requirements.txt
"""

import os
import re
import json
import base64
import requests
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
# Static market data (basic)
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
# Ensure CSV exists with headers
# ------------------------------
def init_leads_csv():
    if not os.path.exists(LEADS_CSV):
        df = pd.DataFrame(columns=["name", "phone", "email", "budget", "preference", "area", "timestamp"])
        df.to_csv(LEADS_CSV, index=False)

init_leads_csv()

# ------------------------------
# GitHub REST API helper
# ------------------------------
def push_csv_to_github_api(commit_message: str = None) -> bool:
    """
    Push data/real_estate_leads.csv to GitHub via REST API.
    Requires Streamlit Cloud secrets:
      - GH_PAT  (token with repo contents write)
      - GH_REPO (owner/repo e.g. myuser/myrepo)
      - GH_BRANCH (optional, default 'main')
    Returns True on success, False on failure.
    """
    token = os.getenv("GH_PAT")
    repo = os.getenv("GH_REPO")
    branch = os.getenv("GH_BRANCH", "main")

    if not token or not repo:
        st.warning("GH_PAT or GH_REPO not set — cannot push leads to GitHub. Add them in Streamlit Cloud Secrets.")
        return False

    api_url = f"https://api.github.com/repos/{repo}/contents/data/real_estate_leads.csv"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    # Read and base64-encode file content
    try:
        with open(LEADS_CSV, "rb") as f:
            content_bytes = f.read()
    except FileNotFoundError:
        st.error("Local leads CSV not found.")
        return False

    content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    if not commit_message:
        commit_message = f"Update leads CSV — {datetime.utcnow().isoformat()}"

    # Check whether the file already exists on the repo to obtain its sha
    try:
        resp_get = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=15)
    except requests.RequestException as e:
        st.error(f"Network error while contacting GitHub: {e}")
        return False

    if resp_get.status_code == 200:
        sha = resp_get.json().get("sha")
    elif resp_get.status_code == 404:
        sha = None
    else:
        st.error(f"GitHub API GET error ({resp_get.status_code}): {resp_get.text}")
        return False

    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    try:
        resp_put = requests.put(api_url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        st.error(f"Network error while uploading to GitHub: {e}")
        return False

    if resp_put.status_code in (200, 201):
        # 201 = created, 200 = updated
        st.info("✅ Leads CSV pushed to GitHub successfully.")
        return True
    else:
        st.error(f"GitHub API PUT error ({resp_put.status_code}): {resp_put.text}")
        return False

# ------------------------------
# Lead handling
# ------------------------------
def save_lead_and_push(fields: dict) -> bool:
    """
    Append lead to local CSV, then attempt to push to GitHub.
    Returns True if push succeeded, False otherwise.
    """
    lead = fields.copy()
    lead["timestamp"] = datetime.utcnow().isoformat()

    # Append to CSV
    try:
        df = pd.read_csv(LEADS_CSV)
    except Exception:
        df = pd.DataFrame(columns=["name", "phone", "email", "budget", "preference", "area", "timestamp"])

    df = pd.concat([df, pd.DataFrame([lead])], ignore_index=True)
    df.to_csv(LEADS_CSV, index=False)

    # Push to GitHub (REST API)
    success = push_csv_to_github_api()
    return success

# ------------------------------
# Recommendation helpers
# ------------------------------
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
    st.title("🏠 Dubai Real Estate Bot — GitHub Push MVP")
    st.write("Please fill out all mandatory fields (*).")
    st.markdown(
        "⚠️ Make sure your Streamlit Cloud app secrets include `GH_PAT` (token) and `GH_REPO` (owner/repo). "
        "Optional: `GH_BRANCH` (default: main)."
    )

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
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
            errors.append("Invalid email format.")
        if not re.match(r"^(\+971\s?\d{7,9}|05\d{7})$", phone):
            errors.append("Invalid UAE phone number (use +971xxxxxxxx or 05xxxxxxx).")

        if errors:
            for e in errors:
                st.error(e)
        else:
            lead_data = {
                "name": name.strip(),
                "phone": phone.strip(),
                "email": email.strip(),
                "budget": int(budget),
                "preference": preference,
                "area": area
            }

            pushed = save_lead_and_push(lead_data)

            if pushed:
                st.success("✅ Lead saved locally and pushed to GitHub.")
            else:
                st.success("✅ Lead saved locally (push to GitHub failed or not configured).")

            st.subheader("Recommended Areas & ROI:")
            st.text(build_recommendation_text(int(budget), area))

            # Show local CSV path and offer download
            st.write(f"Local CSV path: `{LEADS_CSV}`")
            try:
                with open(LEADS_CSV, "rb") as f:
                    st.download_button(
                        label="📥 Download Leads CSV",
                        data=f,
                        file_name="real_estate_leads.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.warning(f"Could not offer download: {e}")

# ------------------------------
if __name__ == "__main__":
    real_estate_dashboard()
