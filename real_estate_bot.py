# real_estate_bot.py
"""
Dubai Real Estate Sales Bot (MVP) — Streamlit with GitHub REST API push + Dynamic ROI insights

Features:
- Form-based lead collection (name, phone, email, budget, property type, area, country, nationality, purpose, horizon, payment_type)
- Mandatory field validation
- Saves leads to CSV automatically (data/real_estate_leads.csv)
- Pushes CSV to GitHub via REST API on each submission using GH_PAT + GH_REPO (set as Streamlit secrets)
- Loads dynamic ROI data (data/roi_data.json) and shows property-type-specific ROI insights and a bar chart
Notes:
- On Streamlit Cloud add secrets:
    GH_PAT = "your_github_pat"
    GH_REPO = "username/repo"   (format: owner/repo)
    GH_BRANCH = "main"          (optional)
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
import matplotlib.pyplot as plt

# ------------------------------
# Config / paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LEADS_CSV = os.path.join(DATA_DIR, "real_estate_leads.csv")
MARKET_DATA_JSON = os.path.join(DATA_DIR, "market_data.json")
ROI_DATA_JSON = os.path.join(DATA_DIR, "roi_data.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# Load / create market_data.json (fallback static)
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
# Load / create ROI data (dynamic)
# ------------------------------
DEFAULT_ROI_SAMPLE = {
    "Dubai Marina": {
        "Studio": {"roi": 6.8, "avg_price": 650000},
        "1BR": {"roi": 6.4, "avg_price": 820000},
        "2BR": {"roi": 6.1, "avg_price": 1200000}
    },
    "Business Bay": {
        "Studio": {"roi": 6.1, "avg_price": 520000},
        "1BR": {"roi": 6.0, "avg_price": 760000},
        "2BR": {"roi": 5.8, "avg_price": 1000000}
    },
    "JVC": {
        "Studio": {"roi": 5.9, "avg_price": 420000},
        "1BR": {"roi": 6.0, "avg_price": 520000},
        "2BR": {"roi": 6.2, "avg_price": 700000}
    },
    "Downtown": {
        "1BR": {"roi": 7.1, "avg_price": 1500000},
        "2BR": {"roi": 6.9, "avg_price": 2200000},
        "Studio": {"roi": 6.6, "avg_price": 900000}
    }
}
if not os.path.exists(ROI_DATA_JSON):
    with open(ROI_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_ROI_SAMPLE, f, indent=2)
with open(ROI_DATA_JSON, "r", encoding="utf-8") as f:
    ROI_DATA = json.load(f)

# ------------------------------
# Form choices & lists
# ------------------------------
PROPERTY_TYPES = ["Studio", "1BR", "2BR", "3BR", "Apartment", "Villa", "Penthouse", "Townhouse"]
AREAS = sorted(list(set(list(MARKET_DATA.keys()) + list(ROI_DATA.keys()))))

PURPOSES = ["Investment", "Personal Use", "Holiday Home", "Mixed"]
HORIZONS = ["<1 year", "1-3 years", "3-5 years", "5+ years"]
PAYMENT_TYPES = ["Cash", "Payment Plan", "Mortgage"]

# small curated country list (expand if needed)
COUNTRIES = [
    "United Arab Emirates", "Saudi Arabia", "Qatar", "Kuwait", "Oman", "Bahrain",
    "France", "Germany", "United Kingdom", "United States", "Canada", "India",
    "Pakistan", "China", "Russia", "Italy", "Spain", "Australia", "South Africa",
    "Brazil", "Turkey", "Egypt", "Lebanon", "Jordan", "Philippines"
]
NATIONALITIES = COUNTRIES.copy()

# ------------------------------
# Ensure leads CSV exists with new headers
# ------------------------------
def init_leads_csv():
    if not os.path.exists(LEADS_CSV):
        df = pd.DataFrame(columns=[
            "name", "phone", "email", "budget", "preference", "area",
            "country", "nationality", "purpose", "horizon", "payment_type", "timestamp"
        ])
        df.to_csv(LEADS_CSV, index=False)
init_leads_csv()

# ------------------------------
# GitHub REST API helper (Streamlit secrets)
# ------------------------------
def push_csv_to_github_api(commit_message: str = None) -> bool:
    token = st.secrets.get("GH_PAT")
    repo = st.secrets.get("GH_REPO")
    branch = st.secrets.get("GH_BRANCH", "main")

    if not token or not repo:
        st.warning("⚠️ GH_PAT or GH_REPO not set — add them in Streamlit Cloud Secrets.")
        return False

    api_url = f"https://api.github.com/repos/{repo}/contents/data/real_estate_leads.csv"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    try:
        with open(LEADS_CSV, "rb") as f:
            content_bytes = f.read()
    except FileNotFoundError:
        st.error("Local leads CSV not found.")
        return False

    content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    if not commit_message:
        commit_message = f"Update leads CSV — {datetime.utcnow().isoformat()}"

    resp_get = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=15)
    sha = resp_get.json().get("sha") if resp_get.status_code == 200 else None

    payload = {"message": commit_message, "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha

    resp_put = requests.put(api_url, headers=headers, json=payload, timeout=30)
    if resp_put.status_code in (200, 201):
        st.info("✅ Leads CSV pushed to GitHub successfully.")
        return True
    else:
        st.error(f"GitHub API error ({resp_put.status_code}): {resp_put.text}")
        return False

# ------------------------------
# Save lead locally and push to GitHub
# ------------------------------
def save_lead_and_push(fields: dict) -> bool:
    lead = fields.copy()
    lead["timestamp"] = datetime.utcnow().isoformat()
    try:
        df = pd.read_csv(LEADS_CSV)
    except Exception:
        df = pd.DataFrame(columns=[
            "name", "phone", "email", "budget", "preference", "area",
            "country", "nationality", "purpose", "horizon", "payment_type", "timestamp"
        ])
    df = pd.concat([df, pd.DataFrame([lead])], ignore_index=True)
    df.to_csv(LEADS_CSV, index=False)
    return push_csv_to_github_api()

# ------------------------------
# ROI utilities
# ------------------------------
def get_roi_for(area: str, prop_type: str):
    """Return ROI and avg_price if available for area+property type."""
    area_data = ROI_DATA.get(area, {})
    typ = area_data.get(prop_type)
    if typ:
        return typ.get("roi"), typ.get("avg_price")
    # fallback to MARKET_DATA if present
    m = MARKET_DATA.get(area, {})
    return m.get("roi"), None

def recommend_by_roi(prop_type: str, top_n: int = 5):
    """Return list of (area, roi, avg_price) sorted by roi desc for given property type."""
    results = []
    for area, types in ROI_DATA.items():
        info = types.get(prop_type)
        if info and info.get("roi") is not None:
            results.append((area, info["roi"], info.get("avg_price")))
    # Also consider MARKET_DATA fallback (if roi exists)
    for area, m in MARKET_DATA.items():
        if area not in [r[0] for r in results] and m.get("roi") is not None:
            results.append((area, m["roi"], m.get("price_range")))
    results_sorted = sorted(results, key=lambda x: (x[1] if x[1] is not None else -1), reverse=True)
    return results_sorted[:top_n]

# ------------------------------
# Streamlit UI
# ------------------------------
def real_estate_dashboard():
    st.title("🏠 Dubai Real Estate Bot — Market Insights & Lead Capture")
    st.write("Please fill out all mandatory fields (*). The app shows ROI insights for the selected property type.")

    with st.form(key="lead_form"):
        name = st.text_input("Full Name *", placeholder="e.g., John Doe")
        phone = st.text_input("Phone (UAE format) *", placeholder="e.g., +971501234567")
        email = st.text_input("Email *", placeholder="e.g., example@gmail.com")
        budget = st.number_input("Budget in AED *", min_value=100_000, step=50_000, value=500_000)
        preference = st.selectbox("Property Type *", PROPERTY_TYPES)
        area = st.selectbox("Preferred Area *", AREAS)

        country = st.selectbox("Country of Residence *", COUNTRIES)
        nationality = st.selectbox("Nationality *", NATIONALITIES)
        purpose = st.radio("Purpose of Investment *", PURPOSES, horizontal=True)
        horizon = st.radio("Investment Horizon *", HORIZONS, horizontal=True)
        payment_type = st.radio("Preferred Payment Type *", PAYMENT_TYPES, horizontal=True)

        submit_btn = st.form_submit_button("Submit")

    if submit_btn:
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
            return

        lead_data = {
            "name": name.strip(),
            "phone": phone.strip(),
            "email": email.strip(),
            "budget": int(budget),
            "preference": preference,
            "area": area,
            "country": country,
            "nationality": nationality,
            "purpose": purpose,
            "horizon": horizon,
            "payment_type": payment_type
        }

        pushed = save_lead_and_push(lead_data)

        if pushed:
            st.success("✅ Lead saved locally and pushed to GitHub.")
        else:
            st.success("✅ Lead saved locally (push to GitHub failed or not configured).")

        # ------------------------------
        # Dynamic ROI insights for the selected property type
        # ------------------------------
        st.subheader(f"📊 ROI Insights for {preference}")
        roi_value, avg_price = get_roi_for(area, preference)
        if roi_value is not None:
            price_txt = f" — avg price: {avg_price} AED" if avg_price else ""
            st.markdown(f"**{preference} in {area}** → **{roi_value:.2f}%** ROI{price_txt}")
        else:
            st.info(f"No specific ROI data for {preference} in {area}. Showing top areas by ROI for this property type below.")

        # Top areas by ROI for the selected property type
        ranked = recommend_by_roi(preference, top_n=10)
        if ranked:
            df_rank = pd.DataFrame(ranked, columns=["area", "roi", "avg_price"])
            st.write("Top areas by estimated ROI:")
            st.dataframe(df_rank.head(10).style.format({"roi": "{:.2f}", "avg_price": lambda v: f'{v}' if isinstance(v, (int, float)) else v}))

            # Bar chart (matplotlib)
            try:
                areas_list = df_rank["area"].tolist()
                rois_list = [float(x) for x in df_rank["roi"].tolist()]
                fig, ax = plt.subplots(figsize=(6, max(3, len(areas_list)*0.4)))
                ax.barh(areas_list[::-1], rois_list[::-1])
                ax.set_xlabel("Estimated ROI (%)")
                ax.set_title(f"ROI by Area — {preference}")
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not render chart: {e}")
        else:
            st.info("No ROI dataset available for this property type yet.")

        # Recommendations based on budget (rules + ROI)
        st.subheader("💡 Suggested areas for your budget")
        budget_recs = []
        # simple mapping + include top ROI picks
        if int(budget) < 500_000:
            budget_recs = ["JVC", "Dubai South"]
        elif 500_000 <= int(budget) <= 1_000_000:
            budget_recs = ["Dubai Marina", "Business Bay"]
        else:
            budget_recs = ["Downtown", "Palm Jumeirah"]

        # enrich with ROI (if available)
        rec_lines = []
        for a in budget_recs:
            r, p = get_roi_for(a, preference)
            rec_lines.append(f"- {a}: ROI {r:.2f}% — price {p if p else MARKET_DATA.get(a, {}).get('price_range','N/A')}" if r else f"- {a}: ROI N/A — price {MARKET_DATA.get(a, {}).get('price_range','N/A')}")
        st.markdown("\n".join(rec_lines))

        # Offer CSV download
        try:
            with open(LEADS_CSV, "rb") as f:
                st.download_button(label="📥 Download Leads CSV", data=f, file_name="real_estate_leads.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Could not provide download: {e}")

# ------------------------------
if __name__ == "__main__":
    real_estate_dashboard()
