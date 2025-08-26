# jobs_app.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ------------------------------
# JOB DASHBOARD CONFIG
# ------------------------------
CATEGORIES = {
    "HR": ["HCM", "HR", "Workday", "SAP SuccessFactors", "HRIS", "RH", "SIRH"],
    "Project Management": ["Project manager", "Chef de projet", "Program Manager"],
    "Supply": ["WMS", "Manhattan Associate", "Supply chain project manager"]
}

LOCATIONS = {
    "France": "fr",
    "Dubai": "ae",
    "Luxembourg": "lu",
    "Switzerland": "ch",
    "Worldwide": "gb"
}

ADZUNA_APP_ID = "2c269bb8"
ADZUNA_APP_KEY = "39be78e26991e138d40ce4313620aebb"

# ------------------------------
# UTILS
# ------------------------------
def clean_text(text):
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text()

def fetch_jobs_adzuna(keyword, country_code, location, max_results=5, remote_only=False, company_filter=None):
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": max_results,
        "what": keyword,
        "where": location if location != "Worldwide" else "",
        "content-type": "application/json"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        jobs = []
        for job in data.get("results", []):
            job_location = clean_text(job.get("location", {}).get("display_name", ""))
            company = clean_text(job.get("company", {}).get("display_name", ""))
            created = job.get("created", "")
            if remote_only and "remote" not in job_location.lower():
                continue
            if company_filter and company_filter.lower() not in company.lower():
                continue
            jobs.append({
                "title": clean_text(job.get("title")),
                "company": company,
                "location": job_location,
                "date": created.split("T")[0] if created else "",
                "link": job.get("redirect_url")
            })
        return pd.DataFrame(jobs)
    except:
        return pd.DataFrame([])

# ------------------------------
# JOB DASHBOARD FUNCTION
# ------------------------------
def jobs_dashboard():
    st.title("💼 Jobs Dashboard (Trello Style)")

    st.sidebar.header("Job Filters")
    location_choice = st.sidebar.selectbox("🌐 Select Location", list(LOCATIONS.keys()))
    remote_only = st.sidebar.checkbox("🏠 Only remote jobs", value=False)
    company_filter = st.sidebar.text_input("🏢 Filter by company (optional)", "")

    st.markdown(f"### 🌍 Showing jobs in **{location_choice}**")

    cols = st.columns(len(CATEGORIES))
    for col, (cat, kws) in zip(cols, CATEGORIES.items()):
        with col:
            st.markdown(
                f"<div style='background-color:#004080;color:white;padding:10px;border-radius:8px;text-align:center;font-weight:bold'>📌 {cat}</div>",
                unsafe_allow_html=True
            )
            found_any = False
            for kw in kws:
                df_jobs = fetch_jobs_adzuna(
                    kw,
                    LOCATIONS[location_choice],
                    location_choice,
                    max_results=5,
                    remote_only=remote_only,
                    company_filter=company_filter if company_filter else None
                )
                if not df_jobs.empty:
                    found_any = True
                    for _, job in df_jobs.iterrows():
                        st.markdown(f"""
                            <div style='background-color:#f8f9fa;padding:10px;border-radius:8px;margin-top:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1)'>
                                <b><a href="{job['link']}" target="_blank" style="text-decoration:none;color:#004080">{job['title']}</a></b><br>
                                <span style='color:gray'>{job['company']} | {job['location']}</span><br>
                                <span style='color:#888'>📅 {job['date']}</span>
                            </div>
                        """, unsafe_allow_html=True)
            if not found_any:
                st.info(f"No jobs found for {cat} in {location_choice}.")
