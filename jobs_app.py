# jobs_app.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

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

REMOTEOK_URL = "https://remoteok.io/api"

# ------------------------------
# UTILS
# ------------------------------
def clean_text(text):
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text()

def fetch_jobs_adzuna(keyword, country_code, location, max_results=50, remote_only=False, company_filter=None, last_n_days=60):
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
        cutoff_date = datetime.today() - timedelta(days=last_n_days)
        for job in data.get("results", []):
            job_location = clean_text(job.get("location", {}).get("display_name", ""))
            company = clean_text(job.get("company", {}).get("display_name", ""))
            created = job.get("created", "")
            job_date = datetime.strptime(created.split("T")[0], "%Y-%m-%d") if created else None

            # Apply filters
            if remote_only and "remote" not in job_location.lower():
                continue
            if company_filter and company_filter.lower() not in company.lower():
                continue
            if job_date and job_date < cutoff_date:
                continue

            jobs.append({
                "title": clean_text(job.get("title")),
                "company": company,
                "location": job_location,
                "date": job_date,
                "link": job.get("redirect_url")
            })
        df_jobs = pd.DataFrame(jobs)
        if not df_jobs.empty:
            df_jobs = df_jobs.sort_values("date", ascending=False)
        return df_jobs
    except:
        return pd.DataFrame([])

def fetch_jobs_remoteok(keyword, last_n_days=60):
    try:
        resp = requests.get(REMOTEOK_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = resp.json()
        jobs = []
        cutoff_date = datetime.today() - timedelta(days=last_n_days)
        for job in data[1:]:  # first item is metadata
            job_date = datetime.strptime(job.get("date", "")[:10], "%Y-%m-%d") if job.get("date") else None
            if job_date and job_date < cutoff_date:
                continue
            description = clean_text(job.get("description", ""))
            if keyword.lower() in job.get("position", "").lower() or keyword.lower() in description.lower():
                jobs.append({
                    "title": job.get("position"),
                    "company": job.get("company"),
                    "location": job.get("location", "Remote"),
                    "date": job_date,
                    "link": job.get("url")
                })
        df_jobs = pd.DataFrame(jobs)
        if not df_jobs.empty:
            df_jobs = df_jobs.sort_values("date", ascending=False)
        return df_jobs
    except:
        return pd.DataFrame([])

# ------------------------------
# TRELLLO-STYLE JOB DASHBOARD
# ------------------------------
def jobs_dashboard():
    st.title("💼 Jobs Dashboard (Trello Style & RemoteOK)")

    # Sidebar filters
    st.sidebar.header("Job Filters")
    location_choice = st.sidebar.selectbox("🌐 Select Location", list(LOCATIONS.keys()))
    remote_only = st.sidebar.checkbox("🏠 Only remote jobs", value=False)
    company_filter = st.sidebar.text_input("🏢 Filter by company (optional)", "")
    last_n_days = st.sidebar.slider("📅 Show jobs opened in last (days)", 30, 90, 60, 10)

    selected_category = st.sidebar.selectbox("📌 Select Job Category", list(CATEGORIES.keys()))
    st.markdown(f"### 🌍 Showing jobs in **{location_choice}** for category **{selected_category}**")

    keywords = CATEGORIES[selected_category]

    # ------------------------------
    # ADZUNA JOBS
    # ------------------------------
    st.subheader("📌 Adzuna Jobs")
    cols = st.columns(len(keywords))
    for col, kw in zip(cols, keywords):
        with col:
            st.markdown(f"<div style='background-color:#004080;color:white;padding:10px;border-radius:8px;text-align:center;font-weight:bold'>📌 {kw}</div>", unsafe_allow_html=True)
            df_jobs = fetch_jobs_adzuna(
                kw,
                LOCATIONS[location_choice],
                location_choice,
                max_results=50,
                remote_only=remote_only,
                company_filter=company_filter if company_filter else None,
                last_n_days=last_n_days
            )
            if not df_jobs.empty:
                for _, job in df_jobs.iterrows():
                    job_date_str = job['date'].strftime("%Y-%m-%d") if job['date'] else "N/A"
                    st.markdown(f"""
                        <div style='background-color:#f8f9fa;padding:10px;border-radius:8px;margin-top:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1)'>
                            <b><a href="{job['link']}" target="_blank" style="text-decoration:none;color:#004080">{job['title']}</a></b><br>
                            <span style='color:gray'>{job['company']} | {job['location']}</span><br>
                            <span style='color:#888'>📅 {job_date_str}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No recent jobs found for {kw}.")

    # ------------------------------
    # REMOTEOK JOBS
    # ------------------------------
    st.subheader("🌐 RemoteOK Jobs")
    cols_remote = st.columns(len(keywords))
    for col, kw in zip(cols_remote, keywords):
        with col:
            st.markdown(f"<div style='background-color:#008080;color:white;padding:10px;border-radius:8px;text-align:center;font-weight:bold'>🌐 {kw}</div>", unsafe_allow_html=True)
            df_jobs_remote = fetch_jobs_remoteok(kw, last_n_days=last_n_days)
            if not df_jobs_remote.empty:
                for _, job in df_jobs_remote.iterrows():
                    job_date_str = job['date'].strftime("%Y-%m-%d") if job['date'] else "N/A"
                    st.markdown(f"""
                        <div style='background-color:#e8f4f4;padding:10px;border-radius:8px;margin-top:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1)'>
                            <b><a href="{job['link']}" target="_blank" style="text-decoration:none;color:#008080">{job['title']}</a></b><br>
                            <span style='color:gray'>{job['company']} | {job['location']}</span><br>
                            <span style='color:#888'>📅 {job_date_str}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No recent RemoteOK jobs found for {kw}.")
