# jobs_app.py
"""Curated job dashboard focused on France, Dubai and Switzerland."""
import html
import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "2c269bb8")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "39be78e26991e138d40ce4313620aebb")

COUNTRIES = {
    "France": {"code": "fr", "where": "France", "linkedin": "France", "indeed": "France"},
    "Dubaï": {"code": "ae", "where": "Dubai", "linkedin": "Dubai, United Arab Emirates", "indeed": "Dubai"},
    "Suisse": {"code": "ch", "where": "Switzerland", "linkedin": "Switzerland", "indeed": "Switzerland"},
}

ROLE_SEARCHES = {
    "HRIS / SAP SuccessFactors": "SAP SuccessFactors HRIS HCM SIRH",
    "SAP Integration / Architecture": "SAP CPI Integration Suite BTP HR Architect",
    "HR Transformation / Project Lead": "HR Transformation HRIS Project Manager Integration Lead",
    "Supply Chain / WMS (secondary)": "WMS Manhattan Supply Chain Project Manager",
}
DAYS_BACK = 30
MAX_PER_COUNTRY = 12
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CuratedJobsDashboard/1.0)"}


def _normalise_date(value):
    if not value:
        return pd.NaT
    try:
        return pd.to_datetime(value, utc=True).tz_convert(None)
    except Exception:
        return pd.NaT


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_adzuna(country_code, location, keyword):
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": 30,
        "what": keyword,
        "where": location,
        "sort_by": "date",
        "content-type": "application/json",
    }
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=(3, 8))
        response.raise_for_status()
        results = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return []

    jobs = []
    cutoff = datetime.utcnow() - timedelta(days=DAYS_BACK)
    for item in results:
        created = _normalise_date(item.get("created"))
        if pd.isna(created) or created.to_pydatetime() < cutoff:
            continue
        jobs.append({
            "title": item.get("title", ""),
            "company": (item.get("company") or {}).get("display_name", "Entreprise non précisée"),
            "location": (item.get("location") or {}).get("display_name", location),
            "date": created,
            "url": item.get("redirect_url", ""),
            "source": "Adzuna",
            "description": item.get("description", ""),
        })
    return jobs


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arbeitnow():
    try:
        response = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            headers=HEADERS,
            timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", []) if isinstance(payload, dict) else []
    except (requests.RequestException, ValueError, TypeError):
        return []


def _arbeitnow_country_match(location, country):
    text = (location or "").lower()
    aliases = {
        "France": ["france", "paris", "lyon", "marseille", "toulouse", "bordeaux", "lille", "nantes", "nice", "grenoble"],
        "Dubaï": ["dubai", "dubaï", "united arab emirates", "uae", "abu dhabi", "sharjah"],
        "Suisse": ["switzerland", "schweiz", "suisse", "zurich", "zürich", "geneva", "genève", "geneve", "lausanne", "basel", "bern", "lugano"],
    }
    return any(alias in text for alias in aliases[country])


def fetch_country_jobs(country):
    cfg = COUNTRIES[country]
    jobs = []
    for role, keyword in ROLE_SEARCHES.items():
        for job in fetch_adzuna(cfg["code"], cfg["where"], keyword):
            job["role"] = role
            jobs.append(job)

    for item in fetch_arbeitnow():
        location = item.get("location", "")
        if not _arbeitnow_country_match(location, country):
            continue
        created = _normalise_date(item.get("created_at"))
        if pd.isna(created) or created < pd.Timestamp(datetime.utcnow() - timedelta(days=DAYS_BACK)):
            continue
        title = item.get("title", "")
        description = item.get("description", "")
        text = f"{title} {description}".lower()
        if not any(term in text for term in [
            "successfactors", "hris", "hcm", "sirh", "sap", "integration",
            "hr architect", "hr transformation", "project manager", "wms", "manhattan"
        ]):
            continue
        jobs.append({
            "title": title,
            "company": item.get("company_name", "Entreprise non précisée"),
            "location": location or country,
            "date": created,
            "url": item.get("url", ""),
            "source": "Arbeitnow",
            "description": description,
            "role": "Offre complémentaire",
        })

    unique = {}
    for job in jobs:
        url = (job.get("url") or "").strip()
        key = url.lower() if url else (
            f"{job.get('title','').lower()}|{job.get('company','').lower()}|{job.get('location','').lower()}"
        )
        if not key.strip("|"):
            continue
        if key not in unique or job["date"] > unique[key]["date"]:
            unique[key] = job

    now = pd.Timestamp.utcnow().tz_localize(None)
    relevance_terms = {
        "successfactors": 8, "sap": 5, "hris": 8, "sirh": 8, "hcm": 6,
        "integration": 7, "cpi": 7, "btp": 7, "architect": 6,
        "transformation": 5, "project manager": 4, "manhattan": 4, "wms": 4,
    }
    ranked = []
    for job in unique.values():
        title = str(job.get("title", "")).lower()
        description = str(job.get("description", "")).lower()
        score = sum(weight * (2 if term in title else 1 if term in description else 0)
                    for term, weight in relevance_terms.items())
        age_days = max(0, (now - pd.Timestamp(job["date"])).total_seconds() / 86400)
        job["score"] = score + max(0, 30 - age_days) / 10
        ranked.append(job)

    ranked.sort(key=lambda x: (x["score"], x["date"]), reverse=True)
    return ranked[:MAX_PER_COUNTRY], []


def _external_search_links(country):
    cfg = COUNTRIES[country]
    query = quote_plus("SAP SuccessFactors HRIS SAP CPI integration HR architect")
    location = quote_plus(cfg["linkedin"])
    indeed_location = quote_plus(cfg["indeed"])
    indeed_domain = "fr.indeed.com" if country == "France" else "ae.indeed.com" if country == "Dubaï" else "ch.indeed.com"
    return [
        ("LinkedIn", f"https://www.linkedin.com/jobs/search/?keywords={query}&location={location}&f_TPR=r2592000"),
        ("Indeed", f"https://{indeed_domain}/jobs?q={query}&l={indeed_location}&fromage=30"),
        ("Welcome to the Jungle", f"https://www.welcometothejungle.com/fr/jobs?query={query}"),
    ]


def _render_job(job):
    title = html.escape(str(job.get("title") or "Intitulé non précisé"))
    company = html.escape(str(job.get("company") or "Entreprise non précisée"))
    location = html.escape(str(job.get("location") or "Localisation non précisée"))
    source = html.escape(str(job.get("source") or "Source"))
    role = html.escape(str(job.get("role") or ""))
    date = pd.Timestamp(job["date"]).strftime("%d/%m/%Y")
    url = job.get("url") or ""
    with st.container(border=True):
        if url:
            st.markdown(f"**[{title}]({url})**")
        else:
            st.markdown(f"**{title}**")
        st.caption(f"{company} · {location}")
        st.caption(f"Publié le {date} · {source} · {role}")


def jobs_dashboard():
    st.title("💼 Sélection d'offres d'emploi")
    st.caption("France · Dubaï · Suisse | 30 derniers jours | HRIS, SAP et intégration")

    st.info(
        "Les annonces automatiques proviennent d'Adzuna et d'Arbeitnow. "
        "LinkedIn est proposé en recherche directe : il ne fournit pas d'API publique ouverte "
        "permettant d'agréger librement ses annonces."
    )

    tabs = st.tabs(["🇫🇷 France", "🇦🇪 Dubaï", "🇨🇭 Suisse"])
    for tab, country in zip(tabs, COUNTRIES):
        with tab:
            with st.spinner(f"Recherche des offres récentes — {country}…"):
                jobs, _ = fetch_country_jobs(country)

            st.markdown(f"### {len(jobs)} offre(s) sélectionnée(s)")
            if jobs:
                for job in jobs:
                    _render_job(job)
            else:
                st.warning(
                    "Aucune offre correspondant aux critères n'a été trouvée dans les flux automatiques "
                    "sur les 30 derniers jours. Cela ne signifie pas qu'il n'y a pas d'offres disponibles."
                )

            st.markdown("#### Rechercher aussi sur les job boards")
            links = _external_search_links(country)
            cols = st.columns(len(links))
            for col, (label, url) in zip(cols, links):
                with col:
                    st.link_button(label, url, use_container_width=True)

            st.caption("Sources automatiques : Adzuna · Arbeitnow. Couverture variable selon le pays.")

    with st.expander("Postes et critères suivis"):
        for role, keywords in ROLE_SEARCHES.items():
            st.markdown(f"- **{role}** — {keywords}")
        st.caption("Tri par adéquation du titre/description et date de publication. Maximum 12 offres par pays, sans doublons.")
