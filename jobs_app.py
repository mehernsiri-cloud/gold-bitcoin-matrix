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
    "France": {"code": None, "where": "France", "linkedin": "France", "indeed": "France"},
    "Dubaï": {"code": None, "where": "Dubai", "linkedin": "Dubai, United Arab Emirates", "indeed": "Dubai"},
    "Suisse": {"code": None, "where": "Switzerland", "linkedin": "Switzerland", "indeed": "Switzerland"},
}

ROLE_SEARCHES = {
    "HRIS / SIRH – IT": "Senior HRIS Manager HR IT Lead HRIS Project Manager SIRH",
    "SAP SuccessFactors": "Senior SAP SuccessFactors Architect Integration Lead Project Manager",
    "Architecte IT RH": "HRIS Solution Architect HR IT Architect HR Enterprise Architect",
    "Chef de projet IT / WMS": "Senior IT Project Manager WMS Manhattan Program Manager",
}
ROLE_QUERIES = [
    "Senior SAP SuccessFactors",
    "SAP SuccessFactors Architect",
    "SAP SuccessFactors Integration Lead",
    "Senior HRIS Manager",
    "HRIS Solution Architect",
    "HR IT Project Manager",
    "Senior WMS Project Manager",
    "Manhattan WMS Program Manager",
]

EXCLUDED_TITLE = (
    "intern", "internship", "stagiaire", "stage", "alternance", "apprentice",
    "apprenticeship", "graduate", "junior", "entry level", "trainee", "working student",
)

def classify_senior_role(title, description=""):
    title_text = str(title or "").lower()
    full_text = f"{title_text} {str(description or '')}".lower()
    if any(term in title_text for term in EXCLUDED_TITLE):
        return None

    # SuccessFactors is deliberately explicit: generic SAP jobs are not enough.
    sf_context = any(term in full_text for term in (
        "successfactors", "success factors", "sap sf", "employee central",
    ))
    hr_context = any(term in full_text for term in (
        "hris", "sirh", "human resources information", "hr it", "hr technology",
        "hcm", "human capital management", "employee central", "people systems",
        "hr solution", "hr architect", "hr systems", "hr transformation",
    ))
    wms_context = any(term in full_text for term in (
        "wms", "manhattan", "warehouse management", "manhattan associates",
    ))
    senior_title = any(term in title_text for term in (
        "senior", "sr.", "lead", "principal", "head", "director", "manager",
        "architect", "program", "programme",
    ))

    if sf_context and senior_title:
        return "SAP SuccessFactors"
    if wms_context and any(term in title_text for term in (
        "project manager", "program manager", "programme manager", "delivery lead",
        "implementation manager", "project lead", "director",
    )):
        return "Chef de projet IT / WMS"
    if hr_context and senior_title and any(term in title_text for term in (
        "architect", "architecture", "project", "program", "programme", "manager",
        "lead", "director", "product owner", "transformation",
    )):
        if "architect" in title_text or "architecture" in title_text:
            return "Architecte IT RH"
        return "HRIS / SIRH – IT"
    return None
DAYS_BACK = 30
MAX_PER_COUNTRY = 12
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CuratedJobsDashboard/1.0)"}


def _normalise_date(value):
    if value is None or value == "":
        return pd.NaT
    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit="s", utc=True).tz_convert(None)
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


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_himalayas(country_code):
    try:
        response = requests.get(
            "https://himalayas.app/jobs/api/search",
            params={"country": country_code, "seniority": "Senior,Manager,Director,Executive", "sort": "recent", "page": 1},
            headers={**HEADERS, "Accept": "application/json"}, timeout=(4, 12),
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("jobs", []) if isinstance(payload, dict) else None
    except (requests.RequestException, ValueError, TypeError):
        return None


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
    remote_jobs = fetch_himalayas(cfg["code"] or {"France": "FR", "Dubaï": "AE", "Suisse": "CH"}[country])
    for item in remote_jobs or []:
        title = item.get("title", "")
        description = item.get("description") or item.get("excerpt", "")
        role = classify_senior_role(title, description)
        created = _normalise_date(item.get("pubDate"))
        if role and not pd.isna(created):
            jobs.append({
                "title": title, "company": item.get("companyName", "Entreprise non précisée"),
                "location": f"Remote · {country}", "date": created,
                "url": item.get("applicationLink", ""), "source": "Himalayas",
                "description": description, "role": role,
            })
    if cfg["code"]:
        for keyword in ROLE_QUERIES:
            for job in fetch_adzuna(cfg["code"], cfg["where"], keyword):
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
        role = classify_senior_role(title, description)
        if not role:
            continue
        jobs.append({
            "title": title,
            "company": item.get("company_name", "Entreprise non précisée"),
            "location": location or country,
            "date": created,
            "url": item.get("url", ""),
            "source": "Arbeitnow",
            "description": description,
            "role": role,
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
    curated = []
    for job in unique.values():
        role = classify_senior_role(job.get("title"), job.get("description"))
        if not role:
            continue
        job["role"] = role
        curated.append(job)

    for job in curated:
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
    query = quote_plus('("SAP SuccessFactors" OR HRIS OR "HR Solution Architect" OR "HR IT Project Manager" OR "WMS Project Manager" OR Manhattan) AND (Senior OR Lead OR Principal OR Manager OR Director)')
    location = quote_plus(cfg["linkedin"])
    indeed_location = quote_plus(cfg["indeed"])
    indeed_domain = "fr.indeed.com" if country == "France" else "ae.indeed.com" if country == "Dubaï" else "ch.indeed.com"
    links = [
        ("LinkedIn", f"https://www.linkedin.com/jobs/search/?keywords={query}&location={location}&f_TPR=r2592000"),
        ("Indeed", f"https://{indeed_domain}/jobs?q={query}&l={indeed_location}&fromage=30"),
    ]
    if country == "France":
        links.extend([
            ("Welcome to the Jungle", f"https://www.welcometothejungle.com/fr/jobs?query={query}"),
            ("France Travail", "https://candidat.francetravail.fr/offres/emploi"),
        ])
    elif country == "Dubaï":
        links.extend([
            ("Bayt", f"https://www.bayt.com/en/uae/jobs/?q={query}"),
            ("GulfTalent", f"https://www.gulftalent.com/uae/jobs?search={query}"),
        ])
    else:
        links.extend([
            ("jobs.ch", f"https://www.jobs.ch/en/vacancies/?term={query}"),
            ("Jobup.ch", f"https://www.jobup.ch/en/jobs/?term={query}"),
        ])
    return links


def _render_job_card(job, accent, accent_border):
    title = html.escape(str(job.get("title") or "Intitulé non précisé"))
    company = html.escape(str(job.get("company") or "Entreprise non précisée"))
    location = html.escape(str(job.get("location") or "Localisation non précisée"))
    source = html.escape(str(job.get("source") or "Source"))
    role = html.escape(str(job.get("role") or "Offre complémentaire"))
    date = pd.Timestamp(job["date"]).strftime("%d/%m/%Y")
    raw_url = str(job.get("url") or "").strip()
    safe_url = raw_url if raw_url.startswith(("https://", "http://")) else ""
    action = (
        f'<a class="job-apply" href="{html.escape(safe_url, quote=True)}" target="_blank" rel="noopener noreferrer">Voir l’annonce ↗</a>'
        if safe_url else '<span class="job-no-link">Lien direct indisponible</span>'
    )
    title_markup = (
        f'<a class="job-title-link" href="{html.escape(safe_url, quote=True)}" target="_blank" rel="noopener noreferrer">{title} ↗</a>'
        if safe_url else f'<span class="job-title-link">{title}</span>'
    )
    st.markdown(
        f"""
        <div class="job-card" style="background-color:{accent};border-top:5px solid {accent_border};color:#17202a">
          <div class="job-card-top">
            <span class="job-source">{source}</span>
            <span class="job-date">{date}</span>
          </div>
          <div class="job-title">{title_markup}</div>
          <div class="job-company">{company}</div>
          <div class="job-location">⌖ &nbsp;{location}</div>
          <div class="job-tags"><span>{role}</span></div>
          <div class="job-card-bottom">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def jobs_dashboard():
    st.markdown(
        """
        <style>
        .jobs-layout-note {color:var(--text-color);opacity:.72;font-size:.88rem;margin-top:-.45rem}
        .job-card {
          min-height: 222px; height:100%; padding:17px; margin:0 0 14px 0;
          border:1px solid rgba(0,0,0,.12);
          border-radius:15px; color:#17202a !important;
          display:flex; flex-direction:column; gap:9px;
          box-shadow:0 3px 12px rgba(0,0,0,.10);
        }
        .job-card-top {display:flex;justify-content:space-between;gap:8px;align-items:center}
        .job-source {font-size:.82rem;font-weight:850;letter-spacing:.03em;text-transform:uppercase;color:#263442;opacity:1}
        .job-date {font-size:.82rem;color:#263442;opacity:1;white-space:nowrap}
        .job-title {font-size:1.28rem;line-height:1.4;font-weight:850;min-height:2.8em}
        .job-title-link {color:#17202a!important;text-decoration:none!important}
        .job-title-link:hover {text-decoration:underline!important;color:#334155!important}
        .job-company {font-size:1.08rem;font-weight:750;color:#17202a;opacity:1}
        .job-location {font-size:.94rem;color:#263442;opacity:1}
        .job-tags {margin-top:2px}
        .job-tags span {display:inline-block;border:1px solid rgba(23,32,42,.28);border-radius:20px;padding:5px 10px;font-size:.78rem;font-weight:700;color:#17202a;opacity:1}
        .job-card-bottom {margin-top:auto;padding-top:8px;display:flex;justify-content:flex-end}
        .job-apply {display:inline-block;background:#ffffff;color:#17202a!important;text-decoration:none!important;padding:10px 16px;border:1px solid rgba(23,32,42,.2);border-radius:18px;font-size:.9rem;font-weight:850;box-shadow:0 2px 5px rgba(0,0,0,.12)}
        .job-apply:hover {filter:brightness(.94)}
        .job-no-link {font-size:.76rem;opacity:.55}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("💼 Offres d’emploi")
    st.markdown(
        '<div class="jobs-layout-note">France · Dubaï · Suisse — postes expérimentés en IT RH, SAP SuccessFactors, architecture et projets WMS</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Filtres")
        st.caption("Zone géographique")
        selected_countries = []
        for country in COUNTRIES:
            if st.checkbox(country, value=True, key=f"jobs_country_{country}"):
                selected_countries.append(country)

        st.divider()
        st.caption("Famille de poste")
        selected_roles = []
        for role in ROLE_SEARCHES:
            if st.checkbox(role, value=True, key=f"jobs_role_{role}"):
                selected_roles.append(role)

        st.divider()
        recency = st.selectbox(
            "Date de publication",
            ["30 derniers jours", "7 derniers jours", "14 derniers jours"],
            index=0,
            key="jobs_recency",
        )
        source_filter = st.selectbox(
            "Source",
            ["Toutes les sources", "Himalayas", "Arbeitnow"],
            index=0,
            key="jobs_source",
        )
        st.divider()
        sort_mode = st.selectbox(
            "Trier par",
            ["Pertinence", "Plus récentes"],
            index=0,
            key="jobs_sort",
        )

    if not selected_countries:
        st.info("Sélectionne au moins un pays dans les filtres.")
        return

    collected = []
    with st.spinner("Recherche des offres correspondant aux filtres…"):
        for country in selected_countries:
            country_jobs, _ = fetch_country_jobs(country)
            for job in country_jobs:
                job = dict(job)
                job["country"] = country
                collected.append(job)

    days_limit = {"7 derniers jours": 7, "14 derniers jours": 14, "30 derniers jours": 30}[recency]
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_limit)
    filtered = [
        job for job in collected
        if pd.Timestamp(job["date"]) >= cutoff
        and job.get("role") in selected_roles
        and (source_filter == "Toutes les sources" or job.get("source") == source_filter)
    ]
    if sort_mode == "Plus récentes":
        filtered.sort(key=lambda job: job["date"], reverse=True)
    else:
        filtered.sort(key=lambda job: (job.get("score", 0), job["date"]), reverse=True)

    header_left, header_right = st.columns([2, 1])
    with header_left:
        st.markdown(f"### {len(filtered)} offre(s) trouvée(s)")
        st.caption(f"Sur {len(selected_countries)} zone(s) sélectionnée(s) · {recency.lower()}")
    with header_right:
        st.caption("Recherche complémentaire")
        all_links = []
        for country in selected_countries:
            all_links.extend((country, label, url) for label, url in _external_search_links(country))
        with st.popover("Job boards par pays", use_container_width=True):
            for country in selected_countries:
                st.markdown(f"**{country}**")
                for _, label, url in [x for x in all_links if x[0] == country]:
                    st.markdown(f"- [{label}]({url})")

    # Trois colonnes fixes : les offres ne sont jamais mélangées entre pays.
    country_styles = {
        "France": {"flag": "🇫🇷", "bg": "#DCEBFA", "border": "#3978B9"},
        "Dubaï": {"flag": "🇦🇪", "bg": "#FCE8CC", "border": "#C47A25"},
        "Suisse": {"flag": "🇨🇭", "bg": "#F7D9DE", "border": "#C44755"},
    }
    columns = st.columns(3, gap="medium")
    for country, column in zip(COUNTRIES.keys(), columns):
        country_jobs = [job for job in filtered if job.get("country") == country]
        style = country_styles[country]
        with column:
            st.markdown(
                f"<div style='padding:12px 14px;margin:0 0 12px 0;"
                f"background:{style['bg']};border-left:5px solid {style['border']};"
                f"border-radius:12px;color:#17202a;font-weight:800;font-size:1.05rem'>"
                f"{style['flag']} {country} <span style='float:right'>"
                f"{len(country_jobs)}</span></div>",
                unsafe_allow_html=True,
            )
            if country not in selected_countries:
                st.caption("Pays désactivé dans les filtres.")
            elif country_jobs:
                for job in country_jobs:
                    _render_job_card(job, style["bg"], style["border"])
            else:
                st.info("Aucune offre trouvée avec les filtres actuels.")

    with st.expander("Sources et périmètre de recherche"):
        st.markdown(
            "- **Sources automatiques :** Himalayas (API publique sans token, postes à distance) et Arbeitnow.\n"
            "- **Sources complémentaires :** LinkedIn, Indeed et job boards locaux via recherche directe.\n"
            "- **Périmètre :** postes expérimentés en IT RH / HRIS, SAP SuccessFactors, architecture IT RH "
            "et gestion de projets IT / WMS (Manhattan). Stages, alternances et postes juniors exclus.\n"
            "- **Tri :** pertinence des mots-clés et date, ou date seule selon le filtre."
        )
        st.caption(
            "LinkedIn est proposé en recherche directe : son accès public ne permet pas "
            "une agrégation libre et fiable de toutes ses annonces."
        )
