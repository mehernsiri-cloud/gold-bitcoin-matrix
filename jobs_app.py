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
    "Dubaï": {"code": None, "where": "Dubai", "linkedin": "Dubai, United Arab Emirates", "indeed": "Dubai"},
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
    if cfg["code"]:
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


def _render_job_card(job, accent):
    title = html.escape(str(job.get("title") or "Intitulé non précisé"))
    company = html.escape(str(job.get("company") or "Entreprise non précisée"))
    location = html.escape(str(job.get("location") or "Localisation non précisée"))
    source = html.escape(str(job.get("source") or "Source"))
    role = html.escape(str(job.get("role") or "Offre complémentaire"))
    date = pd.Timestamp(job["date"]).strftime("%d/%m/%Y")
    raw_url = str(job.get("url") or "").strip()
    safe_url = raw_url if raw_url.startswith(("https://", "http://")) else ""
    action = (
        f'<a class="job-apply" href="{html.escape(safe_url, quote=True)}" target="_blank" rel="noopener noreferrer">Voir l’offre ↗</a>'
        if safe_url else '<span class="job-no-link">Lien non disponible</span>'
    )
    st.markdown(
        f"""
        <div class="job-card" style="--job-accent:{accent}">
          <div class="job-card-top">
            <span class="job-source">{source}</span>
            <span class="job-date">{date}</span>
          </div>
          <div class="job-title">{title}</div>
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
          min-height: 222px; height:100%; padding:16px; margin:0 0 14px 0;
          border:1px solid rgba(128,128,128,.22); border-top:4px solid var(--job-accent);
          border-radius:15px; background:var(--secondary-background-color);
          display:flex; flex-direction:column; gap:9px;
          box-shadow:0 2px 8px rgba(0,0,0,.035);
        }
        .job-card-top {display:flex;justify-content:space-between;gap:8px;align-items:center}
        .job-source {font-size:.72rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;opacity:.72}
        .job-date {font-size:.72rem;opacity:.68;white-space:nowrap}
        .job-title {font-size:1.03rem;line-height:1.3;font-weight:750;min-height:2.6em}
        .job-company {font-size:.9rem;font-weight:600;opacity:.88}
        .job-location {font-size:.82rem;opacity:.72}
        .job-tags {margin-top:2px}
        .job-tags span {display:inline-block;border:1px solid rgba(128,128,128,.35);border-radius:20px;padding:4px 9px;font-size:.69rem;opacity:.82}
        .job-card-bottom {margin-top:auto;padding-top:8px;display:flex;justify-content:flex-end}
        .job-apply {display:inline-block;background:var(--job-accent);color:#17202a!important;text-decoration:none!important;padding:7px 12px;border-radius:18px;font-size:.78rem;font-weight:750}
        .job-no-link {font-size:.76rem;opacity:.55}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("💼 Offres d’emploi")
    st.markdown(
        '<div class="jobs-layout-note">France · Dubaï · Suisse — offres ciblées HRIS, SAP, intégration et transformation</div>',
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
        if st.checkbox("Autres offres pertinentes", value=True, key="jobs_role_other"):
            selected_roles.append("Offre complémentaire")

        st.divider()
        recency = st.selectbox(
            "Date de publication",
            ["30 derniers jours", "7 derniers jours", "14 derniers jours"],
            index=0,
            key="jobs_recency",
        )
        source_filter = st.selectbox(
            "Source",
            ["Toutes les sources", "Adzuna", "Arbeitnow"],
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
        and (not selected_roles or job.get("role") in selected_roles)
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

    if filtered:
        accents = ["#f4d58d", "#c9b8f5", "#f4b5c2", "#a9d9ed", "#b8dfbd", "#f2c6a8"]
        columns = st.columns(3, gap="medium")
        for index, job in enumerate(filtered):
            with columns[index % 3]:
                _render_job_card(job, accents[index % len(accents)])
    else:
        st.warning(
            "Aucune offre ne correspond aux filtres sélectionnés dans les flux automatiques. "
            "Cela ne signifie pas qu’il n’existe aucune offre : utilise les job boards par pays "
            "pour élargir la recherche."
        )

    with st.expander("Sources et périmètre de recherche"):
        st.markdown(
            "- **Sources automatiques :** Adzuna et Arbeitnow (couverture variable selon le pays).\n"
            "- **Sources complémentaires :** LinkedIn, Indeed et job boards locaux via recherche directe.\n"
            "- **Périmètre :** HRIS / SAP SuccessFactors, SAP Integration / BTP / CPI, architecture RH, "
            "transformation RH et, en secondaire, Supply Chain / WMS.\n"
            "- **Tri :** pertinence des mots-clés et date, ou date seule selon le filtre."
        )
        st.caption(
            "LinkedIn est proposé en recherche directe : son accès public ne permet pas "
            "une agrégation libre et fiable de toutes ses annonces."
        )
