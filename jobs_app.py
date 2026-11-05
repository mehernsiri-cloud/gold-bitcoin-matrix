# jobs_app.py
"""Curated job dashboard focused on France, Dubai and Switzerland."""
import html
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

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
        "supply chain", "logistics", "distribution center", "distribution centre",
    ))
    it_context = any(term in full_text for term in (
        "information technology", " it ", "technology", "digital", "enterprise systems",
        "business systems", "applications", "infrastructure", "erp", "crm",
        "software implementation", "it transformation", "project delivery",
        "program delivery", "programme delivery", "system integration",
    ))
    senior_title = any(term in title_text for term in (
        "senior", "sr.", "lead", "principal", "head", "director", "manager",
        "architect", "program", "programme", "project", "delivery",
    ))

    if sf_context and senior_title:
        return "SAP SuccessFactors"

    # Architecture roles are eligible when clearly linked to IT, enterprise systems or HR technology.
    architecture_title = any(term in title_text for term in (
        "architect", "architecture", "enterprise architect", "solution architect",
        "technical architect", "it architect", "systems architect",
    ))
    if architecture_title and senior_title and (hr_context or it_context or wms_context):
        return "Architecte IT RH"

    project_title = any(term in title_text for term in (
        "project manager", "program manager", "programme manager",
        "project director", "program director", "programme director",
        "delivery manager", "delivery lead", "project lead",
        "chef de projet", "directeur de projet",
    ))
    if project_title and senior_title and (it_context or hr_context or wms_context):
        return "Chef de projet IT / WMS"

    if hr_context and senior_title and any(term in title_text for term in (
        "project", "program", "programme", "manager", "lead", "director",
        "product owner", "transformation",
    )):
        return "HRIS / SIRH – IT"
    return None

DAYS_BACK = 30
MAX_PER_COUNTRY = 20
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


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_himalayas(country_code, keyword):
    try:
        response = requests.get(
            "https://himalayas.app/jobs/api/search",
            params={"country": country_code, "q": keyword, "seniority": "Senior,Manager,Director,Executive", "sort": "recent", "page": 1},
            headers={**HEADERS, "Accept": "application/json"}, timeout=(4, 12),
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("jobs", []) if isinstance(payload, dict) else None
    except (requests.RequestException, ValueError, TypeError):
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_jobicy():
    try:
        response = requests.get(
            "https://jobicy.com/api/jobs",
            params={"count": 100},
            headers={**HEADERS, "Accept": "application/json"},
            timeout=(4, 12),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("success") is False:
            return None
        return payload.get("jobs", [])
    except (requests.RequestException, ValueError, TypeError):
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_remoteok():
    try:
        response = requests.get(
            "https://remoteok.com/api",
            headers={**HEADERS, "Accept": "application/json"},
            timeout=(4, 12),
        )
        response.raise_for_status()
        payload = response.json()
        return payload[1:] if isinstance(payload, list) else None
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
        return payload.get("data", []) if isinstance(payload, dict) else None
    except (requests.RequestException, ValueError, TypeError):
        return None


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
    source_status = {}
    country_code = {"France": "FR", "Dubaï": "AE", "Suisse": "CH"}[country]
    search_terms = ["SAP SuccessFactors", "HRIS HR IT", "HR Solution Architect", "WMS Project Manager Manhattan"]
    source_ok = True
    for keyword in search_terms:
        remote_jobs = fetch_himalayas(country_code, keyword)
        if remote_jobs is None:
            source_ok = False
            continue
        for item in remote_jobs:
            title = item.get("title", "")
            description = item.get("description") or item.get("excerpt", "")
            role = classify_senior_role(title, description)
            created = _normalise_date(item.get("pubDate"))
            if role and not pd.isna(created):
                guid = str(item.get("guid") or "")
                source_url = guid if guid.startswith("http") else ("https://himalayas.app" + guid if guid.startswith("/") else "https://himalayas.app")
                jobs.append({
                    "title": title,
                    "company": item.get("companyName", "Entreprise non précisée"),
                    "location": f"Remote · {country}",
                    "date": created,
                    "url": item.get("applicationLink", ""),
                    "source_url": source_url,
                    "source": "Himalayas",
                    "description": description,
                    "role": role,
                })
    source_status["Himalayas"] = "OK" if source_ok else "Erreur partielle/API"

    jobicy = fetch_jobicy()
    source_status["Jobicy"] = "OK" if jobicy is not None else "Erreur API"
    for item in jobicy or []:
        geo = str(item.get("jobGeo") or "")
        geo_lower = geo.lower()
        eligible = (
            "anywhere" in geo_lower or "worldwide" in geo_lower
            or (country == "France" and any(x in geo_lower for x in ("france", "europe")))
            or (country == "Suisse" and any(x in geo_lower for x in ("switzerland", "suisse", "europe")))
            or (country == "Dubaï" and any(x in geo_lower for x in ("united arab emirates", "uae", "dubai")))
        )
        if not eligible:
            continue
        title = item.get("jobTitle", "")
        description = item.get("jobDescription") or item.get("jobExcerpt", "")
        role = classify_senior_role(title, description)
        created = _normalise_date(item.get("pubDate"))
        if role and not pd.isna(created):
            url = item.get("url", "")
            jobs.append({
                "title": title, "company": item.get("companyName", "Entreprise non précisée"),
                "location": f"Remote · {geo or country}", "date": created,
                "url": url, "source_url": url, "source": "Jobicy",
                "description": description, "role": role,
            })

    remoteok = fetch_remoteok()
    source_status["Remote OK"] = "OK" if remoteok is not None else "Erreur API"
    for item in remoteok or []:
        location = str(item.get("location") or "")
        location_lower = location.lower()
        eligible = (
            any(x in location_lower for x in ("worldwide", "anywhere", "remote - global"))
            or _arbeitnow_country_match(location, country)
        )
        if not eligible:
            continue
        title = item.get("position", "")
        description = item.get("description", "")
        role = classify_senior_role(title, description)
        created = _normalise_date(item.get("date"))
        if role and not pd.isna(created):
            url = item.get("url", "")
            jobs.append({
                "title": title, "company": item.get("company", "Entreprise non précisée"),
                "location": location or f"Remote · {country}", "date": created,
                "url": url, "source_url": url, "source": "Remote OK",
                "description": description, "role": role,
            })

    arbeitnow = fetch_arbeitnow()
    source_status["Arbeitnow"] = "OK" if arbeitnow is not None else "Erreur API"
    for item in arbeitnow or []:
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
        if key and (key not in unique or job["date"] > unique[key]["date"]):
            unique[key] = job

    now = pd.Timestamp.utcnow().tz_localize(None)
    relevance_terms = {
        "successfactors": 6, "success factors": 6, "hris": 7, "sirh": 7, "hcm": 5,
        "integration": 5, "cpi": 5, "btp": 5, "architect": 7, "architecture": 6,
        "project manager": 7, "program manager": 7, "programme manager": 7,
        "project director": 6, "delivery manager": 6, "manhattan": 6, "wms": 6,
        "warehouse management": 5, "enterprise": 3, "solution architect": 7,
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

    # Keep the country limit, but reserve space across role families so that
    # SuccessFactors does not crowd out project, architecture and WMS positions.
    ranked.sort(key=lambda x: (x["score"], x["date"]), reverse=True)
    role_order = [
        "HRIS / SIRH – IT", "SAP SuccessFactors",
        "Architecte IT RH", "Chef de projet IT / WMS",
    ]
    buckets = {
        role: [job for job in ranked if job.get("role") == role]
        for role in role_order
    }
    selected = []
    selected_ids = set()
    per_role_limit = max(1, MAX_PER_COUNTRY // len(role_order))
    for role in role_order:
        for job in buckets[role][:per_role_limit]:
            selected.append(job)
            selected_ids.add(id(job))
    if len(selected) < MAX_PER_COUNTRY:
        selected.extend(
            job for job in ranked
            if id(job) not in selected_ids
        )
    selected.sort(key=lambda x: (x["score"], x["date"]), reverse=True)
    return selected[:MAX_PER_COUNTRY], source_status


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
    title_url = str(job.get("source_url") or safe_url)
    safe_title_url = title_url if title_url.startswith(("https://", "http://")) else ""
    title_markup = (
        f'<a class="job-title-link" href="{html.escape(safe_title_url, quote=True)}" target="_blank" rel="noopener noreferrer">{title} ↗</a>'
        if safe_title_url else f'<span class="job-title-link">{title}</span>'
    )
    source_markup = (
        f'<a href="{html.escape(str(job.get("source_url")), quote=True)}" target="_blank" rel="noopener noreferrer">{source}</a>'
        if job.get("source_url") else source
    )
    st.markdown(
        f"""
        <div class="job-card" style="background-color:{accent};border-top:5px solid {accent_border};color:#17202a">
          <div class="job-card-top">
            <span class="job-source">{source_markup}</span>
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
            ["Toutes les sources", "Himalayas", "Jobicy", "Remote OK", "Arbeitnow"],
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
    collection_status = {}
    seen_offers = set()
    with st.spinner("Recherche des offres correspondant aux filtres…"):
        for country in selected_countries:
            country_jobs, source_status = fetch_country_jobs(country)
            collection_status[country] = source_status
            for job in country_jobs:
                job = dict(job)
                # Une même annonce (notamment remote/worldwide) ne doit appartenir
                # qu'à un seul pays. L'URL normalisée est la clé prioritaire.
                raw_url = str(job.get("url") or "").strip()
                canonical_url = raw_url.split("#", 1)[0].split("?", 1)[0].rstrip("/").lower()
                if canonical_url:
                    dedupe_key = ("url", canonical_url)
                else:
                    dedupe_key = (
                        "fallback",
                        str(job.get("title") or "").strip().lower(),
                        str(job.get("company") or "").strip().lower(),
                        str(job.get("location") or "").strip().lower(),
                    )
                if dedupe_key in seen_offers:
                    continue
                seen_offers.add(dedupe_key)
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

    with st.expander("État des sources automatiques"):
        for country, statuses in collection_status.items():
            st.markdown(f"**{country}**")
            for source_name, status in statuses.items():
                st.caption(f"{source_name} : {status}")

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
                # Grille de deux annonces par rangée à l'intérieur de chaque pays.
                for row_start in range(0, len(country_jobs), 2):
                    offer_columns = st.columns(2, gap="small")
                    for offer_column, job in zip(offer_columns, country_jobs[row_start:row_start + 2]):
                        with offer_column:
                            _render_job_card(job, style["bg"], style["border"])
            else:
                st.info("Aucune offre trouvée avec les filtres actuels.")

    with st.expander("Sources et périmètre de recherche"):
        st.markdown(
            "- **Sources automatiques :** Himalayas, Jobicy, Remote OK et Arbeitnow (flux publics sans clé).\n"
            "- **Sources complémentaires :** LinkedIn, Indeed et job boards locaux via recherche directe.\n"
            "- **Périmètre :** postes expérimentés en IT RH / HRIS, SAP SuccessFactors, architecture IT RH "
            "et gestion de projets IT / WMS (Manhattan). Stages, alternances et postes juniors exclus.\n"
            "- **Tri :** pertinence des mots-clés et date, ou date seule selon le filtre."
        )
        st.caption(
            "LinkedIn est proposé en recherche directe : son accès public ne permet pas "
            "une agrégation libre et fiable de toutes ses annonces."
        )
