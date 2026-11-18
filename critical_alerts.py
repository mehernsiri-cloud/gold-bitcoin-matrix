"""Critical daily-life and systemic-risk alerts.

Aggregates official CERT/government advisories and current reporting on incidents
that may disrupt banking, payments, telecoms, energy, healthcare, transport or
other essential services. A news report is not automatically a confirmed attack.
"""
import re
import html
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import feedparser
import pandas as pd
import requests
import streamlit as st

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CriticalRiskMonitor/1.0)"}
TIMEOUT = 12
MAX_AGE_DAYS = 14

OFFICIAL_FEEDS = {
    "CERT-FR / ANSSI": "https://www.cert.ssi.gouv.fr/alerte/feed/",
    "CISA – États-Unis": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
    "NCSC – Royaume-Uni": "https://www.ncsc.gov.uk/api/1/services/v1/report-rss-feed.xml",
}
NEWS_FEEDS = {
    "Cyber – incidents internationaux": "https://www.bleepingcomputer.com/feed/",
    "The Record – cybersécurité": "https://therecord.media/feed",
    "Krebs on Security": "https://krebsonsecurity.com/feed/",
}
GOOGLE_NEWS_QUERIES = {
    "France – banques, paiements et services essentiels":
        '(cyberattaque OR ransomware OR panne OR piratage) (banque OR paiement OR hôpital OR énergie OR télécom OR transport) France when:7d',
    "États-Unis – banques et infrastructures":
        '(cyberattack OR ransomware OR bank outage OR payment outage) (bank OR payments OR power grid OR telecom OR hospital) United States when:7d',
    "International – infrastructures critiques":
        '(cyberattack OR ransomware OR outage) (banking OR SWIFT OR electricity OR energy OR telecom OR hospital OR transport) when:7d',
}

KEYWORDS = {
    "Banques & paiements": [
        "bank", "banque", "banking", "payment", "paiement", "swift", "visa outage",
        "mastercard", "zelle", "virement", "atm", "distributeur", "financial system",
    ],
    "Énergie & eau": [
        "power grid", "electricity", "électricité", "énerg", "electric grid",
        "blackout", "power outage", "water supply", "eau potable", "pipeline",
    ],
    "Télécoms & Internet": [
        "telecom", "télécom", "internet outage", "mobile network", "réseau mobile",
        "cloudflare outage", "dns outage", "通信",
    ],
    "Santé & hôpitaux": [
        "hospital", "hôpital", "healthcare", "santé", "clinic", "patient",
        "medical systems", "pharmacy", "pharmacie",
    ],
    "Transport & logistique": [
        "airport", "aéroport", "rail", "train", "metro", "métro", "port",
        "transport", "logistics", "logistique", "air traffic",
    ],
    "Cyber & données": [
        "cyberattack", "cyberattaque", "ransomware", "data breach", "fuite de données",
        "hacked", "hacké", "intrusion", "malware", "ddos", "supply chain",
    ],
}
CRITICAL_TERMS = [
    "nationwide", "national outage", "major outage", "widespread", "critical infrastructure",
    "infrastructure critique", "banking system", "payment system", "payment outage",
    "power grid", "blackout", "emergency", "urgence", "massive outage",
]
HIGH_TERMS = [
    "cyberattack", "cyberattaque", "ransomware", "bank", "banque", "hospital", "hôpital",
    "energy", "énergie", "telecom", "télécom", "government", "gouvernement",
    "data breach", "fuite de données", "outage", "panne", "swift",
]
COUNTRY_TERMS = {
    "France": ["france", "french", "français", "française", "paris", "anssi", "cert-fr"],
    "États-Unis": ["united states", "u.s.", "usa", "american", "fbi", "cisa", "new york"],
    "Europe": ["europe", "european", "eu ", "european union", "européen"],
    "International": [],
}

def _clean(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(value or "")))).strip()

def _parse_date(entry):
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                import calendar
                return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            except Exception:
                pass
    return None

def _classify(title, summary, source, official=False):
    text = (title + " " + summary).lower()
    # One primary family per incident, avoiding multi-category duplication.
    family_rules = [
        ("Banques & paiements", ["bank", "banque", "banking", "payment", "paiement", "swift", "visa", "mastercard", "zelle", "virement", "atm", "financial system"]),
        ("Énergie & eau", ["power grid", "electricity", "électricité", "énerg", "electric grid", "blackout", "power outage", "water supply", "eau potable", "pipeline"]),
        ("Télécoms & Internet", ["telecom", "télécom", "internet outage", "mobile network", "réseau mobile", "cloudflare", "dns outage"]),
        ("Santé & hôpitaux", ["hospital", "hôpital", "healthcare", "santé", "clinic", "patient", "medical systems", "pharmacy", "pharmacie"]),
        ("Transport & logistique", ["airport", "aéroport", "rail", "train", "metro", "métro", "port", "transport", "logistics", "logistique", "air traffic"]),
        ("Cyber & données", ["cyberattack", "cyberattaque", "ransomware", "data breach", "fuite de données", "hacked", "hacké", "intrusion", "malware", "ddos", "supply chain"]),
    ]
    category = next((name for name, words in family_rules if any(w in text for w in words)), "Autres risques essentiels")
    countries = [name for name, words in COUNTRY_TERMS.items() if any(w in text for w in words)]
    if not countries:
        countries = ["À préciser"]
    if any(w in text for w in CRITICAL_TERMS):
        severity, score = "CRITIQUE", 95
    elif any(w in text for w in HIGH_TERMS):
        severity, score = "ÉLEVÉ", 75
    else:
        severity, score = "À SURVEILLER", 45
    confidence = "Source officielle" if official else "Information média – à corroborer"
    return severity, score, category, ", ".join(countries[:2]), confidence

# Words that add little value when comparing incident headlines.
STOPWORDS = set("""
the and for with from that this have has was are were into after over under about
dans pour avec sans sur les des une un aux par est sont cette ces leur leurs
lorsque selon après avant entre contre chez depuis mais plus moins
cyber cyberattack cyberattaque attack attaque incident update latest
""".split())

def _tokens(value):
    words = re.findall(r"[a-z0-9]{3,}", _clean(value).lower())
    return {w for w in words if w not in STOPWORDS and not w.isdigit()}

def _headline_similarity(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))

def _same_incident(a, b):
    if a["category"] != b["category"]:
        return False
    ca = set(a["country"].split(", "))
    cb = set(b["country"].split(", "))
    if ca != {"À préciser"} and cb != {"À préciser"} and not (ca & cb):
        return False
    title_score = _headline_similarity(a["title"], b["title"])
    combined_score = _headline_similarity(
        a["title"] + " " + a["summary"][:250],
        b["title"] + " " + b["summary"][:250],
    )
    # Require topical overlap; a shared company name alone (e.g. Siemens)
    # must never merge unrelated incidents.
    return title_score >= 0.48 or combined_score >= 0.43

def _group_incidents(items):
    """Cluster cross-source coverage of the same event into one incident card."""
    clusters = []
    ordered = sorted(
        items,
        key=lambda x: (x["score"], x["published"] or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    for item in ordered:
        match = None
        for cluster in clusters:
            if any(_same_incident(item, member) for member in cluster["articles"]):
                match = cluster
                break
        if match is None:
            clusters.append({"articles": [item]})
        else:
            match["articles"].append(item)

    grouped = []
    for cluster in clusters:
        articles = cluster["articles"]
        primary = max(articles, key=lambda x: (x["score"], bool(x["published"])))
        merged = dict(primary)
        merged["sources"] = list(dict.fromkeys(x["source"] for x in articles))
        merged["source_count"] = len(merged["sources"])
        merged["article_count"] = len(articles)
        merged["related_links"] = [
            {"source": x["source"], "url": x["url"], "title": x["title"]}
            for x in articles
        ]
        dates = [x["published"] for x in articles if x["published"]]
        merged["published"] = max(dates) if dates else None
        # Retain official confirmation if any article in the cluster is official.
        if any(x["confidence"] == "Source officielle" for x in articles):
            merged["confidence"] = "Au moins une source officielle"
        grouped.append(merged)
    grouped.sort(
        key=lambda x: (x["score"], x["published"] or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    return grouped

def _google_news_url(query):
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=fr&gl=FR&ceid=FR:fr"

@st.cache_data(ttl=600, show_spinner=False)
def load_critical_alerts():
    feeds = {}
    feeds.update({k: (v, True) for k, v in OFFICIAL_FEEDS.items()})
    feeds.update({k: (v, False) for k, v in NEWS_FEEDS.items()})
    feeds.update({k: (_google_news_url(v), False) for k, v in GOOGLE_NEWS_QUERIES.items()})
    items, statuses = [], {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    for source, (url, official) in feeds.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            count = 0
            for entry in parsed.entries[:35]:
                title = _clean(entry.get("title"))
                link = entry.get("link", "")
                summary = _clean(entry.get("summary", entry.get("description", "")))[:900]
                if not title or not link:
                    continue
                published = _parse_date(entry)
                if published and published < cutoff:
                    continue
                severity, score, category, country, confidence = _classify(title, summary, source, official)
                items.append({
                    "title": title, "summary": summary, "url": link,
                    "source": source, "published": published,
                    "severity": severity, "score": score, "category": category,
                    "country": country, "confidence": confidence,
                })
                count += 1
            statuses[source] = f"OK · {count} publication(s)"
        except Exception as exc:
            statuses[source] = f"Indisponible · {type(exc).__name__}"

    # First remove exact duplicate URLs/headlines, then cluster cross-media coverage.
    seen = set()
    deduped = []
    for item in items:
        key = (item["url"].split("?")[0].rstrip("/"), re.sub(r"[^a-z0-9]", "", item["title"].lower()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    incidents = _group_incidents(deduped)
    return incidents, statuses

def _fmt_date(value):
    return value.astimezone().strftime("%d/%m/%Y · %H:%M") if value else "Date non fournie par la source"

def _render_alert(item):
    colors = {"CRITIQUE": "#b42318", "ÉLEVÉ": "#c66a12", "À SURVEILLER": "#64748b"}
    color = colors.get(item["severity"], "#64748b")
    title = html.escape(item["title"])
    summary = html.escape(item["summary"][:360])
    source = html.escape(item["source"])
    category = html.escape(item["category"])
    country = html.escape(item["country"])
    confidence = html.escape(item["confidence"])
    source_count = int(item.get("source_count", 1))
    source_label = f"{source_count} sources recoupées" if source_count > 1 else "1 source"
    url = html.escape(item["url"], quote=True)
    st.markdown(f"""
    <div style="height:100%;min-height:255px;border:1px solid rgba(128,128,128,.35);
      border-left:6px solid {color};border-radius:13px;padding:15px;
      background:var(--secondary-background-color);display:flex;flex-direction:column">
      <div style="display:flex;justify-content:space-between;gap:8px;align-items:center">
        <span style="background:{color};color:white;padding:5px 9px;border-radius:6px;
          font-size:12px;font-weight:800">{item["severity"]}</span>
        <span style="font-size:11px;opacity:.7">{_fmt_date(item["published"])}</span>
      </div>
      <div style="font-size:16px;font-weight:750;line-height:1.35;margin:12px 0 8px">{title}</div>
      <div style="font-size:13px;opacity:.85;line-height:1.45;flex:1">{summary}</div>
      <div style="font-size:12px;margin-top:12px;opacity:.8">{category} · {country}</div>
      <div style="font-size:11px;margin:5px 0 10px;opacity:.7">{source} · {confidence} · {source_label}</div>
      <a href="{url}" target="_blank" rel="noopener noreferrer"
       style="display:block;text-align:center;padding:8px;border:1px solid rgba(128,128,128,.45);
       border-radius:8px;text-decoration:none;font-weight:650">Lire la source ↗</a>
    </div>
    """, unsafe_allow_html=True)

def render_critical_alerts_dashboard():
    st.title("🚨 Critical Alerts – Daily Life & Systemic Risks")
    st.markdown(
        "Page d'accueil prioritaire : événements pouvant affecter **banques, paiements, énergie, eau, "
        "télécommunications, santé, transports et infrastructures essentielles**. "
        "Les articles parlant du même incident sont regroupés en une seule fiche."
    )
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("↻ Actualiser", use_container_width=True):
            load_critical_alerts.clear()
            st.rerun()
    with c2:
        st.caption("Actualisation automatique toutes les 10 min · événements des 14 derniers jours · doublons multi-sources regroupés")

    with st.spinner("Analyse des sources et regroupement des événements similaires…"):
        incidents, statuses = load_critical_alerts()

    critical = [x for x in incidents if x["severity"] == "CRITIQUE"]
    high = [x for x in incidents if x["severity"] == "ÉLEVÉ"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents critiques", len(critical))
    c2.metric("Incidents élevés", len(high))
    c3.metric("Incidents uniques", len(incidents))
    c4.metric("Sources actives", sum(v.startswith("OK") for v in statuses.values()))
    if critical:
        st.error("🚨 Au moins un événement est classé CRITIQUE. Vérifie son niveau de confirmation et les conséquences concrètes.")
    elif high:
        st.warning("⚠️ Des événements à vigilance élevée sont remontés.")
    else:
        st.info("Aucun événement classé critique ou élevé dans les flux collectés. Cela ne garantit pas l'absence d'incident.")

    # Family sections prevent unrelated topics and repeated company coverage from
    # being mixed into a single undifferentiated stream.
    family_order = [
        "Banques & paiements", "Énergie & eau", "Télécoms & Internet",
        "Santé & hôpitaux", "Transport & logistique", "Cyber & données",
        "Autres risques essentiels",
    ]
    severity_order = {"CRITIQUE": 0, "ÉLEVÉ": 1, "À SURVEILLER": 2}
    for family in family_order:
        family_items = [x for x in incidents if x["category"] == family]
        if not family_items:
            continue
        family_items.sort(key=lambda x: (severity_order.get(x["severity"], 9), -(x["published"].timestamp() if x["published"] else 0)))
        # Show only the two most important incidents in each family.
        # Sorting prioritizes severity, then the most recent publication date.
        top_items = family_items[:2]
        st.subheader(f"{family} · {len(top_items)} événement(s) principal(aux)")
        cols = st.columns(2, gap="medium")
        for col, item in zip(cols, top_items):
            with col:
                _render_alert(item)

    with st.expander("Sources complémentaires et méthode de regroupement", expanded=False):
        for item in incidents:
            if item["source_count"] > 1:
                st.markdown(f"**{item['title']}** — couvert par {item['source_count']} sources")
                for related in item["related_links"]:
                    st.markdown(f"- [{related['source']} : {related['title']}]({related['url']})")
        st.divider()
        for source, status in statuses.items():
            st.write(f"**{source}** — {status}")
        st.caption(
            "Regroupement par famille, pays et similarité du titre/résumé. Le nom d'une entreprise "
            "commun à plusieurs articles ne suffit pas à fusionner deux événements : le sujet doit "
            "également être similaire. Les niveaux de gravité sont des règles de tri, pas une "
            "certification officielle. Les médias sont à corroborer."
        )
