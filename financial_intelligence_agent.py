"""Free financial event-intelligence agent.

No paid LLM/API: RSS collection + deterministic, explainable rules.
Verification here means source/date/link validation and cross-source corroboration,
not independent fact-checking of the underlying claims.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

try:
    import feedparser
except ImportError:
    feedparser = None

DATA_DIR = Path("data")
EVENTS_FILE = DATA_DIR / "financial_events.csv"
ALERTS_FILE = DATA_DIR / "financial_alerts.csv"
STATE_FILE = DATA_DIR / "financial_agent_state.json"
USER_AGENT = "Mozilla/5.0 (compatible; FinancialEventAgent/1.0)"

# Curated editorial, financial and institutional feeds. No API key required.
RSS_FEEDS = {
    "Reuters World (Google News)": "https://news.google.com/rss/search?q=site%3Areuters.com+world+economy+markets&hl=en-US&gl=US&ceid=US:en",
    "Reuters Business (Google News)": "https://news.google.com/rss/search?q=site%3Areuters.com+business+finance+markets&hl=en-US&gl=US&ceid=US:en",
    "Financial Times (Google News)": "https://news.google.com/rss/search?q=site%3Aft.com+markets+economy+central+banks&hl=en-US&gl=US&ceid=US:en",
    "CNBC Markets": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "France 24": "https://www.france24.com/fr/rss",
    "Franceinfo Monde": "https://www.franceinfo.fr/monde.rss",
    "ECB Press": "https://www.ecb.europa.eu/rss/press.html",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "BleepingComputer Cyber": "https://www.bleepingcomputer.com/feed/",
}

TOPICS = {
    "Central banks & rates": ["federal reserve", "fed ", "ecb", "bce", "interest rate", "rate cut", "rate hike", "central bank", "monetary policy", "powell", "lagarde"],
    "Inflation, debt & economy": ["inflation", "cpi", "pce", "recession", "gdp", "sovereign debt", "government debt", "default", "unemployment", "economic growth"],
    "Geopolitics & conflict": ["war ", "war:", "military", "missile", "attack", "invasion", "conflict", "ceasefire", "sanction", "sanctions", "iran", "ukraine", "russia", "israel", "taiwan", "nato", "strait of hormuz"],
    "Energy & commodities": ["oil", "crude", "brent", "wti", "natural gas", "lng", "opec", "gold", "silver", "copper", "uranium"],
    "Financial system & liquidity": ["bank failure", "banking crisis", "liquidity", "credit crunch", "capital control", "deposit", "contagion", "systemic risk", "insolvency", "bank run"],
    "Equity markets & companies": ["stock market", "shares", "equities", "earnings", "profit warning", "guidance", "layoffs", "merger", "acquisition", "ipo"],
    "Crypto & Bitcoin": ["bitcoin", "crypto", "cryptocurrency", "stablecoin", "sec ", "etf inflow", "etf outflow"],
    "AI, electricity & infrastructure": ["artificial intelligence", "data center", "data centre", "electricity demand", "power grid", "grid investment", "semiconductor", "chipmaker"],
    "Defence & aerospace": ["defence", "defense", "military spending", "arms", "missile system", "aerospace"],
    "Cybersecurity": ["cyberattack", "cyber attack", "ransomware", "data breach", "critical infrastructure", "zero-day"],
    "Healthcare": ["healthcare", "pharma", "biotech", "drug approval", "clinical trial"],
    "FX & safe havens": ["swiss franc", "chf", "dollar", "dxy", "currency", "foreign exchange", "safe haven", "gold reserve"],
}
HIGH_IMPACT = ["emergency", "crisis", "default", "bank failure", "capital control", "invasion", "war ", "missile", "sanction", "rate decision", "rate cut", "rate hike", "intervention", "shutdown", "cyberattack", "contagion", "liquidity"]
MEDIUM_IMPACT = ["inflation", "cpi", "recession", "earnings", "profit warning", "opec", "oil supply", "strike", "election", "regulation", "etf inflow", "etf outflow"]
POSITIVE = ["ceasefire", "rate cut", "approval", "record profit", "strong earnings", "deal reached", "stimulus", "inflow", "upgrade"]
NEGATIVE = ["attack", "war", "invasion", "default", "crisis", "downgrade", "profit warning", "recession", "sanction", "outflow", "bank failure", "ransomware"]
SOURCE_TRUST = {
    "Reuters": 5, "Financial Times": 5, "Federal Reserve": 5, "ECB": 5,
    "CNBC": 4, "BBC": 4, "France 24": 4, "Franceinfo": 4,
    "BleepingComputer": 4,
}

COLUMNS = ["event_id", "published_at", "collected_at", "source", "domain", "title",
           "summary", "url", "topic", "impact_score", "priority", "sentiment",
           "confidence", "corroboration_count", "corroborating_sources",
           "why_it_matters", "verification_status"]


def _clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _date(entry) -> datetime | None:
    for key in ("published", "updated", "created"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            try:
                return pd.to_datetime(raw, utc=True).to_pydatetime()
            except Exception:
                continue
    return None


def _event_id(title: str, url: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return hashlib.sha256((normalized[:180] or url).encode("utf-8")).hexdigest()[:16]


def classify(title: str, summary: str, source: str, domain: str) -> dict:
    text = f"{title} {summary}".lower()
    matched = [(topic, sum(1 for word in words if word in text)) for topic, words in TOPICS.items()]
    matched = [(topic, count) for topic, count in matched if count]
    topic = max(matched, key=lambda x: x[1])[0] if matched else "Other / unclassified"
    high = sum(1 for word in HIGH_IMPACT if word in text)
    medium = sum(1 for word in MEDIUM_IMPACT if word in text)
    trust = next((score for brand, score in SOURCE_TRUST.items() if brand.lower() in source.lower() or brand.lower() in domain.lower()), 3)
    impact = min(100, 22 + high * 17 + medium * 8 + (12 if topic in ("Financial system & liquidity", "Geopolitics & conflict") else 0))
    sentiment_hits = sum(1 for word in POSITIVE if word in text) - sum(1 for word in NEGATIVE if word in text)
    sentiment = "Positive" if sentiment_hits > 0 else "Negative" if sentiment_hits < 0 else "Mixed / unclear"
    return {"topic": topic, "impact_score": impact, "source_trust": trust, "sentiment": sentiment}


def collect_events(days: int = 4, max_per_feed: int = 30) -> tuple[pd.DataFrame, list[str]]:
    if feedparser is None:
        return pd.DataFrame(columns=COLUMNS), ["feedparser is not installed"]
    DATA_DIR.mkdir(exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows, errors = [], []
    for source, feed_url in RSS_FEEDS.items():
        try:
            response = requests.get(feed_url, headers={"User-Agent": USER_AGENT}, timeout=(5, 12))
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            if not parsed.entries:
                errors.append(f"{source}: no RSS entries")
                continue
            for entry in parsed.entries[:max_per_feed]:
                title = _clean(entry.get("title", ""))
                link = entry.get("link", "")
                published = _date(entry)
                if not title or not link or not urlparse(link).scheme.startswith("http") or not published or published < cutoff:
                    continue
                summary = _clean(entry.get("summary", entry.get("description", "")))[:3000]
                domain = urlparse(link).netloc.lower().removeprefix("www.")
                signals = classify(title, summary, source, domain)
                rows.append({
                    "event_id": _event_id(title, link), "published_at": published.isoformat(),
                    "collected_at": datetime.now(timezone.utc).isoformat(), "source": source,
                    "domain": domain, "title": title, "summary": summary, "url": link,
                    **signals, "priority": "P1" if signals["impact_score"] >= 75 else "P2" if signals["impact_score"] >= 55 else "P3" if signals["impact_score"] >= 35 else "P4",
                    "confidence": 0, "corroboration_count": 1, "corroborating_sources": source,
                    "why_it_matters": "", "verification_status": "Source/date/link validated; corroboration pending",
                })
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}")
    if not rows:
        return pd.DataFrame(columns=COLUMNS), errors
    df = pd.DataFrame(rows).sort_values("published_at", ascending=False)
    # Group near-identical headlines. Keep the richest item and preserve distinct source names.
    df["_key"] = df["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().str[:100]
    grouped = []
    for _, group in df.groupby("_key", sort=False):
        best = group.sort_values(["source_trust", "impact_score"], ascending=False).iloc[0].to_dict()
        sources = sorted(set(group["source"].astype(str)))
        domains = sorted(set(group["domain"].astype(str)))
        count = len(domains)
        best["corroboration_count"] = count
        best["corroborating_sources"] = "; ".join(sources)
        # Confidence is an explainable heuristic, not a probability of truth.
        best["confidence"] = min(95, 35 + int(best["source_trust"]) * 8 + max(0, count - 1) * 12)
        best["verification_status"] = "Cross-source corroborated" if count >= 2 else "Single-source; not independently corroborated"
        best["impact_score"] = min(100, int(best["impact_score"]) + (8 if count >= 2 else 0))
        best["priority"] = "P1" if best["impact_score"] >= 75 else "P2" if best["impact_score"] >= 55 else "P3" if best["impact_score"] >= 35 else "P4"
        topic = best["topic"]
        best["why_it_matters"] = {
            "Financial system & liquidity": "Potential implications for funding conditions, counterparty risk and capital preservation.",
            "Geopolitics & conflict": "Could affect risk appetite, energy supply, safe-haven demand and market volatility.",
            "Central banks & rates": "May change rate expectations, bond yields, FX and equity valuations.",
            "Energy & commodities": "Could affect inflation expectations, energy producers and transport costs.",
            "AI, electricity & infrastructure": "Potential signal for electricity demand, grid investment and AI infrastructure.",
            "Defence & aerospace": "May influence defence budgets, procurement cycles and sector sentiment.",
            "Crypto & Bitcoin": "Potential impact on crypto regulation, flows, liquidity and volatility.",
            "Cybersecurity": "Could create operational, financial or critical-infrastructure risks.",
        }.get(topic, "Potential market relevance; impact needs follow-up against market prices and primary documents.")
        grouped.append(best)
    out = pd.DataFrame(grouped).drop(columns=["_key"], errors="ignore")
    return out.reindex(columns=COLUMNS + ["source_trust"]), errors


def persist_events(new_events: pd.DataFrame) -> tuple[int, int]:
    DATA_DIR.mkdir(exist_ok=True)
    if new_events.empty:
        return 0, 0
    existing = pd.read_csv(EVENTS_FILE) if EVENTS_FILE.exists() and EVENTS_FILE.stat().st_size else pd.DataFrame(columns=COLUMNS)
    combined = pd.concat([existing, new_events.reindex(columns=COLUMNS)], ignore_index=True)
    combined = combined.drop_duplicates("event_id", keep="first")
    combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
    combined = combined[combined["published_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45)]
    added = max(0, len(combined) - len(existing.drop_duplicates("event_id")))
    combined.to_csv(EVENTS_FILE, index=False)
    # Alert only P1-P3; P4 remains searchable in the event log.
    alerts = combined[combined["priority"].isin(["P1", "P2", "P3"])].copy()
    alerts.to_csv(ALERTS_FILE, index=False)
    STATE_FILE.write_text(json.dumps({"last_run_utc": datetime.now(timezone.utc).isoformat(),
        "events_total": int(len(combined)), "alerts_total": int(len(alerts)),
        "new_events": int(added)}, indent=2), encoding="utf-8")
    return added, len(alerts)


def run_agent(days: int = 4) -> dict:
    events, errors = collect_events(days=days)
    added, alert_count = persist_events(events)
    return {"collected": int(len(events)), "new_events": added, "alerts_total": alert_count,
            "errors": errors, "run_at": datetime.now(timezone.utc).isoformat()}


def render_financial_intelligence_dashboard() -> None:
    """Modern editorial-style financial intelligence dashboard."""
    st.markdown("""
    <style>
    .stApp {background: radial-gradient(ellipse at 15% 0%, #172b45 0%, #0b1422 42%, #080f1b 100%); color:#e8eef7;}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stSidebar"] {background:#0b1524;}
    .block-container {padding-top:2rem; padding-bottom:3rem; max-width:1500px;}
    h1,h2,h3,p,label,span {color:inherit;}
    .fi-hero {padding:26px 30px; border:1px solid #263b55; border-radius:20px;
      background:linear-gradient(115deg,rgba(26,49,76,.96),rgba(13,27,44,.96)); margin-bottom:18px;}
    .fi-eyebrow {font-size:11px; letter-spacing:2px; color:#86a9cf; font-weight:700; text-transform:uppercase;}
    .fi-title {font-size:32px; font-weight:800; color:#f5f8fc; margin:6px 0 4px;}
    .fi-sub {color:#9fb2c9; font-size:14px;}
    .fi-pill {display:inline-block; border:1px solid #35516f; color:#b8d3ee; background:#142a42;
      border-radius:30px; padding:5px 11px; font-size:11px; margin:12px 6px 0 0;}
    .fi-kpi {background:linear-gradient(145deg,#14263b,#101d2e); border:1px solid #263d57;
      border-radius:15px; padding:17px 18px; min-height:112px;}
    .fi-kpi-label {font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#91a8c2;}
    .fi-kpi-value {font-size:30px; font-weight:800; color:#f5f8fc; margin-top:8px;}
    .fi-kpi-note {font-size:11px; color:#8199b5; margin-top:2px;}
    .fi-section {font-size:19px; font-weight:750; color:#f0f5fb; margin:24px 0 12px;}
    .fi-alert {background:linear-gradient(145deg,#14263a,#101d2c); border:1px solid #2a4058;
      border-left:4px solid #5b9bd5; border-radius:14px; padding:16px 18px; margin:10px 0;}
    .fi-alert.p1 {border-left-color:#ff5d6c;}
    .fi-alert.p2 {border-left-color:#f2ad55;}
    .fi-alert.p3 {border-left-color:#62a8e8;}
    .fi-alert-title {font-size:15px; font-weight:700; color:#f2f6fc; line-height:1.45; margin:7px 0;}
    .fi-meta {font-size:11px; color:#91a8c2;}
    .fi-chip {font-size:10px; font-weight:800; padding:4px 8px; border-radius:20px; background:#20364d; color:#b9d7f4;}
    .fi-score {font-size:12px; color:#a9bed5; font-weight:700;}
    .fi-empty {padding:28px; text-align:center; border:1px dashed #38516b; border-radius:14px; color:#9eb1c7;}
    div[data-testid="stMetric"] {background:#14263b; border:1px solid #263d57; padding:12px 15px; border-radius:13px;}
    div[data-testid="stMetricLabel"] {color:#91a8c2;}
    div[data-testid="stMetricValue"] {color:#f5f8fc;}
    .stButton button, .stLinkButton a {border-radius:10px; font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    if EVENTS_FILE.exists() and EVENTS_FILE.stat().st_size:
        df = pd.read_csv(EVENTS_FILE)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    alerts = df[df["priority"].isin(["P1", "P2", "P3"])].copy() if not df.empty else df.copy()
    p1 = int((alerts["priority"] == "P1").sum()) if not alerts.empty else 0
    p2 = int((alerts["priority"] == "P2").sum()) if not alerts.empty else 0
    p3 = int((alerts["priority"] == "P3").sum()) if not alerts.empty else 0
    last_run = str(state.get("last_run_utc", "En attente"))
    last_run_label = last_run[:16].replace("T", " ") + " UTC" if last_run != "En attente" else last_run

    st.markdown(f"""
    <div class="fi-hero">
      <div class="fi-eyebrow">PERSONAL MARKET INTELLIGENCE · DAILY MONITOR</div>
      <div class="fi-title">Financial Intelligence</div>
      <div class="fi-sub">Veille macroéconomique, géopolitique et marchés · collecte gratuite · analyse explicable</div>
      <span class="fi-pill">● AGENT ACTIF</span><span class="fi-pill">11 FLUX RSS CONFIGURÉS</span>
      <span class="fi-pill">DERNIÈRE ANALYSE : {html.escape(last_run_label)}</span>
    </div>
    """, unsafe_allow_html=True)

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.caption("Vue exécutive · événements récents et signaux susceptibles d'influencer les marchés")
    with top_right:
        if st.button("↻ Actualiser", use_container_width=True):
            st.rerun()
    if st.button("▶ Lancer une analyse maintenant", type="primary"):
        with st.spinner("Collecte des flux, déduplication et analyse des événements…"):
            result = run_agent()
        st.success(f"Analyse terminée · {result['collected']} événements collectés · {result['new_events']} nouveaux")
        if result["errors"]:
            st.warning("Flux indisponibles : " + " · ".join(result["errors"][:6]))
        st.rerun()

    k1, k2, k3, k4 = st.columns(4)
    cards = [
        (k1, "ÉVÉNEMENTS ARCHIVÉS", f"{len(df):,}", "Historique glissant · 45 jours"),
        (k2, "ALERTES À EXAMINER", f"{len(alerts):,}", "Priorités P1 à P3"),
        (k3, "SOURCES DISTINCTES", f"{df['domain'].nunique() if not df.empty else 0:,}", "Domaines détectés"),
        (k4, "NOUVEAUX AU DERNIER RUN", f"{int(state.get('new_events', 0)):,}", "Depuis la dernière collecte"),
    ]
    for col, label, value, note in cards:
        with col:
            st.markdown(f'<div class="fi-kpi"><div class="fi-kpi-label">{label}</div><div class="fi-kpi-value">{value}</div><div class="fi-kpi-note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fi-section">Niveau de vigilance</div>', unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    for col, label, count, color, desc in [
        (v1, "P1 · CRITIQUE", p1, "#ff6675", "Événements à vérifier immédiatement"),
        (v2, "P2 · ÉLEVÉE", p2, "#f2ad55", "Impact potentiel significatif"),
        (v3, "P3 · SURVEILLANCE", p3, "#65a9e8", "Signaux à suivre et contextualiser"),
    ]:
        with col:
            st.markdown(f'<div class="fi-kpi"><div class="fi-kpi-label">{label}</div><div class="fi-kpi-value" style="color:{color}">{count}</div><div class="fi-kpi-note">{desc}</div></div>', unsafe_allow_html=True)

    if not df.empty:
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('<div class="fi-section">Répartition par thématique</div>', unsafe_allow_html=True)
            topic_counts = df["topic"].fillna("Non classé").value_counts().head(9)
            st.bar_chart(topic_counts, height=300)
        with right:
            st.markdown('<div class="fi-section">Priorités détectées</div>', unsafe_allow_html=True)
            priority_counts = alerts["priority"].value_counts().reindex(["P1", "P2", "P3"], fill_value=0) if not alerts.empty else pd.Series({"P1":0,"P2":0,"P3":0})
            st.bar_chart(priority_counts, height=300)

    st.markdown('<div class="fi-section">Centre des alertes</div>', unsafe_allow_html=True)
    if df.empty:
        st.markdown('<div class="fi-empty">Aucun événement disponible pour le moment.<br> Lance une collecte pour initialiser le flux.</div>', unsafe_allow_html=True)
        return

    f1, f2, f3 = st.columns([1.5, 1, 1])
    topic_options = ["Toutes les thématiques"] + sorted(alerts["topic"].dropna().unique().tolist())
    with f1:
        chosen_topic = st.selectbox("Thématique", topic_options, label_visibility="collapsed")
    with f2:
        chosen_priority = st.multiselect("Priorité", ["P1", "P2", "P3"], default=["P1", "P2", "P3"])
    with f3:
        period = st.selectbox("Période", ["24 dernières heures", "3 derniers jours", "7 derniers jours", "Tout l'historique"])
    view = alerts.copy()
    if chosen_topic != "Toutes les thématiques":
        view = view[view["topic"] == chosen_topic]
    if chosen_priority:
        view = view[view["priority"].isin(chosen_priority)]
    now = pd.Timestamp.now(tz="UTC")
    if period != "Tout l'historique":
        hours = {"24 dernières heures": 24, "3 derniers jours": 72, "7 derniers jours": 168}[period]
        view = view[view["published_at"] >= now - pd.Timedelta(hours=hours)]

    st.caption(f"{len(view)} alertes correspondant aux filtres · triées par date de publication")
    if view.empty:
        st.markdown('<div class="fi-empty">Aucune alerte ne correspond à ces filtres.</div>', unsafe_allow_html=True)
    else:
        for _, row in view.head(35).iterrows():
            priority = str(row.get("priority", "P3"))
            pclass = priority.lower()
            icon = {"P1":"🚨","P2":"⚠️","P3":"◈"}.get(priority, "•")
            published = row["published_at"].strftime("%d/%m/%Y · %H:%M UTC")
            title = html.escape(str(row.get("title", "Événement sans titre")))
            topic = html.escape(str(row.get("topic", "Non classé")))
            source = html.escape(str(row.get("source", "Source inconnue")))
            summary = html.escape(str(row.get("summary", "") or "Résumé non fourni."))
            why = html.escape(str(row.get("why_it_matters", "") or "Impact à approfondir."))
            status = html.escape(str(row.get("verification_status", "Vérification incomplète")))
            corroboration = html.escape(str(row.get("corroborating_sources", source)))
            score = int(row.get("impact_score", 0) or 0)
            confidence = int(row.get("confidence", 0) or 0)
            url = str(row.get("url", ""))
            st.markdown(f"""
            <div class="fi-alert {pclass}">
              <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap">
                <span class="fi-chip">{icon} {priority} · {topic}</span>
                <span class="fi-meta">{published}</span>
              </div>
              <div class="fi-alert-title">{title}</div>
              <div class="fi-meta">{source} · Impact {score}/100 · Confiance indicative {confidence}/95</div>
              <div style="height:5px;background:#26394d;border-radius:8px;margin:10px 0 12px">
                <div style="height:5px;width:{max(0,min(100,score))}%;background:#5798d0;border-radius:8px"></div>
              </div>
              <div style="font-size:12px;color:#b8c8d9;line-height:1.55">{summary[:650]}</div>
              <div style="font-size:12px;color:#9eb4cb;margin-top:9px"><b>Pourquoi c'est important :</b> {why}</div>
              <div class="fi-meta" style="margin-top:8px">{status} · Sources associées : {corroboration}</div>
            </div>
            """, unsafe_allow_html=True)
            if url.startswith("http"):
                st.markdown(f'<div style="margin-top:-7px;margin-bottom:12px"><a href="{html.escape(url, quote=True)}" target="_blank" style="color:#80b9ed;font-size:12px;text-decoration:none;font-weight:700">Consulter la source ↗</a></div>', unsafe_allow_html=True)

    with st.expander("Journal complet des événements", expanded=False):
        all_topics = ["Toutes"] + sorted(df["topic"].dropna().unique().tolist())
        selected_all_topic = st.selectbox("Filtrer le journal", all_topics, key="all_events_topic")
        all_view = df if selected_all_topic == "Toutes" else df[df["topic"] == selected_all_topic]
        st.dataframe(all_view[["published_at", "priority", "topic", "impact_score", "title", "source", "confidence", "verification_status", "url"]].head(250),
                     use_container_width=True, hide_index=True)

    st.caption("Méthode : scoring déterministe et explicable, validation technique des liens/dates et rapprochement de titres RSS. La corroboration ne constitue pas une vérification indépendante des faits. Les alertes sont des signaux de veille, pas des recommandations d'achat ou de vente.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_agent(args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
