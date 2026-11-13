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
    st.title("🧠 Financial Intelligence Agent")
    st.caption("Veille quotidienne gratuite · collecte RSS · validation des sources · corroboration · classification et scoring explicable")
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    c1, c2, c3 = st.columns(3)
    c1.metric("Dernière exécution (UTC)", str(state.get("last_run_utc", "Pas encore exécuté"))[:16].replace("T", " "))
    c2.metric("Événements archivés", int(state.get("events_total", 0)))
    c3.metric("Alertes actives", int(state.get("alerts_total", 0)))
    if st.button("🔄 Lancer une collecte maintenant", type="primary"):
        with st.spinner("Collecte et analyse des flux RSS…"):
            result = run_agent()
        st.success(f"Collectés : {result['collected']} · Nouveaux : {result['new_events']} · Alertes : {result['alerts_total']}")
        if result["errors"]:
            st.warning("Flux indisponibles : " + " · ".join(result["errors"][:8]))
        st.rerun()
    if not EVENTS_FILE.exists() or EVENTS_FILE.stat().st_size == 0:
        st.info("Aucun événement archivé. Lance une collecte ou attends la prochaine exécution GitHub Actions.")
        return
    df = pd.read_csv(EVENTS_FILE)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
    alerts = df[df["priority"].isin(["P1", "P2", "P3"])].copy()
    a, b, c, d = st.columns(4)
    a.metric("P1 — Critique", int((alerts["priority"] == "P1").sum()))
    b.metric("P2 — Élevée", int((alerts["priority"] == "P2").sum()))
    c.metric("P3 — À surveiller", int((alerts["priority"] == "P3").sum()))
    d.metric("Sources distinctes", int(df["domain"].nunique()))
    st.subheader("Alertes prioritaires")
    selected = alerts[alerts["priority"].isin(["P1", "P2", "P3"])].head(60)
    if selected.empty:
        st.success("Aucune alerte P1–P3 dans les événements collectés.")
    for _, row in selected.iterrows():
        icon = {"P1": "🚨", "P2": "⚠️", "P3": "🔎"}.get(row["priority"], "ℹ️")
        with st.expander(f"{icon} {row['priority']} · {row['topic']} · {row['title']}", expanded=row["priority"] == "P1"):
            st.write(row.get("summary", "") or "Résumé non fourni par le flux.")
            st.caption(f"Publié : {row['published_at']} · Source : {row['source']} · Confiance heuristique : {row['confidence']}/95")
            st.write(f"**Impact estimé :** {row['impact_score']}/100 · **Sentiment lexical :** {row['sentiment']}")
            st.write(f"**Pourquoi c'est important :** {row['why_it_matters']}")
            st.write(f"**Vérification :** {row['verification_status']} · Sources : {row['corroborating_sources']}")
            st.link_button("Ouvrir la source", row["url"])
    st.subheader("Filtre des événements")
    topics = ["Tous"] + sorted(df["topic"].dropna().unique().tolist())
    chosen_topic = st.selectbox("Thématique", topics)
    priorities = st.multiselect("Priorités", ["P1", "P2", "P3", "P4"], default=["P1", "P2", "P3"])
    view = df.copy()
    if chosen_topic != "Tous":
        view = view[view["topic"] == chosen_topic]
    if priorities:
        view = view[view["priority"].isin(priorities)]
    st.dataframe(view[["published_at", "priority", "topic", "impact_score", "title", "source", "confidence", "verification_status", "url"]].head(150),
                 use_container_width=True, hide_index=True)
    st.caption("Limite méthodologique : sans API LLM payante, l'analyse est fondée sur des règles transparentes. La corroboration RSS n'est pas une preuve indépendante ; lire les sources avant toute décision.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_agent(args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
