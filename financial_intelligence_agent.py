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

# User-confirmed portfolio snapshot (13 Nov 2026). Values include €4.93 cash separately.
# Performance percentages are broker-displayed position returns, not live quotes.
PORTFOLIO = [
    {"symbol":"VWCE","name":"Vanguard FTSE All-World UCITS ETF Acc","value_eur":236.61,"return_pct":3.54,"group":"Global equities","factors":["global_growth","rates","usd","equity_risk"]},
    {"symbol":"TTE","name":"TotalEnergies","value_eur":110.84,"return_pct":0.01,"group":"Energy","factors":["brent","gas","energy_supply","usd","rates"]},
    {"symbol":"GLHL","name":"Amundi S&P Global Health Care","value_eur":100.32,"return_pct":12.56,"group":"Healthcare","factors":["healthcare","rates","usd","equity_risk"]},
    {"symbol":"XOM","name":"Exxon Mobil","value_eur":98.59,"return_pct":6.65,"group":"Energy","factors":["brent","gas","energy_supply","usd","rates"]},
    {"symbol":"XDWU","name":"Xtrackers MSCI World Utilities","value_eur":95.11,"return_pct":6.72,"group":"Utilities","factors":["electricity","rates","regulation","usd"]},
    {"symbol":"AI","name":"Air Liquide","value_eur":64.84,"return_pct":14.43,"group":"Industrials","factors":["industrial_growth","energy_costs","rates","usd"]},
    {"symbol":"EWSP","name":"iShares S&P 500 Equal Weight ETF","value_eur":51.54,"return_pct":3.07,"group":"US equities","factors":["us_growth","rates","usd","equity_risk"]},
    {"symbol":"FTAI","name":"FTAI Aviation","value_eur":49.61,"return_pct":-9.12,"group":"Aerospace","factors":["aviation","credit","rates","industrial_growth"]},
    {"symbol":"FRO","name":"Frontline","value_eur":35.45,"return_pct":1.02,"group":"Shipping","factors":["tanker_rates","brent","trade_routes","energy_supply"]},
    {"symbol":"LMT","name":"Lockheed Martin","value_eur":28.31,"return_pct":13.31,"group":"Defence","factors":["defence_budgets","geopolitics","rates","usd"]},
    {"symbol":"HO","name":"Thales","value_eur":23.92,"return_pct":-1.71,"group":"Defence & technology","factors":["defence_budgets","geopolitics","cybersecurity","industrial_growth"]},
]
CASH_EUR = 4.93
PORTFOLIO_TOTAL_EUR = 899.42
PORTFOLIO_RETURN_PCT = 4.54

# Topic-to-portfolio transmission map. Directions are conditional sensitivities, not price forecasts.
IMPACT_MAP = {
    "Central banks & rates": {"positive":["TTE","XOM","FRO"],"negative":["XDWU","VWCE","GLHL","EWSP","FTAI","AI","LMT","HO"],"watch":["USD exposure in US-listed holdings"],"reason":"Une hausse des taux peut peser sur les valorisations et le financement ; l'effet devise et les anticipations de croissance peuvent compenser."},
    "Inflation, debt & economy": {"positive":["TTE","XOM"],"negative":["XDWU","VWCE","GLHL","EWSP","FTAI","AI"],"watch":["FRO","LMT","HO"],"reason":"Une inflation persistante peut retarder les baisses de taux ; l'effet dépend de sa composante énergie/salaires et de la réaction des rendements."},
    "Geopolitics & conflict": {"positive":["TTE","XOM","LMT","HO"],"negative":["VWCE","EWSP","FTAI"],"watch":["FRO","XDWU","GLHL","AI"],"reason":"Le canal dépend des sanctions, des flux énergétiques, des routes maritimes et des budgets de défense ; les valeurs défense ne montent pas mécaniquement."},
    "Energy & commodities": {"positive":["TTE","XOM"],"negative":["AI","GLHL"],"watch":["FRO","VWCE","XDWU","EWSP"],"reason":"Un choc d'offre pétrolière peut soutenir les producteurs mais renchérir les coûts et peser sur la demande ; les tarifs des tankers dépendent aussi de l'offre de navires et des routes."},
    "Financial system & liquidity": {"positive":[],"negative":["VWCE","GLHL","XDWU","EWSP","FTAI"],"watch":["TTE","XOM","FRO","AI","LMT","HO"],"reason":"Un resserrement du crédit touche d'abord les actifs risqués et les entreprises dépendantes du financement ; rechercher des signes confirmés de contagion."},
    "AI, electricity & infrastructure": {"positive":["XDWU","AI"],"negative":[],"watch":["VWCE","EWSP"],"reason":"La hausse des investissements réseau et de la demande électrique peut soutenir utilities et fournisseurs industriels, sous réserve de valorisations, contrats et financement."},
    "Defence & aerospace": {"positive":["LMT","HO"],"negative":[],"watch":["FTAI","VWCE","EWSP"],"reason":"Les annonces budgétaires ne deviennent un catalyseur qu'après budgets votés, contrats attribués et calendrier d'exécution."},
    "Cybersecurity": {"positive":["HO"],"negative":[],"watch":["VWCE","EWSP"],"reason":"Thales a une exposition aux activités cyber, mais l'effet boursier dépend de l'ampleur de l'incident et des contrats ; un incident n'est pas automatiquement positif."},
    "Healthcare": {"positive":["GLHL"],"negative":[],"watch":["VWCE","EWSP"],"reason":"Les décisions réglementaires et résultats cliniques peuvent toucher les composants santé ; l'ETF diversifié amortit les risques propres à une entreprise."},
    "Crypto & Bitcoin": {"positive":[],"negative":[],"watch":["VWCE","EWSP","XDWU"],"reason":"Aucune exposition directe au Bitcoin n'apparaît dans le portefeuille communiqué ; seul un effet indirect via liquidité et appétit pour le risque est possible."},
    "FX & safe havens": {"positive":[],"negative":[],"watch":["VWCE","TTE","XOM","GLHL","EWSP","LMT"],"reason":"Les variations EUR/USD affectent la conversion en euros des actifs et revenus en dollars ; le sens dépend de la devise de cotation, couverture et revenus."},
    "Equity markets & companies": {"positive":[],"negative":[],"watch":["VWCE","GLHL","EWSP","XDWU","TTE","XOM","FTAI","FRO","AI","LMT","HO"],"reason":"L'impact dépend de l'entreprise ou de l'indice cité, de la surprise par rapport aux attentes et de la réaction du cours."},
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


def _decision_view(df: pd.DataFrame) -> pd.DataFrame:
    """Select market-moving topics and attach a concise investor interpretation."""
    if df.empty:
        return df.copy()
    out = df.copy()
    text = (out["title"].fillna("").astype(str) + " " + out["summary"].fillna("").astype(str)).str.lower()
    noise = ["sex scandal", "celebrity", "reality tv", "dating", "royal family", "football transfer", "movie star", "pop star", "gossip", "influencer"]
    out = out[~text.loc[out.index].apply(lambda s: any(k in s for k in noise))].copy()
    lenses = {
        "Central banks & rates": ("Taux, obligations, devises et valorisation des actions", "Une inflexion des banques centrales peut modifier rapidement les anticipations de taux et les valorisations.", "Surveiller les rendements obligataires et les anticipations de taux ; ne pas agir sur un titre seul.", "SURVEILLER LES TAUX"),
        "Inflation, debt & economy": ("Inflation, croissance, crédit et pouvoir d'achat", "Une surprise macroéconomique peut déplacer les taux, les devises et l'appétit pour le risque.", "Attendre la donnée confirmée puis observer la réaction des taux et des indices.", "ATTENDRE CONFIRMATION"),
        "Geopolitics & conflict": ("Pétrole, transport, défense, devises refuges et volatilité", "Le risque devient financier s'il perturbe l'approvisionnement, les routes commerciales, les sanctions ou les flux de capitaux.", "Vérifier les conséquences concrètes sur énergie, transport maritime, or et actifs refuges.", "VÉRIFIER L'IMPACT MARCHÉ"),
        "Energy & commodities": ("Brent, gaz, inflation et producteurs d'énergie", "Un choc d'offre peut affecter les marges des producteurs, les coûts de transport et l'inflation.", "Comparer l'annonce aux prix et aux volumes réellement perturbés.", "SURVEILLER LES PRIX"),
        "Financial system & liquidity": ("Liquidité, banques, crédit et risque de contrepartie", "C'est un signal de préservation du capital si dépôts, financement ou paiements sont touchés.", "Rechercher des confirmations auprès des autorités et de plusieurs sources indépendantes.", "VÉRIFICATION PRIORITAIRE"),
        "AI, electricity & infrastructure": ("Électricité, réseaux, semi-conducteurs et investissement IA", "Peut signaler une évolution structurelle de la demande électrique et des dépenses d'infrastructure.", "Distinguer contrats financés et projets annoncés ; surveiller résultats et carnets de commandes.", "SUIVRE LES FONDAMENTAUX"),
        "Defence & aerospace": ("Budgets de défense, commandes et industriels", "L'effet dépend des budgets votés, contrats attribués et calendriers de livraison.", "Vérifier les financements et commandes réels avant toute conclusion sectorielle.", "VÉRIFIER LES CONTRATS"),
        "Crypto & Bitcoin": ("Bitcoin, liquidité globale, réglementation et flux ETF", "Les flux et conditions de liquidité peuvent amplifier les mouvements à court terme.", "Comparer l'annonce aux flux, volumes et prix ; tenir compte de la volatilité.", "SURVEILLER LES FLUX"),
        "Healthcare": ("Santé mondiale, médicaments et réglementation", "Les résultats cliniques, décisions réglementaires et dépenses de santé peuvent influencer les composants de l'ETF GLHL.", "Distinguer les annonces scientifiques des autorisations et résultats financiers ; surveiller l'impact sur GLHL.", "SUIVRE LA SANTÉ"),
        "Cybersecurity": ("Risque opérationnel et infrastructures critiques", "Un incident devient systémique s'il touche paiements, énergie, télécoms ou fournisseurs majeurs.", "Vérifier l'ampleur et les systèmes touchés avant d'extrapoler.", "ÉVALUER L'AMPLEUR"),
        "FX & safe havens": ("EUR/CHF, dollar, or et flux refuges", "Les mouvements de change influencent la valeur en euros des actifs en devises.", "Comparer le mouvement aux taux et au contexte de risque avant toute réallocation.", "SURVEILLER LES DEVISES"),
        "Equity markets & companies": ("Résultats, guidance, crédit et sentiment actions", "L'impact dépend de l'écart entre résultats publiés, attentes et perspectives.", "Comparer les chiffres aux attentes puis observer la réaction du cours.", "COMPARER AUX ATTENTES"),
    }
    out = out[out["topic"].isin(lenses)].copy()
    if out.empty:
        return out
    out["decision_channel"] = out["topic"].map(lambda t: lenses[t][0])
    out["decision_why"] = out["topic"].map(lambda t: lenses[t][1])
    out["decision_action"] = out["topic"].map(lambda t: lenses[t][2])
    out["decision_tag"] = out["topic"].map(lambda t: lenses[t][3])
    for col, default in [("impact_score", 0), ("confidence", 0), ("corroboration_count", 1)]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    age = (pd.Timestamp.now(tz="UTC") - out["published_at"]).dt.total_seconds().clip(lower=0) / 3600
    recency = (24 / (24 + age)).clip(0.15, 1.0)
    out["decision_score"] = (out["impact_score"] * .55 + out["confidence"] * .25 + out["corroboration_count"].clip(upper=3) * 7) * recency
    out = out.sort_values(["decision_score", "published_at"], ascending=False)
    return out.drop_duplicates(subset=["topic"], keep="first")


def _portfolio_impact(topic: str) -> dict:
    mapping = IMPACT_MAP.get(topic, {})
    by_symbol = {p["symbol"]: p for p in PORTFOLIO}
    return {
        "positive": [by_symbol[x] for x in mapping.get("positive", []) if x in by_symbol],
        "negative": [by_symbol[x] for x in mapping.get("negative", []) if x in by_symbol],
        "watch": [by_symbol[x] for x in mapping.get("watch", []) if x in by_symbol],
        "reason": mapping.get("reason", "Transmission au portefeuille non suffisamment établie."),
    }


def _event_polarity(row: pd.Series) -> tuple[int, str]:
    """Return event-specific transmission direction, not generic article sentiment."""
    text = f"{row.get('title','')} {row.get('summary','')}".lower()
    topic = str(row.get("topic", ""))
    hawkish = ["higher than expected", "above expectations", "hotter", "accelerat", "rate hike", "hawkish", "inflation rises", "inflation higher"]
    dovish = ["below expectations", "cooler than expected", "slowing inflation", "disinflation", "rate cut", "dovish", "inflation falls"]
    escalation = ["escalat", "attack", "invasion", "missile", "blockade", "sanction", "strike", "war"]
    deescalation = ["ceasefire", "peace deal", "de-escalat", "deal reached", "truce"]
    positive = ["beats estimates", "above expectations", "record profit", "upgrade", "approval", "strong earnings", "deal reached"]
    negative = ["misses estimates", "below expectations", "profit warning", "downgrade", "default", "bank failure", "recession"]
    if topic in ("Central banks & rates", "Inflation, debt & economy"):
        if any(k in text for k in hawkish): return -1, "Surprise restrictive : pression potentielle sur les taux et les valorisations."
        if any(k in text for k in dovish): return 1, "Surprise désinflationniste/dovish : détente potentielle des taux."
    if topic == "Geopolitics & conflict":
        if any(k in text for k in escalation): return -1, "Escalade : hausse possible de l'aversion au risque et perturbation des flux."
        if any(k in text for k in deescalation): return 1, "Désescalade : détente possible des primes de risque et des coûts logistiques."
    if topic == "Energy & commodities":
        if any(k in text for k in ["supply disruption", "production cut", "oil output cut", "pipeline attack", "supply outage"]):
            return 1, "Choc d'offre énergétique : soutien possible aux producteurs, mais pression sur les coûts."
        if any(k in text for k in ["supply increase", "output increase", "production rises", "oil glut"]):
            return -1, "Détente de l'offre : pression possible sur les producteurs, bénéfice potentiel pour les coûts."
    if any(k in text for k in negative): return -1, "Surprise défavorable explicitement identifiée dans l'événement."
    if any(k in text for k in positive): return 1, "Surprise favorable explicitement identifiée dans l'événement."
    return 0, "Direction non établie par le titre : les deux branches restent ouvertes."


def _event_exposure(topic: str) -> dict:
    mapping = IMPACT_MAP.get(topic, {})
    return {
        "positive": list(mapping.get("positive", [])),
        "negative": list(mapping.get("negative", [])),
        "watch": list(mapping.get("watch", [])),
        "reason": mapping.get("reason", "Transmission au portefeuille non suffisamment établie."),
    }


def _event_scenario(row: pd.Series, live: dict) -> dict | None:
    """Event-specific portfolio sensitivity. Explicitly heuristic; no generic volatility curve."""
    title = str(row.get("title", ""))
    text = f"{title} {row.get('summary','')}".lower()
    topic = str(row.get("topic", ""))
    # Distinct transmission fingerprints: percentage move assumptions for a moderate event.
    # Values are illustrative sensitivities, not forecasts or measured causal betas.
    profiles = [
        (["inflation", "cpi", "pce", "hotter than expected", "higher than expected"], "Inflation surprise",
         {"VWCE":-0.65,"TTE":0.20,"GLHL":-0.35,"XOM":0.15,"XDWU":-0.90,"AI":-0.45,"EWSP":-0.70,"FTAI":-0.75,"FRO":0.05,"LMT":-0.15,"HO":-0.15}),
        (["fed", "federal reserve", "rate hike", "hawkish", "higher for longer", "ecb", "interest rate"], "Taux plus élevés",
         {"VWCE":-0.80,"TTE":-0.10,"GLHL":-0.55,"XOM":-0.10,"XDWU":-1.00,"AI":-0.65,"EWSP":-0.85,"FTAI":-0.90,"FRO":-0.10,"LMT":-0.30,"HO":-0.25}),
        (["hormuz", "oil supply", "oil disruption", "opec", "brent", "crude", "pipeline"], "Choc d'offre énergétique",
         {"VWCE":-0.25,"TTE":0.90,"GLHL":-0.20,"XOM":0.85,"XDWU":-0.45,"AI":-0.35,"EWSP":-0.30,"FTAI":-0.25,"FRO":0.35,"LMT":0.05,"HO":0.05}),
        (["war", "invasion", "missile", "attack", "escalation", "sanction", "ceasefire"], "Choc géopolitique",
         {"VWCE":-0.85,"TTE":0.40,"GLHL":-0.35,"XOM":0.35,"XDWU":-0.35,"AI":-0.30,"EWSP":-0.90,"FTAI":-0.65,"FRO":-0.55,"LMT":0.35,"HO":0.35}),
        (["data center", "data centre", "electricity demand", "power grid", "grid investment", "ai infrastructure"], "Investissements IA & électricité",
         {"VWCE":0.15,"TTE":0.00,"GLHL":0.00,"XOM":0.00,"XDWU":0.95,"AI":0.75,"EWSP":0.20,"FTAI":0.00,"FRO":0.00,"LMT":0.00,"HO":0.10}),
        (["defence budget", "defense budget", "military spending", "arms contract", "missile system", "procurement"], "Commandes & budgets défense",
         {"VWCE":0.05,"TTE":0.00,"GLHL":0.00,"XOM":0.00,"XDWU":0.00,"AI":0.05,"EWSP":0.05,"FTAI":0.10,"FRO":0.00,"LMT":0.85,"HO":0.75}),
        (["cyberattack", "cyber attack", "ransomware", "data breach", "zero-day"], "Incident cyber",
         {"VWCE":-0.10,"TTE":0.00,"GLHL":0.00,"XOM":0.00,"XDWU":0.00,"AI":0.00,"EWSP":-0.10,"FTAI":0.00,"FRO":0.00,"LMT":0.15,"HO":0.65}),
        (["clinical trial", "drug approval", "healthcare", "pharma"], "Catalyseur santé",
         {"VWCE":0.05,"TTE":0.00,"GLHL":0.65,"XOM":0.00,"XDWU":0.00,"AI":0.00,"EWSP":0.05,"FTAI":0.00,"FRO":0.00,"LMT":0.00,"HO":0.00}),
        (["bank failure", "liquidity", "credit crunch", "banking crisis", "default"], "Stress de liquidité / crédit",
         {"VWCE":-1.00,"TTE":-0.45,"GLHL":-0.60,"XOM":-0.45,"XDWU":-0.70,"AI":-0.65,"EWSP":-1.10,"FTAI":-1.25,"FRO":-0.80,"LMT":-0.35,"HO":-0.35}),
    ]
    selected = next((p for p in profiles if any(k in text for k in p[0])), None)
    if not selected:
        return None
    _, event_type, impacts = selected
    # Event intensity depends on explicit surprise / disruption language and corroboration.
    intensity = 1.0
    if any(k in text for k in ["unexpected", "higher than expected", "below expectations", "emergency", "halt", "shutdown", "attack", "default", "invasion"]):
        intensity = 1.5
    if any(k in text for k in ["minor", "slight", "marginal", "in line with expectations"]):
        intensity = 0.55
    confidence = float(pd.to_numeric(pd.Series([row.get("confidence", 0)]), errors="coerce").fillna(0).iloc[0])
    corroboration = int(pd.to_numeric(pd.Series([row.get("corroboration_count", 1)]), errors="coerce").fillna(1).iloc[0])
    evidence_factor = min(1.0, 0.65 + confidence / 250 + max(0, corroboration-1)*0.08)
    holdings = live["holdings"].set_index("symbol")
    baseline = float(live["total"])
    base = {}; favorable = {}; adverse = {}; rows=[]
    for symbol, pct in impacts.items():
        if symbol not in holdings.index or abs(pct) < 0.04: continue
        value=float(holdings.loc[symbol,"value_eur"])
        # Directional impact is specific to the event and the security's economic exposure.
        central=pct*intensity*evidence_factor
        # Best/worst are alternative event outcomes, not the same volatility band for every headline.
        favorable_pct=central*0.45
        adverse_pct=central*1.65
        base[symbol]=value
        favorable[symbol]=value*(1+favorable_pct/100)
        adverse[symbol]=value*(1+adverse_pct/100)
        rows.append({"symbol":symbol,"value_eur":value,"impact_pct":central,
                     "favorable_eur":value*favorable_pct/100,"adverse_eur":value*adverse_pct/100})
    if not rows: return None
    current=float(sum(base.values()))
    favorable_delta=sum(favorable[k]-base[k] for k in base)
    adverse_delta=sum(adverse[k]-base[k] for k in base)
    return {"event_type":event_type,"rows":rows,"exposed_value":current,
            "favorable_delta":favorable_delta,"adverse_delta":adverse_delta,
            "favorable_pct":(favorable_delta/current*100 if current else 0),
            "adverse_pct":(adverse_delta/current*100 if current else 0),
            "baseline":baseline,"affected_count":len(rows),"intensity":intensity,
            "evidence_factor":evidence_factor}


YF_SYMBOLS={"VWCE":"VWCE.DE","TTE":"TTE.PA","GLHL":"WELS.PA","XOM":"XOM","XDWU":"XDWU.DE","AI":"AI.PA","EWSP":"EWSP.L","FTAI":"FTAI","FRO":"FRO","LMT":"LMT","HO":"HO.PA"}
QUOTE_CURRENCY={"XOM":"USD","FTAI":"USD","FRO":"USD","LMT":"USD","EWSP":"GBP"}
QUOTE_BASELINE_FILE=DATA_DIR/"portfolio_quote_baseline_v2.json"


def _portfolio_live_snapshot():
    try:
        import yfinance as yf
        symbols=list(YF_SYMBOLS.values())+["EURUSD=X","EURGBP=X"]
        raw=yf.download(symbols,period="6mo",interval="1d",auto_adjust=True,group_by="ticker",progress=False,threads=True)
        if raw.empty: return None,"Yahoo Finance n'a retourné aucun cours."
        close={}
        for sym in symbols:
            try:
                series=raw[sym]["Close"].dropna()
                if not series.empty: close[sym]=series
            except Exception: pass
        baseline={}
        if QUOTE_BASELINE_FILE.exists():
            try: baseline=json.loads(QUOTE_BASELINE_FILE.read_text(encoding="utf-8"))
            except Exception: baseline={}
        latest={}; histories={}
        fx_usd=close.get("EURUSD=X")
        fx_gbp=close.get("EURGBP=X")
        for p in PORTFOLIO:
            sym=p["symbol"]; yf_sym=YF_SYMBOLS[sym]
            if yf_sym not in close: continue
            s=close[yf_sym].dropna()
            cur=QUOTE_CURRENCY.get(sym,"EUR")
            if cur=="USD":
                if fx_usd is None: continue
                s=s/fx_usd.reindex(s.index).ffill()
            elif cur=="GBP":
                if fx_gbp is None: continue
                s=s/fx_gbp.reindex(s.index).ffill()
            s=s.dropna()
            if s.empty: continue
            histories[sym]=s
            latest[sym]=float(s.iloc[-1])
            if sym not in baseline:
                baseline[sym]={"price_eur":latest[sym],"anchor_value_eur":float(p["value_eur"])}
        QUOTE_BASELINE_FILE.write_text(json.dumps(baseline,indent=2),encoding="utf-8")
        # Broker-confirmed snapshot is the anchor. Live prices only move each line from this anchor.
        anchor_positions=sum(float(p["value_eur"]) for p in PORTFOLIO)
        scale=(PORTFOLIO_TOTAL_EUR-CASH_EUR)/anchor_positions if anchor_positions else 1.0
        rows=[]
        for p in PORTFOLIO:
            sym=p["symbol"]
            anchor=float(p["value_eur"])*scale
            if sym in latest and sym in baseline:
                b=baseline[sym]
                value=anchor*latest[sym]/float(b["price_eur"])
                quote_change=(latest[sym]/float(b["price_eur"])-1)*100
            else:
                value=anchor
                quote_change=0.0
            rows.append({"symbol":sym,"value_eur":value,"change_pct":quote_change,"anchor_value_eur":anchor,"quote_available":sym in latest})
        holdings=pd.DataFrame(rows)
        total=float(holdings.value_eur.sum()+CASH_EUR)
        # Broker performance is the starting point; daily quote changes update the return.
        anchor_cost=PORTFOLIO_TOTAL_EUR/(1+PORTFOLIO_RETURN_PCT/100)
        perf=(total/anchor_cost-1)*100
        updated=max(s.index.max() for s in histories.values()) if histories else pd.Timestamp.now(tz="UTC")
        return {"holdings":holdings,"total":total,"initial":anchor_cost,"perf":perf,"histories":histories,"updated":updated},None
    except Exception as exc:
        return None,f"Erreur cours : {type(exc).__name__}: {exc}"


def render_financial_intelligence_dashboard() -> None:
    st.markdown("## Financial Intelligence · Event Impact")
    st.caption("Deux classements côte à côte : événements à impact net favorable à gauche, événements à risque net défavorable à droite.")
    with st.spinner("Actualisation des cours…"):
        live,error=_portfolio_live_snapshot()
    if live is None:
        st.error(error or "Données de marché indisponibles"); return
    if error: st.warning(error)
    a,b,c=st.columns(3)
    a.metric("Portefeuille",f"{live['total']:,.2f} €",f"{live['perf']:+.2f}%")
    b.metric("Actifs avec cours",f"{int(live['holdings']['quote_available'].sum())}/{len(PORTFOLIO)}")
    c.metric("Cours jusqu'au",live["updated"].strftime("%d/%m/%Y"))
    st.caption("Instantané communiqué : 899,42 € et +4,54 %. Les variations ultérieures sont estimées à partir des cours gratuits disponibles ; elles ne remplacent pas le PRU du courtier.")
    st.markdown("### Événements classés selon leur impact estimé sur ton portefeuille")
    if not EVENTS_FILE.exists() or not EVENTS_FILE.stat().st_size:
        st.info("Aucun événement collecté. Lance une collecte.")
        return
    events=pd.read_csv(EVENTS_FILE)
    events["published_at"]=pd.to_datetime(events["published_at"],errors="coerce",utc=True)
    decisions=_decision_view(events.dropna(subset=["published_at"]))
    if decisions.empty:
        st.info("Aucun événement suffisamment pertinent.")
        return
    topic_colors = {
        "Central banks & rates":"#4776C8", "Inflation, debt & economy":"#D97732",
        "Geopolitics & conflict":"#C44E62", "Energy & commodities":"#D49A28",
        "Financial system & liquidity":"#7656B5", "AI, electricity & infrastructure":"#168C87",
        "Defence & aerospace":"#536A9C", "Cybersecurity":"#27856A",
        "Healthcare":"#B65B91", "FX & safe havens":"#4C91B8",
        "Equity markets & companies":"#68727D",
    }
    names={p["symbol"]:p["name"] for p in PORTFOLIO}
    positive_events=[]
    risk_events=[]
    for _,row in decisions.iterrows():
        scenario=_event_scenario(row,live)
        if scenario is None: continue
        impact=pd.DataFrame(scenario["rows"])
        impact["name"]=impact["symbol"].map(names).fillna(impact["symbol"])
        impact["delta_eur"]=impact["value_eur"]*impact["impact_pct"]/100
        gains=impact[impact["delta_eur"]>0.005].sort_values("delta_eur",ascending=False)
        losses=impact[impact["delta_eur"]<-0.005].sort_values("delta_eur")
        gain_total=float(gains["delta_eur"].sum()) if not gains.empty else 0.0
        loss_total=float(losses["delta_eur"].sum()) if not losses.empty else 0.0
        net=gain_total-abs(loss_total)
        confidence=float(pd.to_numeric(pd.Series([row.get("confidence",0)]),errors="coerce").fillna(0).iloc[0])
        corroboration=int(pd.to_numeric(pd.Series([row.get("corroboration_count",1)]),errors="coerce").fillna(1).iloc[0])
        # Ranking combines the net portfolio impact with source confidence and corroboration.
        evidence=max(0.25,min(1.0,0.55+confidence/250+max(0,corroboration-1)*0.08))
        item={"row":row,"scenario":scenario,"gains":gains,"losses":losses,
              "gain_total":gain_total,"loss_total":loss_total,"net":net,
              "rank":abs(net)*evidence}
        if net > 0.005:
            positive_events.append(item)
        elif net < -0.005:
            risk_events.append(item)
    positive_events.sort(key=lambda x:x["rank"],reverse=True)
    risk_events.sort(key=lambda x:x["rank"],reverse=True)

    def render_event(item, side, card_index):
        row=item["row"]; scenario=item["scenario"]
        color=topic_colors.get(str(row.get("topic","")),"#68727D")
        pastel_topics = {
            "Central banks & rates":"#EEF3FC", "Inflation, debt & economy":"#FFF4E8",
            "Geopolitics & conflict":"#FCEEF0", "Energy & commodities":"#FFF8E8",
            "Financial system & liquidity":"#F3EFFB", "AI, electricity & infrastructure":"#EAF7F5",
            "Defence & aerospace":"#EEF1F9", "Cybersecurity":"#EAF6F0",
            "Healthcare":"#F9EEF5", "FX & safe havens":"#EDF5FA",
            "Equity markets & companies":"#F0F2F4",
        }
        # The side tint reinforces the meaning while topic colors keep events visually distinct.
        bg=pastel_topics.get(str(row.get("topic","")), "#F4F5F7")
        if side=="positive":
            bg={"#FCEEF0":"#F0F8F1","#FFF4E8":"#F3F9EF"}.get(bg,bg)
            accent="#23845B"
        else:
            bg={"#EEF3FC":"#FFF0EF","#EAF7F5":"#FFF1F0"}.get(bg,bg)
            accent="#C43D4D"
        card_key=f"event_card_{side}_{card_index}"
        st.markdown(
            f"<style>.st-key-{card_key}{{background:{bg};border:1px solid {color}55;"
            f"border-left:5px solid {accent};border-radius:12px;padding:14px 16px 10px 16px;}}</style>",
            unsafe_allow_html=True,
        )
        with st.container(key=card_key):
            st.markdown(f"<div style='height:4px;background:{color};border-radius:5px;margin-bottom:8px'></div>",unsafe_allow_html=True)
            st.markdown(f"**{row.get('title','')}**")
            st.caption(f"{scenario['event_type']} · {row.get('source','')} · {pd.to_datetime(row['published_at']).strftime('%d/%m %H:%M UTC')}")
            k1,k2,k3=st.columns(3)
            k1.metric("Impact net",f"{item['net']:+.2f} €")
            k2.metric("Potentiel +",f"+{item['gain_total']:.2f} €")
            k3.metric("Risque −",f"−{item['loss_total']:.2f} €")
            st.caption(f"{scenario['affected_count']} supports exposés · {scenario['exposed_value']:,.0f} € concernés")
            total_swing=item["gain_total"]+item["loss_total"]
            if total_swing>0:
                gain_share=item["gain_total"]/total_swing
                risk_share=item["loss_total"]/total_swing
                st.markdown(
                    f"<div style='font-size:0.78rem;font-weight:600;margin:5px 0 4px'>Répartition gain / risque</div>"
                    f"<div style='display:flex;height:12px;border-radius:8px;overflow:hidden;background:#E5E7EB'>"
                    f"<div style='width:{gain_share*100:.1f}%;background:#26966A'></div>"
                    f"<div style='width:{risk_share*100:.1f}%;background:#D94B58'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:0.72rem;margin-top:3px'>"
                    f"<span style='color:#18764F'>Gain {item['gain_total']:.2f} €</span>"
                    f"<span style='color:#B83243'>Risque {item['loss_total']:.2f} €</span></div>",
                    unsafe_allow_html=True,
                )
            if side=="positive":
                st.markdown("**Principaux moteurs positifs**")
                assets=item["gains"].head(4)
                bar_color="#26966A"
                if assets.empty:
                    st.caption("Aucun support positif identifié.")
                else:
                    max_delta=max(float(x) for x in assets["delta_eur"].abs()) or 1.0
                    for _,asset in assets.iterrows():
                        name_col,value_col=st.columns([2.5,1])
                        name_col.markdown(f"**{asset['name']}**")
                        value_col.markdown(f"<div style='text-align:right;color:{bar_color};font-weight:700'>+{asset['delta_eur']:.2f} €</div>",unsafe_allow_html=True)
                        width=max(2.0,min(100.0,abs(float(asset["delta_eur"]))/max_delta*100))
                        st.markdown(f"<div style='height:7px;background:#DCE8E0;border-radius:8px;margin:-5px 0 7px'><div style='height:7px;width:{width:.1f}%;background:{bar_color};border-radius:8px'></div></div>",unsafe_allow_html=True)
                if item["loss_total"]>0.005:
                    st.caption(f"Contre-effet négatif : −{item['loss_total']:.2f} €")
            else:
                st.markdown("**Principaux facteurs de perte**")
                assets=item["losses"].head(4)
                bar_color="#D94B58"
                if assets.empty:
                    st.caption("Aucun support négatif identifié.")
                else:
                    max_delta=max(float(x) for x in assets["delta_eur"].abs()) or 1.0
                    for _,asset in assets.iterrows():
                        name_col,value_col=st.columns([2.5,1])
                        name_col.markdown(f"**{asset['name']}**")
                        value_col.markdown(f"<div style='text-align:right;color:{bar_color};font-weight:700'>{asset['delta_eur']:.2f} €</div>",unsafe_allow_html=True)
                        width=max(2.0,min(100.0,abs(float(asset["delta_eur"]))/max_delta*100))
                        st.markdown(f"<div style='height:7px;background:#F1DCDD;border-radius:8px;margin:-5px 0 7px'><div style='height:7px;width:{width:.1f}%;background:{bar_color};border-radius:8px'></div></div>",unsafe_allow_html=True)
                if item["gain_total"]>0.005:
                    st.caption(f"Compensation positive : +{item['gain_total']:.2f} €")
            st.caption("Barres vertes = contributions positives · barres rouges = contributions négatives. Longueur relative aux supports affichés, pas une probabilité.")
            url=str(row.get("url","") or "")
            if url.startswith("http"):
                st.markdown(f"[Source de l'événement]({url})")

    left,right=st.columns(2,gap="large")
    with left:
        st.markdown("### 🟢 Événements favorables")
        st.caption("Impact positif net estimé le plus élevé sur le portefeuille.")
        if not positive_events:
            st.info("Aucun événement avec un avantage net identifiable pour le moment.")
        else:
            for idx,item in enumerate(positive_events[:6]):
                render_event(item,"positive",idx)
    with right:
        st.markdown("### 🔴 Événements à risque")
        st.caption("Impact négatif net estimé le plus élevé sur le portefeuille.")
        if not risk_events:
            st.info("Aucun événement avec un risque net identifiable pour le moment.")
        else:
            for idx,item in enumerate(risk_events[:6]):
                render_event(item,"risk",idx)
    with st.expander("Méthode et limites"):
        st.write("Les événements sont répartis dans la colonne gauche si la somme des impacts positifs estimés dépasse la somme des impacts négatifs ; ils sont placés à droite dans le cas inverse. Chaque colonne est triée par ampleur de l'impact net, pondérée par la confiance dans les sources et leur corroboration.")
        st.write("Ce classement ne constitue pas une probabilité que l'événement se produise ni une probabilité statistique de gain ou de perte. Les sensibilités restent heuristiques et conditionnelles ; elles ne sont pas des prévisions de cours.")
        st.write("Le suivi des cours est ancré à l'instantané communiqué. Sans quantités exactes, les variations sont estimées à partir des valeurs de départ. Certains tickers ou conversions peuvent être indisponibles.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_agent(args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
