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
    # New user-confirmed ASML purchase: $100 converted at ECB reference rate 1 EUR = 1.0963 USD (24 Nov 2026).
    {"symbol":"ASML","name":"ASML Holding NV","value_eur":91.22,"return_pct":0.0,"group":"Semiconductors","factors":["ai_investment","semiconductor_demand","rates","usd","equity_risk"]},
]
CASH_EUR = 4.93
PORTFOLIO_TOTAL_EUR = 990.64
PORTFOLIO_RETURN_PCT = 4.10

# Planned purchases are shown separately and are not included in portfolio valuation until executed.
PLANNED_BUYS = [
    {"asset":"Or","amount_usd":100.0,"amount_eur":91.22,"status":"Prévu · non exécuté",
     "note":"À intégrer au portefeuille après confirmation de l'achat et du support utilisé."},
]

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
    """Merge fresh events with the existing store, even when a feed returns zero rows."""
    DATA_DIR.mkdir(exist_ok=True)
    existing = pd.read_csv(EVENTS_FILE) if EVENTS_FILE.exists() and EVENTS_FILE.stat().st_size else pd.DataFrame(columns=COLUMNS)
    incoming = new_events.reindex(columns=COLUMNS) if new_events is not None else pd.DataFrame(columns=COLUMNS)
    combined = pd.concat([existing.reindex(columns=COLUMNS), incoming], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates("event_id", keep="first")
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)
        combined = combined.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
        combined = combined[combined["published_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45)]
    added = max(0, len(combined) - len(existing.drop_duplicates("event_id")))
    combined.to_csv(EVENTS_FILE, index=False)
    # Alert only P1-P3; P4 remains searchable in the event log.
    alerts = combined[combined["priority"].isin(["P1", "P2", "P3"])].copy() if not combined.empty else pd.DataFrame(columns=COLUMNS)
    alerts.to_csv(ALERTS_FILE, index=False)
    STATE_FILE.write_text(json.dumps({"last_run_utc": datetime.now(timezone.utc).isoformat(),
        "events_total": int(len(combined)), "alerts_total": int(len(alerts)),
        "new_events": int(added)}, indent=2), encoding="utf-8")
    return added, len(alerts)


def run_agent(days: int = 7) -> dict:
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
            "favorable_pct":(favorable_delta/current*100 if current else 0),            "adverse_pct":(adverse_delta/current*100 if current else 0),
            "baseline":baseline,"affected_count":len(rows),"intensity":intensity,
            "evidence_factor":evidence_factor}


YF_SYMBOLS={"VWCE":"VWCE.DE","TTE":"TTE.PA","GLHL":"WELS.PA","XOM":"XOM","XDWU":"XDWU.DE","AI":"AI.PA","EWSP":"EWSP.L","FTAI":"FTAI","FRO":"FRO","LMT":"LMT","HO":"HO.PA","ASML":"ASML.AS"}
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



# -------------------------------------------------------------------
# PORTFOLIO ENRICHMENT — TOP 5 OPPORTUNITIES
# -------------------------------------------------------------------
# A diversified watchlist; avoid duplicating the portfolio's largest sector bets.
ENRICHMENT_CANDIDATES = [
    {"symbol": "MSFT", "name": "Microsoft", "type": "Action", "theme": "Logiciels · cloud · IA", "color": "#4776C8", "reason": "FY2026 : Azure progresse de 41 % et Microsoft Cloud de 27 %, tandis que les engagements commerciaux restant à exécuter atteignent 678 Md$ (+84 %), donnant une forte visibilité sur les revenus futurs."},
    {"symbol": "GOOGL", "name": "Alphabet", "type": "Action", "theme": "Publicité · cloud · IA", "color": "#168C87", "reason": "T3 2026 : Google Cloud croît de 103 % à 30,7 Md$ et la recherche de 12 %, montrant qu'Alphabet monétise l'IA tout en conservant un moteur publicitaire majeur."},
    {"symbol": "SAP", "name": "SAP", "type": "Action", "theme": "Logiciels d'entreprise", "color": "#D49A28", "reason": "S1 2026 : le carnet de commandes cloud courant atteint 22,9 Md€ (+27 %) et le chiffre d'affaires cloud progresse de 22 %, apportant visibilité et revenus récurrents."},
    {"symbol": "ASML", "name": "ASML", "type": "Action", "theme": "Semi-conducteurs", "color": "#7656B5", "reason": "T3 2026 : ventes record de 11,7 Md€ et marge brute de 57,5 % ; ASML indique que sa capacité prévue pour 2027–2028 ne suffira pas à répondre à la demande de ses clients."},
    {"symbol": "V", "name": "Visa", "type": "Action", "theme": "Paiements", "color": "#C44E62", "reason": "Exercice 2026 : chiffre d'affaires +15 %, volume de paiements +9 % et transactions transfrontalières +13 %, confirmant la croissance du réseau de paiement mondial."},
    {"symbol": "DBK.DE", "name": "Deutsche Bank", "type": "Action", "theme": "Banques / finance", "color": "#4776C8", "reason": "9 premiers mois 2026 : bénéfice après impôt record de 5,9 Md€ (+7 %) et ratio coûts/revenus ramené à 61,4 % ; la hausse des provisions de crédit reste un risque à surveiller."},
    {"symbol": "ALV.DE", "name": "Allianz", "type": "Action", "theme": "Assurance / finance", "color": "#168C87", "reason": "9 premiers mois 2026 : résultat opérationnel record de 14,1 Md€ (+7,8 %) et ratio de solvabilité Solvency II de 219 %, combinant progression des bénéfices et coussin de capital élevé."},
    {"symbol": "CARL-B.CO", "name": "Carlsberg", "type": "Action", "theme": "Biens de consommation", "color": "#D49A28", "reason": "T3 2026 : chiffre d'affaires organique +3 %, catégories de croissance en volume +5 % et boissons sans alcool +11 % ; la faiblesse de l'Asie (-6,7 % en volume) reste un risque."},
    {"symbol": "EUNI.L", "name": "iShares MSCI Emerging Markets Small Cap UCITS ETF", "type": "ETF UCITS", "theme": "Marchés émergents", "color": "#7656B5", "reason": "L'ETF répartit l'exposition sur 1 783 petites capitalisations émergentes ; ses dix premières lignes pèsent 4,77 %, mais les frais de 0,74 % et le poids de Taïwan (26,46 %) comptent."},
    {"symbol": "XDJP.DE", "name": "Xtrackers Nikkei 225 UCITS ETF", "type": "ETF UCITS", "theme": "Japon", "color": "#C44E62", "reason": "L'ETF donne accès à 225 grandes sociétés japonaises, notamment dans les semi-conducteurs et l'automatisation ; attention, ses dix premières lignes représentent 47,09 % du portefeuille."},
]
SP500_REFERENCE_RETURN = 17.0
SP500_REFERENCE_LABEL = "Goldman Sachs · prévision S&P 500 d'environ +17 % jusqu'à fin 2027 (17/11/2026)"


def _as_number(value):
    """Convert Yahoo Finance values to finite floats, otherwise None."""
    try:
        number = pd.to_numeric(value, errors="coerce")
        return float(number) if pd.notna(number) and pd.notna(float(number)) else None
    except Exception:
        return None


def _first_valid(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = _as_number(mapping.get(key))
        if value is not None and value > 0:
            return value
    return None


@st.cache_data(ttl=21600, show_spinner=False)
def _load_enrichment_data(symbol: str) -> dict:
    """Retrieve each Yahoo Finance data family independently; one failure must not erase the others."""
    result = {
        "price": None, "target": None, "expected_return": None, "opinions": None,
        "strong_buy": None, "buy": None, "hold": None, "sell": None, "strong_sell": None,
        "ratings_total": None, "buy_share": None, "holder_count": None,
        "institutional_pct": None, "consensus_label": None, "analyst_source": None, "price_source": None, "errors": [],
    }
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
    except Exception as exc:
        result["errors"].append(f"Initialisation Yahoo Finance : {type(exc).__name__}")
        return result

    # 1) Quote and summary information. Try info, fast_info and recent history separately.
    info = {}
    try:
        info = ticker.get_info() or {}
    except Exception as exc:
        result["errors"].append(f"Fiche société : {type(exc).__name__}")
        try:
            info = ticker.info or {}
        except Exception as fallback_exc:
            result["errors"].append(f"Fiche alternative : {type(fallback_exc).__name__}")
    result["price"] = _first_valid(info, ("currentPrice", "regularMarketPrice", "navPrice"))
    result["target"] = _first_valid(info, ("targetMeanPrice",))
    result["opinions"] = _first_valid(info, ("numberOfAnalystOpinions",))
    result["institutional_pct"] = _as_number(info.get("heldPercentInstitutions"))
    if result["institutional_pct"] is not None:
        result["institutional_pct"] = max(0.0, min(1.0, result["institutional_pct"]))

    if result["price"] is None:
        try:
            fast_info = ticker.fast_info
            result["price"] = _first_valid(dict(fast_info), ("last_price", "regular_market_price"))
        except Exception as exc:
            result["errors"].append(f"Cours fast_info : {type(exc).__name__}")
    if result["price"] is None:
        try:
            hist = ticker.history(period="5d", auto_adjust=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
                result["price"] = _as_number(hist["Close"].dropna().iloc[-1])
        except Exception as exc:
            result["errors"].append(f"Historique des cours : {type(exc).__name__}")

    # 2) Analyst price target: info first, then the dedicated price-target endpoint.
    try:
        targets = ticker.analyst_price_targets
        if isinstance(targets, dict):
            result["target"] = result["target"] or _first_valid(targets, ("mean", "median", "current"))
        elif isinstance(targets, pd.Series):
            result["target"] = result["target"] or _first_valid(targets.to_dict(), ("mean", "median", "current"))
    except Exception as exc:
        result["errors"].append(f"Objectifs analystes : {type(exc).__name__}")

    # 3) Consensus: try recommendations_summary, then recommendations/get_recommendations.
    rec_frame = None
    try:
        rec_frame = ticker.recommendations_summary
    except Exception as exc:
        result["errors"].append(f"Consensus résumé : {type(exc).__name__}")
    if not isinstance(rec_frame, pd.DataFrame) or rec_frame.empty:
        try:
            rec_frame = ticker.recommendations
        except Exception as exc:
            result["errors"].append(f"Consensus détaillé : {type(exc).__name__}")
    if (not isinstance(rec_frame, pd.DataFrame) or rec_frame.empty) and hasattr(ticker, "get_recommendations"):
        try:
            rec_frame = ticker.get_recommendations()
        except Exception as exc:
            result["errors"].append(f"Consensus alternatif : {type(exc).__name__}")
    if isinstance(rec_frame, pd.DataFrame) and not rec_frame.empty:
        normalized = {str(c).lower().replace("_", "").replace(" ", ""): c for c in rec_frame.columns}
        row = rec_frame.iloc[0]
        counts = {}
        for key in ("strongbuy", "buy", "hold", "sell", "strongsell"):
            col = normalized.get(key)
            counts[key] = max(0, int(_as_number(row.get(col)) or 0)) if col is not None else None
        # Recommendations history may use a single "To Grade" column instead of counts.
        if all(v is None for v in counts.values()) and "tograde" in normalized:
            grades = rec_frame[normalized["tograde"]].astype(str).str.lower()
            counts = {
                "strongbuy": int(grades.str.contains("strong buy", regex=False).sum()),
                "buy": int((grades.eq("buy") | grades.eq("outperform") | grades.eq("overweight")).sum()),
                "hold": int((grades.eq("hold") | grades.eq("neutral") | grades.eq("equal-weight")).sum()),
                "sell": int((grades.eq("sell") | grades.eq("underperform") | grades.eq("underweight")).sum()),
                "strongsell": int(grades.str.contains("strong sell", regex=False).sum()),
            }
        if any(v is not None for v in counts.values()):
            for key, value in counts.items():
                result[{"strongbuy":"strong_buy","strongsell":"strong_sell"}.get(key,key)] = value
            vals = [v for v in counts.values() if v is not None]
            result["ratings_total"] = sum(vals)
            result["opinions"] = result["opinions"] or result["ratings_total"]
            if result["ratings_total"] > 0:
                result["buy_share"] = ((counts["strongbuy"] or 0) + (counts["buy"] or 0)) / result["ratings_total"]

    # 3b) Yahoo public-page fallback. The old regex was over-escaped.
    if result["price"] is None or result["target"] is None or result["buy_share"] is None:
        try:
            page = requests.get(
                f"https://finance.yahoo.com/quote/{symbol}/",
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=(5, 12),
            )
            page.raise_for_status()
            body = page.text
            def _html_raw(key):
                patterns = (
                    rf'"{re.escape(key)}"\s*:\s*\{{\s*"raw"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)',
                    rf'"{re.escape(key)}"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)',
                )
                for pattern in patterns:
                    match = re.search(pattern, body)
                    if match:
                        return _as_number(match.group(1))
                return None
            result["price"] = result["price"] or _html_raw("regularMarketPrice") or _html_raw("currentPrice")
            result["target"] = result["target"] or _html_raw("targetMeanPrice")
            result["opinions"] = result["opinions"] or _html_raw("numberOfAnalystOpinions")
            rec_counts = {k: _html_raw(k) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
            if any(v is not None for v in rec_counts.values()):
                total = sum(int(v or 0) for v in rec_counts.values())
                if total:
                    result["strong_buy"] = int(rec_counts["strongBuy"] or 0)
                    result["buy"] = int(rec_counts["buy"] or 0)
                    result["hold"] = int(rec_counts["hold"] or 0)
                    result["sell"] = int(rec_counts["sell"] or 0)
                    result["strong_sell"] = int(rec_counts["strongSell"] or 0)
                    result["ratings_total"] = total
                    result["opinions"] = result["opinions"] or total
                    result["buy_share"] = (result["strong_buy"] + result["buy"]) / total
        except Exception as exc:
            result["errors"].append(f"Page Yahoo Finance : {type(exc).__name__}")

    # 3c) StockAnalysis fallback for US-listed shares (S&P Global analyst data).
    if (result["price"] is None or result["target"] is None or
            result["buy_share"] is None) and "." not in symbol and symbol.isalpha():
        try:
            page = requests.get(
                f"https://stockanalysis.com/stocks/{symbol.lower()}/forecast/",
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=(5, 12),
            )
            page.raise_for_status()
            visible = html.unescape(re.sub(r"<[^>]+>", " ", page.text))
            visible = re.sub(r"\s+", " ", visible)
            if result["price"] is None:
                price_match = re.search(
                    r"(?:Real-Time|Delayed) Price\s*[·|]\s*[A-Z]{3}\s+([0-9,]+(?:\.[0-9]+)?)",
                    visible, re.I,
                )
                if price_match:
                    result["price"] = _as_number(price_match.group(1).replace(",", ""))
            analyst_match = re.search(
                r"According to\s+(\d+)\s+analysts.*?(?:average price target (?:of|is)|12-month stock price target is)\s*[$€£]\s*([0-9,]+(?:\.[0-9]+)?)",
                visible, re.I,
            )
            if analyst_match:
                result["opinions"] = result["opinions"] or _as_number(analyst_match.group(1))
                result["target"] = result["target"] or _as_number(analyst_match.group(2).replace(",", ""))
            consensus_match = re.search(
                r'(?:consensus rating (?:is|of)|Analyst Consensus:)\s*["\']?(Strong Buy|Buy|Hold|Neutral|Sell|Strong Sell)',
                visible, re.I,
            )
            if consensus_match:
                result["consensus_label"] = consensus_match.group(1).title()
                result["analyst_source"] = "StockAnalysis / S&P Global"
            if result["price"] is not None:
                result["price_source"] = result.get("price_source") or "StockAnalysis"
        except Exception as exc:
            result["errors"].append(f"StockAnalysis : {type(exc).__name__}")

    if result["price"] and result["target"]:
        result["expected_return"] = (result["target"] / result["price"] - 1) * 100

    # 4) Institutional data is independent from the analyst endpoints.
    try:
        holders = ticker.get_institutional_holders()
        if isinstance(holders, pd.DataFrame):
            result["holder_count"] = int(len(holders))
    except Exception as exc:
        result["errors"].append(f"Détention institutionnelle : {type(exc).__name__}")
        try:
            holders = ticker.institutional_holders
            if isinstance(holders, pd.DataFrame):
                result["holder_count"] = int(len(holders))
        except Exception as fallback_exc:
            result["errors"].append(f"Détention alternative : {type(fallback_exc).__name__}")

    return result


def _score_enrichment_candidate(candidate: dict, data: dict) -> dict:
    """Score /100 from analyst-implied upside and consensus; ownership is optional."""
    result = {**candidate, **data, "score": None, "return_score": None,
              "analyst_score": None, "institutional_score": None, "excess_return": None}
    expected_return = data.get("expected_return")
    opinions = _as_number(data.get("opinions") or data.get("ratings_total"))
    consensus_label = str(data.get("consensus_label") or "").lower()
    label_score = {
        "strong buy": 1.0, "buy": 0.8, "hold": 0.5, "neutral": 0.5,
        "sell": 0.2, "strong sell": 0.0,
    }.get(consensus_label)
    buy_share = data.get("buy_share")
    if buy_share is None:
        buy_share = label_score
    missing = []
    if expected_return is None:
        missing.append("objectif analyste ou cours manquant")
    if buy_share is None or not opinions or opinions <= 0:
        missing.append("consensus analystes ou nombre d’analystes manquant")
    if missing:
        result["status"] = " · ".join(missing)
        return result
    return_score = max(0.0, min(60.0, (float(expected_return) + 10.0) / 40.0 * 60.0))
    analyst_reliability = min(1.0, float(opinions) / 10.0)
    analyst_score = 40.0 * float(buy_share) * analyst_reliability
    result.update({
        "return_score": return_score, "analyst_score": analyst_score,
        "institutional_score": None,
        "score": round(return_score + analyst_score, 1),
        "status": "Score calculable · détention institutionnelle informative uniquement",
    })
    return result

def _gadget_metric(label: str, value: str, color: str, detail: str = "") -> str:
    detail_html = f"<div style='font-size:11px;color:#64748b;margin-top:3px'>{detail}</div>" if detail else ""
    return (
        f"<div style='background:#ffffff;border:1px solid #e5e7eb;border-top:3px solid {color};"
        f"border-radius:10px;padding:9px 10px;min-height:72px'>"
        f"<div style='font-size:10px;letter-spacing:.04em;text-transform:uppercase;color:#64748b'>{label}</div>"
        f"<div style='font-size:17px;font-weight:750;color:{color};margin-top:4px'>{value}</div>"
        f"{detail_html}</div>"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _alphabet_market_data() -> dict:
    """Live GOOGL price history and six-month performance from Yahoo Finance."""
    try:
        import yfinance as yf
        hist = yf.Ticker("GOOGL").history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            return {"error": "Historique Yahoo Finance indisponible."}
        close = hist["Close"].dropna()
        if len(close) < 2:
            return {"error": "Historique de cours insuffisant."}
        six_month = close.loc[close.index >= close.index[-1] - pd.DateOffset(months=6)]
        month = close.loc[close.index >= close.index[-1] - pd.DateOffset(months=1)]
        if len(six_month) < 2 or len(month) < 2:
            return {"error": "Historique récent incomplet."}
        last, high_52w = float(close.iloc[-1]), float(close.tail(252).max())
        return {"history": six_month, "price": last,
                "change_6m": (last / float(six_month.iloc[0]) - 1) * 100,
                "change_1m": (last / float(month.iloc[0]) - 1) * 100,
                "high_52w": high_52w, "drawdown_high": (last / high_52w - 1) * 100,
                "updated": close.index[-1]}
    except Exception as exc:
        return {"error": f"Impossible de récupérer GOOGL : {type(exc).__name__}"}


@st.cache_data(ttl=3600, show_spinner=False)
def _candidate_market_data(symbol: str) -> dict:
    """Comparable adjusted market history for recommended equities."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            return {"error": "Historique Yahoo Finance indisponible."}
        close = hist["Close"].dropna()
        if len(close) < 30:
            return {"error": "Historique de cours insuffisant."}
        last = float(close.iloc[-1])
        six_month = close.loc[close.index >= close.index[-1] - pd.DateOffset(months=6)]
        month = close.loc[close.index >= close.index[-1] - pd.DateOffset(months=1)]
        if len(six_month) < 2 or len(month) < 2:
            return {"error": "Historique récent incomplet."}
        high = float(close.tail(252).max())
        return {"history": six_month, "price": last,
                "change_6m": (last / float(six_month.iloc[0]) - 1) * 100,
                "change_1m": (last / float(month.iloc[0]) - 1) * 100,
                "high_52w": high, "drawdown_high": (last / high - 1) * 100,
                "updated": close.index[-1]}
    except Exception as exc:        return {"error": f"Historique indisponible : {type(exc).__name__}"}

def render_alphabet_analysis() -> None:
    st.markdown("### 🔎 Alphabet (GOOGL) · Analyse d'achat et exposition indirecte")
    st.caption("Cours et performance actualisés depuis Yahoo Finance · exposition ETF calculée à partir des dernières positions connues du portefeuille.")
    data = _alphabet_market_data()
    if data.get("error"):
        st.warning(data["error"])
    else:
        a, b, c, d = st.columns(4)
        a.metric("Cours GOOGL", f"{data['price']:,.2f} USD")
        b.metric("Performance 6 mois", f"{data['change_6m']:+.1f} %")
        c.metric("Performance 1 mois", f"{data['change_1m']:+.1f} %")
        d.metric("Écart au plus haut 52 semaines", f"{data['drawdown_high']:.1f} %")
        st.line_chart(data["history"].rename("GOOGL · USD"), height=210)
        st.caption(f"Dernière séance disponible : {pd.Timestamp(data['updated']).strftime('%d/%m/%Y')} · plus haut 52 semaines : {data['high_52w']:,.2f} USD. Performance calculée sur les cours ajustés.")

    # VWCE: Vanguard factsheet 31 Oct 2026, Alphabet Class A + C combined.
    # EWSP: indicative estimate for two Alphabet classes in an equal-weight S&P 500 index.
    vwce_value = next((float(p["value_eur"]) for p in PORTFOLIO if p["symbol"] == "VWCE"), 0.0)
    ewsp_value = next((float(p["value_eur"]) for p in PORTFOLIO if p["symbol"] == "EWSP"), 0.0)
    vwce_weight, ewsp_weight = 0.0281, 0.0040
    vwce_exposure, ewsp_exposure = vwce_value * vwce_weight, ewsp_value * ewsp_weight
    total_exposure = vwce_exposure + ewsp_exposure
    st.markdown("#### Ton exposition actuelle à Alphabet")
    x, y, z = st.columns(3)
    x.metric("Via VWCE", f"{vwce_exposure:.2f} €", f"{vwce_weight*100:.2f} % du fonds")
    y.metric("Via EWSP", f"≈ {ewsp_exposure:.2f} €", "estimation · pondération égale")
    z.metric("Total indirect estimé", f"≈ {total_exposure:.2f} €", f"{total_exposure/float(PORTFOLIO_TOTAL_EUR)*100:.2f} % du portefeuille")
    st.caption("VWCE : poids cumulé Alphabet A + C publié par Vanguard au 31/10/2026. EWSP : estimation indicative basée sur deux classes Alphabet dans un indice S&P 500 à pondération égale ; poids exact à confirmer dans le fichier quotidien iShares. Montants calculés sur les valeurs communiquées, pas sur un inventaire broker en temps réel.")

    st.markdown("#### Ce qui explique la baisse — et ce qui peut soutenir le titre")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Points de pression**")
        st.markdown("- Free cash-flow du T3 2026 : **−9,1 Md$** ; dépenses d'investissement très élevées.")
        st.markdown("- Capex 2026 guidé à **195–205 Md$**, avec une nouvelle hausse significative annoncée pour 2027.")
        st.markdown("- Revenus publicitaires de **82 Md$**, inférieurs aux attentes ; interrogation sur la monétisation des réponses IA.")
        st.markdown("- Utilisation accrue de capacités cloud tierces : pression attendue sur les marges à court terme.")
    with right:
        st.markdown("**Éléments de solidité**")
        st.markdown("- Chiffre d'affaires T3 : **+25 % à 127,6 Md$**.")
        st.markdown("- Google Cloud : **+103 % à 30,7 Md$** ; forte demande en infrastructure IA.")
        st.markdown("- Google Search : **+12 %** et croissance des requêtes au plus haut depuis cinq ans.")
        st.markdown("- Résultat opérationnel : **+30 %** sur un an.")
    st.caption("Prochain catalyseur à surveiller : résultats du T4 2026 attendus début février 2027, notamment le capex 2027, la croissance Search et les marges Cloud.")

    st.markdown("#### Lecture d'investissement")
    if not data.get("error") and data["change_6m"] <= -15:
        st.warning("Correction importante : le titre mérite une surveillance active, mais une baisse de 15–20 % ne prouve pas à elle seule une sous-évaluation. Le risque central est le rendement futur des dépenses IA et la génération de trésorerie.")
    elif not data.get("error") and data["change_6m"] < 0:
        st.info("Le titre est en repli sur six mois ; il faut distinguer correction de marché et dégradation durable des fondamentaux.")
    else:
        st.info("Le cours ne montre pas actuellement une correction de six mois supérieure à 15 % ; ne pas conclure à une opportunité sur le seul prix.")
    st.markdown("**Approche prudente :** si tu souhaites ouvrir une ligne, privilégier une entrée fractionnée en 2 ou 3 achats plutôt qu'un achat intégral immédiat, et définir à l'avance le montant maximal d'exposition directe + ETF. Ne pas renforcer automatiquement avant d'avoir vérifié les résultats et les dépenses 2027.")


def render_portfolio_enrichment() -> None:
    st.markdown("### ⭐ Opportunités d’investissement")
    st.caption("Univers élargi d’actions et d’ETF · sélection finale orientée diversification du portefeuille.")

    refresh_col, info_col = st.columns([1, 3])
    with refresh_col:
        refresh = st.button("🔄 Actualiser les données", key="refresh_enrichment")
    with info_col:
        st.caption("Sources : Yahoo Finance puis StockAnalysis / S&P Global pour les actions US · cache 6 h.")
    if refresh:
        _load_enrichment_data.clear()

    with st.spinner("Vérification des données de marché et des signaux analystes…"):
        rows = [
            _score_enrichment_candidate(candidate, _load_enrichment_data(candidate["symbol"]))
            for candidate in ENRICHMENT_CANDIDATES
        ]

    frame = pd.DataFrame(rows)
    # A quote alone is not an investment signal. Hide empty cards rather than
    # filling the dashboard with N/D values and zero institutional-holder counts.
    def has_usable_signal(row):
        has_target = pd.notna(row.get("expected_return"))
        has_consensus = pd.notna(row.get("buy_share")) and pd.notna(row.get("ratings_total")) and row.get("ratings_total", 0) > 0
        has_institutional = pd.notna(row.get("institutional_pct")) and pd.notna(row.get("holder_count")) and row.get("holder_count", 0) > 0
        return bool(has_target or has_consensus or has_institutional)

    usable = frame[frame.apply(has_usable_signal, axis=1)].copy()
    ranked = usable[usable["score"].notna()].sort_values("score", ascending=False)
    complete_count = len(ranked)
    hidden_count = len(frame) - len(usable)

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px'>"
        f"<span style='background:#E8F7EF;color:#167447;padding:6px 10px;border-radius:20px;font-size:12px;font-weight:700'>"
        f"🟢 {complete_count} opportunité(s) avec score complet</span>"
        f"<span style='background:#EEF3FC;color:#315A9B;padding:6px 10px;border-radius:20px;font-size:12px;font-weight:700'>"
        f"🔵 {len(usable)-complete_count} candidat(s) avec signaux partiels</span>"
        f"<span style='background:#F1F2F4;color:#59636E;padding:6px 10px;border-radius:20px;font-size:12px;font-weight:700'>"
        f"⚪ {hidden_count} candidat(s) masqué(s) faute d’information exploitable</span></div>",
        unsafe_allow_html=True,
    )

    if usable.empty:
        st.info(
            "Aucune opportunité ne dispose actuellement de données suffisamment exploitables. "
            "Les candidats sans objectif de cours, consensus vérifiable ni détention institutionnelle "
            "documentée sont masqués : aucun cours isolé ni valeur N/D n’est présenté comme une opportunité."
        )
        with st.expander("Pourquoi aucun candidat n’est affiché ?"):
            for _, row in frame.iterrows():
                errors = row.get("errors") or []
                reason = " · ".join(errors[:3]) if errors else str(row.get("status", "Signaux d’investissement indisponibles"))
                st.write(f"**{row['name']} ({row['symbol']})** — {reason}")
            st.caption(
                "La couverture institutionnelle n’est pas toujours publiée pour les actions européennes. "
                "Les ETF UCITS ne disposent généralement pas d’un consensus analystes comparable à celui d’une action. "
                "Ces absences ne sont pas remplacées par des zéros."
            )
    else:
        # Two-column opportunity gadgets. Missing metrics are omitted completely.
        for offset in range(0, len(usable), 2):
            cols = st.columns(2, gap="medium")
            for col, (_, row) in zip(cols, usable.iloc[offset:offset + 2].iterrows()):
                with col:
                    with st.container(border=True):
                        accent = row.get("color", "#64748b")
                        score = row.get("score")
                        score_text = f"{score:.1f}/100" if pd.notna(score) else "Signaux partiels"
                        score_color = "#168653" if pd.notna(score) and score >= 70 else "#B7791F" if pd.notna(score) else "#64748b"
                        st.markdown(
                            f"<div style='background:linear-gradient(110deg,{accent}16,#ffffff 72%);"
                            f"border:1px solid {accent}55;border-left:5px solid {accent};border-radius:13px;"
                            f"padding:13px 14px 10px;margin:3px 0 8px'>"
                            f"<div style='display:flex;justify-content:space-between;gap:8px;align-items:flex-start'>"
                            f"<div><div style='font-size:11px;color:#64748b;font-weight:700'>{row['type']} · {row['symbol']}</div>"
                            f"<div style='font-size:17px;font-weight:750;color:#182230;margin-top:3px'>{row['name']}</div>"
                            f"<div style='font-size:11px;color:#64748b;margin-top:2px'>{row['theme']}</div></div>"
                            f"<div style='background:{score_color}18;color:{score_color};padding:6px 9px;border-radius:9px;"
                            f"font-size:13px;font-weight:800;white-space:nowrap'>{score_text}</div></div></div>",
                            unsafe_allow_html=True,
                        )

                        reason = row.get("reason")
                        if reason:
                            st.markdown(
                                f"<div style='font-size:12px;line-height:1.45;color:#344054;"
                                f"background:#F7F9FC;border-left:3px solid {accent};"
                                f"padding:8px 10px;border-radius:5px;margin:2px 0 9px'>"
                                f"<b>Pourquoi cette valeur est suivie :</b> {html.escape(str(reason))}</div>",
                                unsafe_allow_html=True,
                            )

                        metrics = []
                        if pd.notna(row.get("expected_return")):
                            price = row.get("price")
                            target = row.get("target")
                            detail = f"Cours {price:.2f} · objectif moyen {target:.2f}" if pd.notna(price) and pd.notna(target) else "Objectif moyen des analystes"
                            metrics.append(("Potentiel 12 mois", f"{row['expected_return']:+.1f} %", "#4776C8", detail))
                        if pd.notna(row.get("buy_share")) and pd.notna(row.get("ratings_total")) and row.get("ratings_total", 0) > 0:
                            metrics.append(("Buy / Strong Buy", f"{row['buy_share']*100:.0f} %", "#168C87", f"{int(row['ratings_total'])} recommandations"))
                        elif row.get("consensus_label"):
                            analyst_count = int(row.get("opinions") or 0)
                            metrics.append(("Consensus analystes", str(row["consensus_label"]), "#168C87", f"{analyst_count} analystes · {row.get('analyst_source') or 'source publique'}"))
                        if pd.notna(row.get("institutional_pct")) and pd.notna(row.get("holder_count")) and row.get("holder_count", 0) > 0:
                            metrics.append(("Détention institutionnelle", f"{row['institutional_pct']*100:.0f} %", "#7656B5", f"{int(row['holder_count'])} détenteurs déclarés"))

                        metric_cols = st.columns(len(metrics))
                        for metric_col, (label, value, color, detail) in zip(metric_cols, metrics):
                            metric_col.markdown(_gadget_metric(label, value, color, detail), unsafe_allow_html=True)

                        if pd.notna(score):
                            st.progress(min(1.0, max(0.0, float(score) / 100.0)), text=f"Score complet · {score:.1f}/100")
                        else:
                            st.caption("Score non calculé : objectif de cours ou consensus vérifiable indisponible.")
                        if row.get("type") == "Action":
                            with st.expander(f"🔎 Analyse détaillée · {row['name']} ({row['symbol']})"):
                                if row["symbol"] == "GOOGL":
                                    render_alphabet_analysis()
                                else:
                                    st.caption("Cours ajustés et performances calculés à partir de Yahoo Finance · cache 1 heure.")
                                    market = _candidate_market_data(row["symbol"])
                                    if market.get("error"):
                                        st.warning(market["error"])
                                    else:
                                        ma, mb, mc, md = st.columns(4)
                                        ma.metric("Cours indicatif", f"{market['price']:,.2f}")
                                        mb.metric("Performance 6 mois", f"{market['change_6m']:+.1f} %")
                                        mc.metric("Performance 1 mois", f"{market['change_1m']:+.1f} %")
                                        md.metric("Écart au plus haut 52 sem.", f"{market['drawdown_high']:.1f} %")
                                        st.line_chart(market["history"].rename(f"{row['symbol']} · cours ajusté"), height=180)
                                        st.caption(f"Dernière séance : {pd.Timestamp(market['updated']).strftime('%d/%m/%Y')} · plus haut 52 semaines : {market['high_52w']:,.2f}.")
                                    st.markdown("**Thèse d’investissement / fondamentaux**")
                                    st.write(row.get("reason") or "Aucune thèse fondamentale suffisamment documentée.")
                                    held = next((float(p["value_eur"]) for p in PORTFOLIO if p["symbol"] == row["symbol"]), 0.0)
                                    st.metric("Exposition directe actuelle", f"{held:.2f} €", f"{held / PORTFOLIO_TOTAL_EUR * 100:.2f} % du portefeuille")
                                    if row["symbol"] == "ASML":
                                        # VWCE: ASML weight 0.66%, holdings snapshot dated 30 Sep 2026.
                                        # EWSP tracks the S&P 500 Equal Weight; ASML is Dutch-domiciled
                                        # and is not eligible for the S&P 500's US-company universe.
                                        vwce_value = next((float(p["value_eur"]) for p in PORTFOLIO if p["symbol"] == "VWCE"), 0.0)
                                        ewsp_value = next((float(p["value_eur"]) for p in PORTFOLIO if p["symbol"] == "EWSP"), 0.0)
                                        vwce_weight = 0.0066
                                        vwce_exposure = vwce_value * vwce_weight
                                        ewsp_exposure = 0.0
                                        total_indirect = vwce_exposure + ewsp_exposure
                                        st.markdown("#### Ton exposition actuelle à ASML")
                                        total_asml_exposure = held + total_indirect
                                        exp_a, exp_b, exp_c = st.columns(3)
                                        exp_a.metric("Via VWCE", f"{vwce_exposure:.2f} €", f"{vwce_weight*100:.2f} % du fonds")
                                        exp_b.metric("Via EWSP", f"{ewsp_exposure:.2f} €", "pas de ligne ASML dans le S&P 500")
                                        exp_c.metric("Exposition totale estimée", f"≈ {total_asml_exposure:.2f} €", f"{total_asml_exposure/float(PORTFOLIO_TOTAL_EUR)*100:.2f} % du portefeuille")
                                        st.caption(
                                            "Exposition totale = position directe ASML + exposition indirecte via ETF. "
                                            "VWCE : pondération ASML de 0,66 %, dernière composition identifiée au 30/09/2026 ; "
                                            "à actualiser avec le prochain fichier de positions Vanguard. "
                                            "EWSP : 0 € ; ASML est domiciliée aux Pays-Bas, tandis que le S&P 500 retient des sociétés "
                                            "américaines. Montants calculés sur les valeurs du portefeuille communiquées, pas sur un inventaire broker en temps réel."
                                        )
                                    if pd.notna(row.get("expected_return")):
                                        st.metric("Potentiel indicatif 12 mois", f"{float(row['expected_return']):+.1f} %", "objectif moyen analystes")
                                    if pd.notna(row.get("buy_share")) and pd.notna(row.get("ratings_total")) and float(row.get("ratings_total") or 0) > 0:
                                        st.caption(f"Consensus : {float(row['buy_share'])*100:.0f} % Buy / Strong Buy sur {int(row['ratings_total'])} recommandations. Indicateur secondaire, pas un signal d’achat autonome.")
                                    st.markdown("**À vérifier avant achat**")
                                    st.write("Valorisation au cours actuel, trajectoire des marges et du cash-flow, endettement, dernières perspectives de la direction et date des prochains résultats. Les données non disponibles ne sont pas inventées.")
                                    if not market.get("error") and market["change_6m"] <= -15:
                                        st.warning("Correction importante : elle justifie une analyse approfondie, mais ne constitue pas à elle seule un signal d’achat. Privilégier une entrée fractionnée si les fondamentaux restent solides.")
                                    elif not market.get("error") and market["change_6m"] < 0:
                                        st.info("Le titre recule sur six mois : distinguer correction de marché et dégradation durable des fondamentaux.")
                                    elif not market.get("error"):
                                        st.info("Le titre progresse sur six mois : éviter de poursuivre le cours sans comparer valorisation, croissance et risques.")

                        if row.get("errors"):
                            with st.expander(f"Sources à vérifier · {row['symbol']}"):
                                st.caption(" · ".join(row["errors"][:5]))

    if complete_count:
        best = ranked.iloc[0]
        st.success(f"Meilleur score vérifiable : {best['name']} · {best['score']:.1f}/100.")
        with st.expander("Classement des opportunités complètes"):
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                st.write(f"**{rank}. {row['name']} ({row['symbol']}) — {row['score']:.1f}/100**")

    with st.expander("Méthode de notation et limites"):
        st.write(
            "Score /100 : potentiel de rendement relatif au repère S&P 500 (50 points), "
            "part des recommandations Buy / Strong Buy et profondeur de couverture (30 points), "
            "détention institutionnelle déclarée (20 points). Le score global n’est calculé que si les trois piliers sont renseignés."
        )
        st.write(
            "Les cartes sans signal exploitable sont volontairement masquées. Une donnée absente n’est jamais assimilée à zéro. "
            "La détention institutionnelle déclarée n’est pas un flux d’achats récents, peut être publiée avec retard et n’est plus un critère éliminatoire. "
            "Les ETF UCITS sont évalués séparément lorsque les données de fonds pertinentes sont disponibles ; "
            "un consensus d’action ne leur est pas artificiellement attribué."
        )
        st.caption(
            "Univers de surveillance, pas une recommandation personnalisée. Avant achat : vérifier le KID PRIIPs, "
            "les frais, la liquidité, la devise, la réplication et l’éligibilité chez le courtier."
        )

def render_financial_intelligence_dashboard() -> None:
    render_portfolio_enrichment()
    st.markdown("---")
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
    st.caption("Base portefeuille mise à jour avec l'achat ASML de 100 USD, converti à titre indicatif en 91,22 € au taux de référence du 24/11/2026. La performance de départ est recalculée par dilution à rendement nul sur ce nouvel apport ; les variations live restent estimatives et ne remplacent pas le PRU du courtier.")
    if PLANNED_BUYS:
        st.markdown("### Achats prévus · non comptabilisés")
        planned_cols = st.columns(len(PLANNED_BUYS))
        for planned_col, item in zip(planned_cols, PLANNED_BUYS):
            planned_col.metric(f"{item['asset']} · {item['status']}", f"{item['amount_usd']:.0f} USD", f"≈ {item['amount_eur']:.2f} €")
            planned_col.caption(item["note"])
    st.markdown("### Événements classés selon leur impact estimé sur ton portefeuille")

    # A missing or stale CSV must trigger a real collection, not a dead-end empty state.
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    last_run = pd.to_datetime(state.get("last_run_utc"), errors="coerce", utc=True)
    stale = pd.isna(last_run) or (pd.Timestamp.now(tz="UTC") - last_run) > pd.Timedelta(hours=12)
    has_store = EVENTS_FILE.exists() and EVENTS_FILE.stat().st_size > 0
    col_collect, col_status = st.columns([1, 3])
    with col_collect:
        manual_collect = st.button("🔄 Collecter les événements", key="financial_intelligence_collect")
    if manual_collect or (not has_store and not st.session_state.get("_fi_initial_collection_attempted", False)) or (stale and not st.session_state.get("_fi_stale_collection_attempted", False)):
        if not has_store:
            st.session_state["_fi_initial_collection_attempted"] = True
        if stale:
            st.session_state["_fi_stale_collection_attempted"] = True
        with st.spinner("Collecte des actualités financières et géopolitiques depuis les flux RSS…"):
            result = run_agent(days=7)
        st.session_state["_fi_last_collection_result"] = result
        if result["collected"] > 0:
            st.success(f"Collecte terminée : {result['collected']} événement(s) détecté(s), {result['new_events']} nouveau(x), {result['alerts_total']} alerte(s) suivie(s).")
        elif result["errors"]:
            st.warning("La collecte n'a retourné aucun événement. Consulte les erreurs de flux ci-dessous ; les événements déjà enregistrés sont conservés.")
        else:
            st.info("Les flux ont répondu, mais aucun article daté des 7 derniers jours n'a été retenu.")
        st.rerun()

    last_result = st.session_state.get("_fi_last_collection_result")
    if last_result:
        with st.expander("Détails de la dernière collecte", expanded=bool(last_result.get("errors"))):
            st.write(f"Exécution : {last_result.get('run_at', 'inconnue')} · {last_result.get('collected', 0)} événement(s) · {last_result.get('new_events', 0)} nouveau(x)")
            if last_result.get("errors"):
                st.warning("Sources en erreur : " + " · ".join(last_result["errors"]))
    if not has_store:
        st.error("Le fichier d'événements est absent ou vide et aucune actualité n'a pu être enregistrée. Utilise « Collecter les événements » pour relancer la collecte et consulte les erreurs de flux.")
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
        # Use the broadly supported container API; older Streamlit Cloud
        # runtimes do not accept the newer container(key=...) argument.
        with st.container(border=True):
            st.markdown(
                f"<div style='background:{bg};border-left:5px solid {accent};"
                f"border-radius:9px;padding:10px 12px;margin:0 0 10px 0'>"
                f"<span style='font-size:0.78rem;color:{accent};font-weight:700'>"
                f"{'ÉVÉNEMENT FAVORABLE' if side=='positive' else 'ÉVÉNEMENT À RISQUE'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
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

    # Biais directionnel agrégé sur les événements actuellement retenus.
    # Les événements peuvent se recouper : ce cumul est un indicateur de pression,
    # pas une prévision de performance ni une probabilité statistique.
    favorable_pressure=sum(item["net"] for item in positive_events)
    adverse_pressure=sum(abs(item["net"]) for item in risk_events)
    directional_net=favorable_pressure-adverse_pressure
    gross_pressure=favorable_pressure+adverse_pressure
    directional_pct=(directional_net/live["total"]*100) if live["total"] else 0.0
    adverse_share=(adverse_pressure/gross_pressure*100) if gross_pressure else 0.0
    if gross_pressure <= 0.005:
        bias_label="Neutre · aucun biais net mesurable"
        bias_color="#68727D"
        bias_icon="⚪"
    elif directional_net > gross_pressure*0.10:
        bias_label="Biais favorable"
        bias_color="#168653"
        bias_icon="🟢"
    elif directional_net < -gross_pressure*0.10:
        bias_label="Biais défavorable"
        bias_color="#C74646"
        bias_icon="🔴"
    else:
        bias_label="Biais équilibré · incertitude élevée"
        bias_color="#B7791F"
        bias_icon="🟠"

    st.markdown("### Biais directionnel net du portefeuille")
    with st.container(border=True):
        st.markdown(
            f"<div style='padding:12px 14px;border-radius:10px;background:#F4F6F8;"
            f"border-left:5px solid {bias_color};margin-bottom:10px'>"
            f"<div style='font-size:0.78rem;color:#59636E;font-weight:700'>PRESSION AGRÉGÉE DES ÉVÉNEMENTS SUIVIS</div>"
            f"<div style='font-size:1.35rem;font-weight:750;color:{bias_color};margin-top:3px'>{bias_icon} {bias_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        m1,m2,m3=st.columns(3)
        m1.metric("Impact net estimé",f"{directional_net:+,.2f} €",f"{directional_pct:+.2f}% du portefeuille")
        m2.metric("Pression favorable",f"+{favorable_pressure:,.2f} €",f"{len(positive_events)} événement(s)")
        m3.metric("Pression défavorable",f"-{adverse_pressure:,.2f} €",f"{len(risk_events)} événement(s)")
        if gross_pressure > 0.005:
            green_width=100.0-adverse_share
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:0.76rem;color:#59636E;margin:5px 0 4px'>"
                f"<span>Favorable {green_width:.0f}%</span><span>Défavorable {adverse_share:.0f}%</span></div>"
                f"<div style='height:13px;display:flex;overflow:hidden;border-radius:20px;background:#E7EAEE'>"
                f"<div style='width:{green_width:.2f}%;background:#20A36A'></div>"
                f"<div style='width:{adverse_share:.2f}%;background:#D94B4B'></div></div>",
                unsafe_allow_html=True,
            )
        st.caption("Lecture directionnelle des impacts nets estimés sur les événements affichés. Les événements peuvent se recouper : les montants ne sont pas additifs au sens d'une prévision de perte ou de gain.")
    st.markdown("---")

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
        st.write("Le biais directionnel net additionne les impacts estimés des événements retenus pour donner une indication de pression favorable ou défavorable. Des événements peuvent se recouper ou décrire le même facteur macroéconomique : ce cumul ne doit donc pas être interprété comme une perte/gain attendu ni comme une prévision de performance.")
        st.write("Le suivi des cours est ancré à l'instantané communiqué. Sans quantités exactes, les variations sont estimées à partir des valeurs de départ. Certains tickers ou conversions peuvent être indisponibles.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_agent(args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()