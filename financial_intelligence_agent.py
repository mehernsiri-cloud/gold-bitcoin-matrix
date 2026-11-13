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
    {"symbol":"WELS","name":"Amundi S&P Global Health Care","value_eur":100.32,"return_pct":12.56,"group":"Healthcare","factors":["healthcare","rates","usd","equity_risk"]},
    {"symbol":"XOM","name":"Exxon Mobil","value_eur":98.59,"return_pct":6.65,"group":"Energy","factors":["brent","gas","energy_supply","usd","rates"]},
    {"symbol":"XDWU","name":"Xtrackers MSCI World Utilities","value_eur":95.11,"return_pct":6.72,"group":"Utilities","factors":["electricity","rates","regulation","usd"]},
    {"symbol":"AI","name":"Air Liquide","value_eur":64.84,"return_pct":14.43,"group":"Industrials","factors":["industrial_growth","energy_costs","rates","usd"]},
    {"symbol":"O4JO","name":"iShares S&P 500 Equal Weight ETF","value_eur":51.54,"return_pct":3.07,"group":"US equities","factors":["us_growth","rates","usd","equity_risk"]},
    {"symbol":"FTAI","name":"FTAI Aviation","value_eur":49.61,"return_pct":-9.12,"group":"Aerospace","factors":["aviation","credit","rates","industrial_growth"]},
    {"symbol":"FRO","name":"Frontline","value_eur":35.45,"return_pct":1.02,"group":"Shipping","factors":["tanker_rates","brent","trade_routes","energy_supply"]},
    {"symbol":"LMT","name":"Lockheed Martin","value_eur":28.31,"return_pct":13.31,"group":"Defence","factors":["defence_budgets","geopolitics","rates","usd"]},
    {"symbol":"HO","name":"Thales","value_eur":23.92,"return_pct":-1.71,"group":"Defence & technology","factors":["defence_budgets","geopolitics","cybersecurity","industrial_growth"]},
]
CASH_EUR = 4.93
PORTFOLIO_TOTAL_EUR = 900.07
PORTFOLIO_RETURN_PCT = 4.63

# Topic-to-portfolio transmission map. Directions are conditional sensitivities, not price forecasts.
IMPACT_MAP = {
    "Central banks & rates": {"positive":["TTE","XOM","FRO"],"negative":["XDWU","VWCE","WELS","O4JO","FTAI","AI","LMT","HO"],"watch":["USD exposure in US-listed holdings"],"reason":"Une hausse des taux peut peser sur les valorisations et le financement ; l'effet devise et les anticipations de croissance peuvent compenser."},
    "Inflation, debt & economy": {"positive":["TTE","XOM"],"negative":["XDWU","VWCE","WELS","O4JO","FTAI","AI"],"watch":["FRO","LMT","HO"],"reason":"Une inflation persistante peut retarder les baisses de taux ; l'effet dépend de sa composante énergie/salaires et de la réaction des rendements."},
    "Geopolitics & conflict": {"positive":["TTE","XOM","LMT","HO"],"negative":["VWCE","O4JO","FTAI"],"watch":["FRO","XDWU","WELS","AI"],"reason":"Le canal dépend des sanctions, des flux énergétiques, des routes maritimes et des budgets de défense ; les valeurs défense ne montent pas mécaniquement."},
    "Energy & commodities": {"positive":["TTE","XOM"],"negative":["AI","WELS"],"watch":["FRO","VWCE","XDWU","O4JO"],"reason":"Un choc d'offre pétrolière peut soutenir les producteurs mais renchérir les coûts et peser sur la demande ; les tarifs des tankers dépendent aussi de l'offre de navires et des routes."},
    "Financial system & liquidity": {"positive":[],"negative":["VWCE","WELS","XDWU","O4JO","FTAI"],"watch":["TTE","XOM","FRO","AI","LMT","HO"],"reason":"Un resserrement du crédit touche d'abord les actifs risqués et les entreprises dépendantes du financement ; rechercher des signes confirmés de contagion."},
    "AI, electricity & infrastructure": {"positive":["XDWU","AI"],"negative":[],"watch":["VWCE","O4JO"],"reason":"La hausse des investissements réseau et de la demande électrique peut soutenir utilities et fournisseurs industriels, sous réserve de valorisations, contrats et financement."},
    "Defence & aerospace": {"positive":["LMT","HO"],"negative":[],"watch":["FTAI","VWCE","O4JO"],"reason":"Les annonces budgétaires ne deviennent un catalyseur qu'après budgets votés, contrats attribués et calendrier d'exécution."},
    "Cybersecurity": {"positive":["HO"],"negative":[],"watch":["VWCE","O4JO"],"reason":"Thales a une exposition aux activités cyber, mais l'effet boursier dépend de l'ampleur de l'incident et des contrats ; un incident n'est pas automatiquement positif."},
    "Healthcare": {"positive":["WELS"],"negative":[],"watch":["VWCE","O4JO"],"reason":"Les décisions réglementaires et résultats cliniques peuvent toucher les composants santé ; l'ETF diversifié amortit les risques propres à une entreprise."},
    "Crypto & Bitcoin": {"positive":[],"negative":[],"watch":["VWCE","O4JO","XDWU"],"reason":"Aucune exposition directe au Bitcoin n'apparaît dans le portefeuille communiqué ; seul un effet indirect via liquidité et appétit pour le risque est possible."},
    "FX & safe havens": {"positive":[],"negative":[],"watch":["VWCE","TTE","XOM","WELS","O4JO","LMT"],"reason":"Les variations EUR/USD affectent la conversion en euros des actifs et revenus en dollars ; le sens dépend de la devise de cotation, couverture et revenus."},
    "Equity markets & companies": {"positive":[],"negative":[],"watch":["VWCE","WELS","O4JO","XDWU","TTE","XOM","FTAI","FRO","AI","LMT","HO"],"reason":"L'impact dépend de l'entreprise ou de l'indice cité, de la surprise par rapport aux attentes et de la réaction du cours."},
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


def render_financial_intelligence_dashboard() -> None:
    """Decision-first dashboard connected to the user's confirmed portfolio snapshot."""
    st.markdown("""
    <style>
    .stApp{background:#F5F7FB;color:#26384D}
    [data-testid="stHeader"]{background:rgba(255,255,255,.92)}
    [data-testid="stSidebar"]{background:#F0F3F8}
    .block-container{padding-top:1.7rem;padding-bottom:2.5rem;max-width:1420px}
    .fi-hero{background:linear-gradient(110deg,#fff,#EDF4FC);border:1px solid #DCE5F0;border-radius:16px;padding:22px 26px;margin-bottom:16px}
    .fi-title{font-size:29px;font-weight:800;color:#243B59;margin:5px 0}
    .fi-card{background:#fff;border:1px solid #DFE6EF;border-radius:14px;padding:17px 19px;margin:11px 0;box-shadow:0 2px 8px rgba(30,55,85,.04)}
    .fi-muted{font-size:12px;color:#718198}
    .fi-tag{display:inline-block;padding:5px 9px;border-radius:20px;background:#EAF2FB;color:#315F8A;font-size:10px;font-weight:800}
    .fi-verdict{background:#F0F6FC;border-left:3px solid #4D83B8;padding:11px 13px;border-radius:7px;color:#334B66;font-size:13px;margin-top:12px}
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
    decisions = _decision_view(df) if not df.empty else df.copy()
    last_run = str(state.get("last_run_utc", "Non disponible"))[:16].replace("T", " ") + " UTC"

    st.markdown(f"""
    <div class="fi-hero">
      <div class="fi-tag">PORTFOLIO-LINKED FINANCIAL INTELLIGENCE</div>
      <div class="fi-title">Qu'est-ce qui peut réellement affecter tes investissements ?</div>
      <div style="color:#66778D">Sélection d'événements, transmission vers tes positions et scénarios conditionnels.</div>
      <div class="fi-muted" style="margin-top:10px">Snapshot portefeuille communiqué · Valeurs non actualisées automatiquement · Dernière collecte : {html.escape(last_run)}</div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3=st.columns(3)
    k1.metric("Valeur du portefeuille", f"{PORTFOLIO_TOTAL_EUR:,.2f} €".replace(",", " "), f"+{PORTFOLIO_RETURN_PCT:.2f} %")
    k2.metric("Positions investies", f"{len(PORTFOLIO)}")
    k3.metric("Liquidités", f"{CASH_EUR:.2f} €")
    st.caption("Snapshot de référence fourni le 13/11/2026. Les cotations, quantités et performances ne sont pas rafraîchies par cette version.")

    with st.expander("Voir mes positions et leurs pondérations", expanded=False):
        pview=pd.DataFrame(PORTFOLIO)
        pview["poids_pct"]=(pview["value_eur"]/PORTFOLIO_TOTAL_EUR*100).round(1)
        pview=pview.rename(columns={"symbol":"Ticker","name":"Support","value_eur":"Valeur €","return_pct":"Performance %","group":"Classe","poids_pct":"Poids %"})
        st.dataframe(pview[["Ticker","Support","Classe","Valeur €","Performance %","Poids %"]],use_container_width=True,hide_index=True)
        st.caption("Les 11 positions totalisent 895,14 € ; les 4,93 € restants sont des liquidités en devises. Total : 900,07 €.")

    c1,c2=st.columns([5,1])
    with c1:
        st.caption("L'agent doit expliquer les effets directs, indirects et les incertitudes — pas seulement compter les articles.")
    with c2:
        if st.button("Actualiser",use_container_width=True):
            st.rerun()
    if st.button("Lancer la collecte et l'analyse",type="primary"):
        with st.spinner("Collecte et classement des événements…"):
            result=run_agent()
        st.success(f"{result['collected']} articles analysés · {result['new_events']} nouveaux événements.")
        if result["errors"]:
            st.warning("Flux indisponibles : "+" · ".join(result["errors"][:5]))
        st.rerun()

    st.markdown("### Les événements qui touchent tes positions")
    if decisions.empty:
        st.info("Aucun événement suffisamment pertinent dans les archives actuelles. Pas d'alerte artificielle.")
    else:
        for rank,(_,row) in enumerate(decisions.head(5).iterrows(),1):
            impact=_portfolio_impact(str(row.get("topic","")))
            title=html.escape(str(row.get("title","Événement sans titre")))
            summary=html.escape(str(row.get("summary","") or "Résumé indisponible.")[:650])
            topic=html.escape(str(row.get("topic","")))
            source=html.escape(str(row.get("source","")))
            reason=html.escape(impact["reason"])
            url=str(row.get("url",""))
            st.markdown(f"""
            <div class="fi-card">
              <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">
                <span class="fi-tag">ÉVÉNEMENT #{rank} · {html.escape(str(row.get('priority','P3')))}</span>
                <span class="fi-muted">{html.escape(str(row.get('published_at').strftime('%d/%m/%Y %H:%M UTC')))} · {source}</span>
              </div>
              <h3 style="font-size:18px;color:#253B56;margin:12px 0 7px">{title}</h3>
              <div style="font-size:13px;color:#53677F;line-height:1.55">{summary}</div>
              <div class="fi-verdict"><b>Pourquoi cela touche ton portefeuille :</b> {reason}</div>
            </div>""",unsafe_allow_html=True)

            pos_names=", ".join(f"{p['symbol']} ({p['value_eur']:.2f} €)" for p in impact["positive"]) or "Aucun bénéficiaire direct identifié"
            neg_names=", ".join(f"{p['symbol']} ({p['value_eur']:.2f} €)" for p in impact["negative"]) or "Aucun perdant direct identifié"
            watch_names=", ".join(f"{p['symbol']} ({p['value_eur']:.2f} €)" for p in impact["watch"]) or "Aucun"
            a,b=st.columns(2)
            with a:
                st.markdown("**Canal potentiellement favorable**")
                st.success(pos_names)
                st.markdown("**Canal potentiellement défavorable**")
                st.error(neg_names)
            with b:
                st.markdown("**À surveiller / effet incertain**")
                st.info(watch_names)
                st.markdown("**Lecture conditionnelle**")
                st.write("Le sens indiqué est une sensibilité théorique au facteur identifié, pas une prévision du cours ni une recommandation d'achat/vente.")
            if url.startswith("http"):
                st.markdown(f"[Lire la source]({url})")

    st.markdown("### Exposition transversale de ton portefeuille")
    st.write("Plusieurs positions peuvent réagir au même facteur :")
    factor_rows=[
        ("Taux et valorisations","VWCE · WELS · XDWU · O4JO · FTAI · Air Liquide · LMT · Thales"),
        ("Pétrole et énergie","TotalEnergies · Exxon Mobil · Frontline"),
        ("Dollar et marchés américains","VWCE · Exxon Mobil · WELS · O4JO · Lockheed Martin"),
        ("Défense et géopolitique","Lockheed Martin · Thales"),
        ("Électricité et IA","XDWU · Air Liquide, avec exposition indirecte de VWCE/O4JO"),
    ]
    for label,assets in factor_rows:
        st.markdown(f"- **{label} :** {assets}")

    st.markdown("### Ce que l'agent ne sait pas encore mesurer")
    st.warning("Cette version relie les événements à tes positions avec une cartographie explicable. Elle ne calcule pas encore de P&L prédictif en euros : il manque des sensibilités historiques, des cotations synchronisées, les devises de chaque ligne et une calibration par scénario. Aucun rendement cible ou perte chiffrée ne sera inventé.")
    with st.expander("Historique complet et méthode"):
        st.write(f"{len(df)} événements archivés · {df['domain'].nunique() if not df.empty and 'domain' in df else 0} domaines sources · dernière collecte {last_run}.")
        if not df.empty:
            st.dataframe(df[["published_at","priority","topic","impact_score","title","source","verification_status","url"]].head(100),use_container_width=True,hide_index=True)
    st.caption("Agent gratuit sans API payante : collecte RSS et règles explicables. La corroboration de sources ne constitue pas une vérification indépendante des faits.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_agent(args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
