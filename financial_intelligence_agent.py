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


def render_financial_intelligence_dashboard() -> None:
    """Light, decision-first investor brief matching the application's light visual language."""
    st.markdown("""
    <style>
    .stApp{background:#F5F7FB;color:#26384D}
    [data-testid="stHeader"]{background:rgba(255,255,255,.92)}
    [data-testid="stSidebar"]{background:#F0F3F8}
    .block-container{padding-top:1.7rem;padding-bottom:2.5rem;max-width:1420px}
    .fi-hero{background:linear-gradient(110deg,#fff,#EDF4FC);border:1px solid #DCE5F0;border-radius:16px;padding:22px 26px;margin-bottom:16px}
    .fi-eyebrow{font-size:11px;letter-spacing:1.5px;color:#66809F;font-weight:800}
    .fi-title{font-size:30px;font-weight:800;color:#243B59;margin:5px 0}
    .fi-sub{font-size:14px;color:#66778D}
    .fi-card{background:#fff;border:1px solid #DFE6EF;border-radius:14px;padding:17px 19px;margin:11px 0;box-shadow:0 2px 8px rgba(30,55,85,.04)}
    .fi-card-title{font-size:17px;font-weight:750;color:#253B56;line-height:1.4;margin:8px 0}
    .fi-muted{font-size:12px;color:#718198}
    .fi-tag{display:inline-block;padding:5px 9px;border-radius:20px;background:#EAF2FB;color:#315F8A;font-size:10px;font-weight:800}
    .fi-verdict{background:#F0F6FC;border-left:3px solid #4D83B8;padding:11px 13px;border-radius:7px;color:#334B66;font-size:13px;margin-top:12px}
    .fi-action{background:#F7F9FC;border:1px solid #E4EAF1;border-radius:8px;padding:10px 12px;color:#465B73;font-size:13px;margin-top:9px}
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
      <div class="fi-eyebrow">PERSONAL INVESTMENT INTELLIGENCE · DAILY BRIEF</div>
      <div class="fi-title">Ce qui mérite ton attention aujourd'hui</div>
      <div class="fi-sub">Une sélection courte des événements susceptibles de changer le contexte d'investissement — pas un flux d'actualités à trier.</div>
      <div class="fi-muted" style="margin-top:10px">Dernière collecte : {html.escape(last_run)} · {len(decisions)} thèmes décisionnels retenus</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([5, 1])
    with c1:
        st.caption("Synthèse orientée décision · importance potentielle, fiabilité indicative et fraîcheur.")
    with c2:
        if st.button("Actualiser", use_container_width=True):
            st.rerun()
    if st.button("Lancer l'analyse maintenant", type="primary"):
        with st.spinner("Collecte et hiérarchisation des événements…"):
            result = run_agent()
        st.success(f"Collecte terminée : {result['collected']} articles analysés, {result['new_events']} nouveaux.")
        if result["errors"]:
            st.warning("Flux indisponibles : " + " · ".join(result["errors"][:5]))
        st.rerun()

    st.markdown("### Synthèse décisionnelle")
    if decisions.empty:
        st.info("Aucun événement suffisamment pertinent n'a été identifié. L'agent préfère ne rien signaler plutôt que remplir le dashboard de bruit.")
    else:
        for rank, (_, row) in enumerate(decisions.head(5).iterrows(), 1):
            title = html.escape(str(row.get("title", "Événement sans titre")))
            summary = html.escape(str(row.get("summary", "") or "Résumé non fourni.")[:520])
            topic = html.escape(str(row.get("topic", "")))
            source = html.escape(str(row.get("source", "")))
            channel = html.escape(str(row.get("decision_channel", "")))
            why = html.escape(str(row.get("decision_why", "")))
            action = html.escape(str(row.get("decision_action", "")))
            tag = html.escape(str(row.get("decision_tag", "")))
            priority = str(row.get("priority", "P3"))
            published = row["published_at"].strftime("%d/%m/%Y · %H:%M UTC")
            score = float(row.get("decision_score", 0) or 0)
            conf = int(row.get("confidence", 0) or 0)
            url = str(row.get("url", ""))
            st.markdown(f"""
            <div class="fi-card">
              <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">
                <span class="fi-tag">#{rank} · {tag}</span><span class="fi-muted">{published} · {source}</span>
              </div>
              <div class="fi-card-title">{title}</div>
              <div class="fi-muted">{topic} · Canal de transmission : {channel}</div>
              <div style="font-size:13px;color:#53677F;line-height:1.55;margin-top:9px">{summary}</div>
              <div class="fi-verdict"><b>Pourquoi cela compte :</b> {why}</div>
              <div class="fi-action"><b>Lecture de l'agent :</b> {action}</div>
              <div class="fi-muted" style="margin-top:9px">Priorité de source : {priority} · Score de sélection : {score:.0f} · Confiance indicative : {conf}/95</div>
            </div>""", unsafe_allow_html=True)
            if url.startswith("http"):
                st.markdown(f'<div style="margin-top:-5px;margin-bottom:10px"><a href="{html.escape(url, quote=True)}" target="_blank" style="color:#3977B5;font-size:12px;font-weight:700;text-decoration:none">Lire la source ↗</a></div>', unsafe_allow_html=True)

    st.markdown("### Décision proposée")
    st.info("Pour le moment : surveiller et vérifier les signaux ci-dessus, sans transaction automatique. L'agent n'a pas encore accès aux quantités, prix de revient et limites de risque de ton portefeuille ; il ne peut donc pas conclure qu'une position précise doit être renforcée ou réduite.")
    if not df.empty:
        background = df[~df.index.isin(decisions.index)].head(6)
        if not background.empty:
            with st.expander(f"{len(background)} autres sujets — arrière-plan, non prioritaires"):
                for _, row in background.iterrows():
                    st.write(f"• {row.get('topic','')} — {row.get('title','')} ({row.get('source','')})")
    with st.expander("Méthode et historique technique"):
        st.write(f"Événements archivés : {len(df)} · Sources distinctes : {df['domain'].nunique() if not df.empty and 'domain' in df else 0} · Dernière collecte : {last_run}.")
        st.write("Le classement privilégie les thèmes macro-financiers, liquidité, géopolitique, énergie, devises, Bitcoin, cybersécurité et secteurs d'investissement. Les sujets sans canal de transmission financier identifiable sont écartés.")
        if not df.empty:
            st.dataframe(df[["published_at","priority","topic","impact_score","title","source","verification_status","url"]].head(100), use_container_width=True, hide_index=True)
    st.caption("Limite actuelle : l'analyse repose sur titres et résumés RSS et des règles explicables. Le score hiérarchise la veille, il ne mesure pas une probabilité. La prochaine étape sera l'analyse du texte intégral et la confrontation aux données de marché.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_agent(args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
