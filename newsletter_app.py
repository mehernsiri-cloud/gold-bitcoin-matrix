# WELS Actions — tableau de bord financier et veille
# Installation : pip install -r requirements.txt
# Lancement : streamlit run newsletter_app.py

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
import html
import re

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import trafilatura
except ImportError:
    trafilatura = None


APP_TITLE = "WELS Actions"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WELSActions/1.0)"}

RSS_FEEDS = {
    "France Info Économie": "https://www.francetvinfo.fr/economie.rss",
    "France Info Monde": "https://www.francetvinfo.fr/monde.rss",
    "France Info Société": "https://www.francetvinfo.fr/societe.rss",
    "Boursorama": "https://www.boursorama.com/rss/actualites/",
    # Reuters' legacy feeds can be unavailable; failures are handled per source.
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "GDELT — conflits": "https://api.gdeltproject.org/api/v2/doc/doc?query=(war%20OR%20conflict%20OR%20missile%20OR%20sanctions%20OR%20ceasefire)&mode=artlist&format=rss&maxrecords=40&sort=datedesc",
    "GDELT — France approvisionnement": "https://api.gdeltproject.org/api/v2/doc/doc?query=(France%20AND%20(shortage%20OR%20strike%20OR%20blockade%20OR%20fuel%20OR%20electricity%20OR%20supply))&mode=artlist&format=rss&maxrecords=30&sort=datedesc",
}

TOPICS = {
    "Tous les thèmes": [],
    "Marchés et économie": ["market", "markets", "bourse", "marché", "actions", "inflation", "fed", "bce", "banque centrale", "taux", "obligation", "economy", "économie", "recession", "récession", "dette"],
    "Or et Bitcoin": ["gold", "or ", "bitcoin", "crypto", "btc", "xau"],
    "Énergie et transport maritime": ["oil", "pétrole", "energy", "énergie", "gas", "gaz", "shipping", "maritime", "tanker", "totalenergies", "exxon"],
    "Santé": ["health", "santé", "pharma", "medical", "médical", "biotech", "healthcare", "médicament"],
    "Électricité et utilities": ["electricity", "électricité", "utilities", "grid", "power", "data centre", "data center", "schneider", "siemens"],
    "Défense et géopolitique": ["defence", "defense", "guerre", "war", "military", "militaire", "conflict", "conflit", "otan", "nato", "sanction", "géopolitique"],
    "France : approvisionnement": ["france", "île-de-france", "carburant", "fuel", "water", "eau", "food", "alimentation", "strike", "grève", "blocage", "supply", "approvisionnement", "médicament", "baby", "électricité"],
    "Suisse et devises": ["swiss", "suisse", "switzerland", "franc suisse", "chf", "eur/chf", "eur/aed", "banque nationale suisse"],
}

MARKET_SYMBOLS = {
    "TotalEnergies": "TTE.PA",
    "Exxon Mobil": "XOM",
    "Vanguard FTSE All-World (VWRL)": "VWRL.AS",
    "Frontline": "FRO",
    "FTAI Aviation": "FTAI",
    "Thales": "HO.PA",
    "Lockheed Martin": "LMT",
    "Air Liquide": "AI.PA",
    "ETF utilities XDWU (symbole à vérifier)": "XDWU.L",
    "Schneider Electric": "SU.PA",
    "Siemens": "SIE.DE",
    "S&P 500": "^GSPC",
    "MSCI ACWI (ACWI)": "ACWI",
}
FX_SYMBOLS = {"EUR/CHF": "EURCHF=X", "EUR/AED": "EURAED=X"}
PERIODS = {"1 mois": "1mo", "3 mois": "3mo", "6 mois": "6mo", "1 an": "1y", "2 ans": "2y", "5 ans": "5y"}


def clean_html(value):
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def parse_date(value):
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        try:
            return pd.to_datetime(value, utc=True).tz_convert(None).to_pydatetime()
        except Exception:
            return None


def topic_for(text):
    text = (text or "").lower()
    for topic, words in TOPICS.items():
        if topic != "Tous les thèmes" and any(word in text for word in words):
            return topic
    return "Autres actualités"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(max_per_feed=25):
    if feedparser is None:
        return pd.DataFrame(), ["Le module feedparser n'est pas installé."]
    articles, errors = [], []
    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            if not parsed.entries:
                errors.append(f"{source} : flux vide ou format non reconnu.")
            for item in parsed.entries[:max_per_feed]:
                title = clean_html(item.get("title", ""))
                link = item.get("link", "")
                if not title or not link:
                    continue
                summary = clean_html(item.get("summary", item.get("description", "")))
                published = item.get("published", item.get("updated", ""))
                articles.append({
                    "Date": parse_date(published),
                    "Source": source,
                    "Domaine": urlparse(link).netloc.replace("www.", ""),
                    "Titre": title,
                    "Résumé": summary[:2500],
                    "Lien": link,
                })
        except Exception as exc:
            errors.append(f"{source} : {type(exc).__name__}")
    if not articles:
        return pd.DataFrame(), errors
    df = pd.DataFrame(articles).drop_duplicates(subset=["Lien"])
    df["Date_tri"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Thème"] = (df["Titre"].fillna("") + " " + df["Résumé"].fillna("")).apply(topic_for)
    return df.sort_values("Date_tri", ascending=False, na_position="last"), errors


@st.cache_data(ttl=900, show_spinner=False)
def fetch_article_text(url):
    """Best-effort article extraction; many publishers block automated access."""
    if not trafilatura:
        return ""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
        response.raise_for_status()
        return trafilatura.extract(response.text, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol, period="1y"):
    if yf is None or not symbol:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        if "Close" not in df.columns:
            return pd.DataFrame()
        result = df[[date_col, "Close"]].copy()
        result.columns = ["Date", "Cours"]
        result["Cours"] = pd.to_numeric(result["Cours"], errors="coerce")
        return result.dropna(subset=["Cours"])
    except Exception:
        return pd.DataFrame()


def show_news_cards(df, limit=30, full_text=False):
    if df.empty:
        st.info("Aucun article ne correspond aux filtres. Élargissez la période ou changez de thème.")
        return
    for _, article in df.head(limit).iterrows():
        date = article["Date"]
        date_label = date.strftime("%d/%m/%Y %H:%M UTC") if isinstance(date, datetime) else "Date inconnue"
        st.markdown(f"**{article['Titre']}**")
        st.caption(f"{date_label} · {article['Source']} · {article['Thème']}")
        summary = article["Résumé"] or "Le flux RSS ne fournit pas de résumé."
        st.write(summary)
        if full_text:
            with st.expander("Lire le contenu disponible"):
                article_text = fetch_article_text(article["Lien"])
                if article_text:
                    st.write(article_text[:12000])
                else:
                    st.info("Le site ne permet pas d'extraire automatiquement le texte intégral. Le résumé RSS ci-dessus reste disponible.")
        st.markdown(f"[Ouvrir l’article original]({article['Lien']})")
        st.divider()


def newsletter_page():
    st.header("📰 Briefing du matin")
    st.caption("Les articles sont chargés depuis des flux RSS. Le résumé affiché dépend de ce que chaque éditeur fournit.")
    col1, col2, col3 = st.columns([1.3, 1, 1.5])
    with col1:
        topic = st.selectbox("Thème", list(TOPICS.keys()) + ["Autres actualités"], key="news_topic")
    with col2:
        days = st.selectbox("Période", [1, 3, 7, 14, 30], index=2, format_func=lambda x: f"{x} jour(s)", key="news_days")
    with col3:
        keyword = st.text_input("Mot-clé", placeholder="CHF, pétrole, Bitcoin…", key="news_keyword")
    if st.button("🔄 Actualiser les actualités", key="refresh_news"):
        fetch_news.clear()
    with st.spinner("Récupération des flux RSS…"):
        df, errors = fetch_news()
    if errors:
        with st.expander(f"Sources indisponibles ou avec avertissement ({len(errors)})"):
            for error in errors:
                st.write("• " + error)
    if df.empty:
        st.error("Aucune actualité n’a pu être récupérée. Vérifiez la connexion Internet et les flux RSS.")
        return
    cutoff = datetime.utcnow() - timedelta(days=days)
    filtered = df[df["Date"].isna() | (df["Date"] >= cutoff)].copy()
    if topic != "Tous les thèmes":
        filtered = filtered[filtered["Thème"] == topic]
    if keyword.strip():
        mask = (filtered["Titre"].fillna("") + " " + filtered["Résumé"].fillna("") + " " + filtered["Source"].fillna("")).str.contains(re.escape(keyword.strip()), case=False, na=False)
        filtered = filtered[mask]
    st.write(f"**{len(filtered)} article(s)** correspondant aux filtres")
    full_text = st.checkbox("Essayer de récupérer le texte intégral (si le site l’autorise)", value=False)
    show_news_cards(filtered, full_text=full_text)


def market_summary_card(label, symbol, period):
    hist = fetch_history(symbol, period)
    if hist.empty:
        st.warning(f"Données indisponibles pour {label} ({symbol}).")
        return
    last = float(hist["Cours"].iloc[-1])
    previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
    first = float(hist["Cours"].iloc[0])
    daily = (last / previous - 1) * 100 if previous else 0
    total = (last / first - 1) * 100 if first else 0
    st.metric(label, f"{last:,.2f}", f"{daily:+.2f}% dernière séance")
    st.caption(f"Performance sur {PERIODS_LABEL[period]} : {total:+.2f}% · symbole {symbol}")
    fig = px.line(hist, x="Date", y="Cours", title=f"{label} — {PERIODS_LABEL[period]}")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)


PERIODS_LABEL = {v: k for k, v in PERIODS.items()}


def markets_page():
    st.header("📈 Marchés et portefeuille")
    period_label = st.selectbox("Période des graphiques", list(PERIODS.keys()), index=3, key="market_period")
    period = PERIODS[period_label]
    st.subheader("Or et Bitcoin")
    c1, c2 = st.columns(2)
    with c1:
        market_summary_card("Or (USD/once)", "GC=F", period)
    with c2:
        market_summary_card("Bitcoin (USD)", "BTC-USD", period)

    st.subheader("Actions et ETF")
    defaults = ["TotalEnergies", "Exxon Mobil", "Frontline", "FTAI Aviation", "Thales", "Lockheed Martin", "Air Liquide", "Vanguard FTSE All-World (VWRL)"]
    selected = st.multiselect("Actifs à comparer", list(MARKET_SYMBOLS), default=defaults, key="market_assets")
    histories, summary = [], []
    progress = st.progress(0, text="Chargement des cours…")
    for idx, name in enumerate(selected):
        symbol = MARKET_SYMBOLS[name]
        hist = fetch_history(symbol, period)
        progress.progress((idx + 1) / max(len(selected), 1), text=f"Chargement : {name}")
        if hist.empty:
            continue
        first, last = float(hist["Cours"].iloc[0]), float(hist["Cours"].iloc[-1])
        if first == 0:
            continue
        normalized = hist.copy()
        normalized["Performance (%)"] = (normalized["Cours"] / first - 1) * 100
        normalized["Actif"] = name
        histories.append(normalized[["Date", "Performance (%)", "Actif"]])
        summary.append({"Actif": name, "Symbole": symbol, "Dernier cours": round(last, 4), "Performance période (%)": round((last / first - 1) * 100, 2), "Date du cours": hist["Date"].iloc[-1]})
    progress.empty()
    if histories:
        combined = pd.concat(histories, ignore_index=True)
        fig = px.line(combined, x="Date", y="Performance (%)", color="Actif", title=f"Performance comparée — {period_label}")
        fig.update_layout(height=450, yaxis_title="Performance (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else:
        st.info("Aucune série de cours disponible pour cette sélection.")
    st.caption("Les cours peuvent être différés et les devises diffèrent selon les actifs. Vérifiez les symboles ETF auprès de l’émetteur avant toute décision.")


def currencies_page():
    st.header("💱 Change EUR/CHF et EUR/AED")
    period_label = st.selectbox("Période", list(PERIODS.keys()), index=3, key="fx_period")
    period = PERIODS[period_label]
    c1, c2 = st.columns(2)
    for col, (name, symbol) in zip([c1, c2], FX_SYMBOLS.items()):
        with col:
            hist = fetch_history(symbol, period)
            if hist.empty:
                st.warning(f"Cours indisponible pour {name} ({symbol}).")
                continue
            last = float(hist["Cours"].iloc[-1])
            previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
            st.metric(name, f"{last:.4f}", f"{(last / previous - 1) * 100:+.2f}% dernière séance")
            fig = px.line(hist, x="Date", y="Cours", title=f"{name} — {period_label}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    st.caption("L’AED est arrimé au dollar américain. Les cours de change peuvent être indisponibles selon le fournisseur de données.")


def sources_page():
    st.header("🧭 Sources et limites")
    st.markdown("""
    **Sources officielles utiles**
    - [Gouvernement français](https://www.gouvernement.fr/)
    - [Service-Public.fr](https://www.service-public.fr/)
    - [Préfecture de police de Paris](https://www.prefecturedepolice.interieur.gouv.fr/)
    - [Vigilance Météo-France](https://vigilance.meteofrance.fr/)
    - [Banque de France](https://www.banque-france.fr/)
    - [Banque nationale suisse](https://www.snb.ch/)
    - [Agence internationale de l’énergie](https://www.iea.org/)

    **Limites**
    - Les flux RSS peuvent tomber en panne ou ne contenir qu’un titre et un bref résumé.
    - L’extraction du texte intégral dépend des autorisations et de la structure de chaque site.
    - Les cours peuvent être différés, incomplets ou indisponibles.
    - Les mots-clés classent les actualités automatiquement : cette classification peut être imparfaite.
    - Cette application ne détecte pas automatiquement les urgences et ne constitue pas un conseil financier.

    **Bonnes pratiques**
    - Vérifier la date, la zone géographique et la source.
    - Distinguer les faits confirmés des hypothèses.
    - Ne pas prendre de décision financière sur un titre seul.
    - Éviter les achats de panique et surveiller la concentration et la liquidité.
    """)


def home_page():
    st.header("Vue d’ensemble")
    st.write("Un point de départ pour surveiller les marchés, les devises et les actualités prioritaires.")
    now = datetime.now().astimezone()
    st.caption(f"Dernière ouverture : {now.strftime('%d/%m/%Y à %H:%M %Z')}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Marchés")
        for label, symbol in [("Or (USD/once)", "GC=F"), ("Bitcoin (USD)", "BTC-USD"), ("Vanguard FTSE All-World", "VWRL.AS")]:
            hist = fetch_history(symbol, "1mo")
            if hist.empty:
                st.write(f"**{label} :** données indisponibles")
            else:
                last = float(hist["Cours"].iloc[-1])
                prev = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
                st.metric(label, f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}% dernière séance")
    with c2:
        st.subheader("Actualités récentes")
        df, errors = fetch_news()
        if df.empty:
            st.warning("Les flux RSS ne sont pas disponibles. Consultez l’onglet Newsletter.")
        else:
            recent = df.head(6)
            for _, article in recent.iterrows():
                date = article["Date"].strftime("%d/%m %H:%M UTC") if isinstance(article["Date"], datetime) else "Date inconnue"
                st.markdown(f"**[{article['Titre']}]({article['Lien']})**")
                st.caption(f"{date} · {article['Source']} · {article['Thème']}")
                if article["Résumé"]:
                    st.write(article["Résumé"][:350] + ("…" if len(article["Résumé"]) > 350 else ""))
                st.divider()
    st.info("Pour lire davantage, utilisez « Newsletter ». Pour comparer les performances, ouvrez « Marchés & portefeuille ».")



# -------------------------------------------------------------------
# WELS Actions — extensions Market / War / CHF / supply risk
# -------------------------------------------------------------------
WAR_TERMS = ["war", "invasion", "airstrike", "missile", "drone attack", "armed conflict",
             "ceasefire", "nato", "otan", "sanction", "strait of hormuz", "red sea",
             "ukraine", "iran", "israel", "gaza", "taiwan", "russia", "guerre",
             "conflit", "attaque", "escalade", "blocus"]
SUPPLY_TERMS = ["shortage", "stockout", "rationing", "blockade", "strike", "fuel shortage",
                "power outage", "blackout", "gas supply", "medicine shortage", "supply disruption",
                "pénurie", "rupture de stock", "rationnement", "blocage", "grève",
                "carburant", "approvisionnement", "coupure", "médicament"]
CHF_TERMS = ["swiss", "switzerland", "suisse", "franc suisse", "chf", "snb",
             "banque nationale suisse", "eur/chf"]

def classify_tags(title, summary, source):
    text = f"{title} {summary} {source}".lower()
    tags = []
    if any(w in text for w in WAR_TERMS): tags.append("Géopolitique / conflit")
    if any(w in text for w in SUPPLY_TERMS): tags.append("Approvisionnement")
    if any(w in text for w in CHF_TERMS): tags.append("Suisse / CHF")
    if any(w in text for w in TOPICS["Marchés et économie"]): tags.append("Marchés")
    return tags or ["Autres"]

def triage_priority(tags, title, summary):
    text = f"{title} {summary}".lower()
    priority_words = ["invasion", "missile attack", "airstrike", "strait of hormuz closed",
                      "national emergency", "rationing", "pénurie nationale", "national shortage"]
    watch_words = ["escalation", "sanctions", "strike", "blockade", "shortage", "disruption",
                   "escalade", "grève", "blocage", "rupture de stock"]
    if any(w in text for w in priority_words): return "À examiner en priorité"
    if any(w in text for w in watch_words) or "Géopolitique / conflit" in tags: return "À surveiller"
    return "Information"

def enrich_news_frame(df):
    if df.empty: return df
    df = df.copy()
    df["Thèmes"] = df.apply(lambda x: classify_tags(x.get("Titre", ""), x.get("Résumé", ""), x.get("Source", "")), axis=1)
    df["Priorité"] = df.apply(lambda x: triage_priority(x["Thèmes"], x.get("Titre", ""), x.get("Résumé", "")), axis=1)
    return df

def render_alert_articles(df, limit=15):
    if df.empty:
        st.info("Aucun article correspondant sur la période.")
        return
    for _, article in df.head(limit).iterrows():
        dt = article.get("Date")
        date_label = dt.strftime("%d/%m/%Y %H:%M UTC") if isinstance(dt, datetime) else "Date inconnue"
        tags = article.get("Thèmes", [])
        st.markdown(f"**{article.get('Titre', '')}**")
        st.caption(f"{date_label} · {article.get('Source', '')} · {' / '.join(tags)} · {article.get('Priorité', '')}")
        if article.get("Résumé"): st.write(article["Résumé"])
        st.markdown(f"[Lire la source originale]({article.get('Lien', '')})")
        st.divider()

def market_watch_page():
    st.header("📈 Marchés & portefeuille suivi")
    st.caption("Cours indicatifs via Yahoo Finance ; possibles retards, indisponibilités et erreurs de symbole.")
    period_label = st.selectbox("Période", list(PERIODS.keys()), index=3, key="wels_market_period")
    period = PERIODS[period_label]
    watch = dict(MARKET_SYMBOLS)
    watch.update({"Schneider Electric": "SU.PA", "Siemens": "SIE.DE"})
    selected = st.multiselect("Actifs suivis", list(watch), default=[
        "TotalEnergies", "Exxon Mobil", "Vanguard FTSE All-World (VWRL)",
        "Frontline", "FTAI Aviation", "Thales", "Lockheed Martin", "Air Liquide",
        "Schneider Electric", "Siemens"
    ], key="wels_market_assets")
    summary, curves = [], []
    progress = st.progress(0, text="Chargement des cours…")
    for i, name in enumerate(selected):
        hist = fetch_history(watch[name], period)
        progress.progress((i + 1) / max(len(selected), 1), text=f"Chargement : {name}")
        if hist.empty: continue
        first, last = float(hist["Cours"].iloc[0]), float(hist["Cours"].iloc[-1])
        prev = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
        summary.append({"Actif": name, "Symbole": watch[name], "Dernier cours": round(last, 4),
                        "Variation séance (%)": round((last / prev - 1) * 100, 2) if prev else None,
                        "Variation période (%)": round((last / first - 1) * 100, 2) if first else None,
                        "Date": hist["Date"].iloc[-1]})
        if first:
            chart = hist.copy()
            chart["Performance (%)"] = (chart["Cours"] / first - 1) * 100
            chart["Actif"] = name
            curves.append(chart[["Date", "Performance (%)", "Actif"]])
    progress.empty()
    if summary: st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    else: st.warning("Aucun cours disponible pour la sélection.")
    if curves:
        fig = px.line(pd.concat(curves, ignore_index=True), x="Date", y="Performance (%)",
                      color="Actif", title=f"Performance comparée — {period_label}")
        fig.update_layout(height=450, yaxis_title="Performance (%)")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("ETF thématiques à surveiller")
    st.markdown(
        "- **Santé mondiale :** SPDR MSCI World Health Care UCITS ETF (IE00BYTRRB94) ; comparer avec iShares MSCI World Health Care Sector Advanced UCITS ETF. Contrôler TER, encours, spread, réplication et éligibilité au support envisagé.\n"
        "- **Électrification / énergie :** VanEck Electrification and Power Infrastructure UCITS ETF (IE000YYVSM16). Fonds récent : suivre encours, liquidité, concentration et historique limité.\n"
        "- **Schneider Electric / Siemens :** commandes data centres et réseaux, marges, cash-flow, dette, valorisation et chevauchement avec l'ETF."
    )
    st.caption("Les symboles de marché ne remplacent pas la vérification de l'ISIN, de la place de cotation et de la devise auprès de l'émetteur.")

def fx_chf_page():
    st.header("💱 CHF & devises — EUR/CHF et EUR/AED")
    st.caption("Indicateurs de marché, pas d'ordre de conversion automatique.")
    period_label = st.selectbox("Historique", list(PERIODS.keys()), index=3, key="wels_fx_period")
    period = PERIODS[period_label]
    for name, symbol in FX_SYMBOLS.items():
        hist = fetch_history(symbol, period)
        st.subheader(name)
        if hist.empty:
            st.warning(f"Cours indisponible ({symbol}).")
            continue
        last = float(hist["Cours"].iloc[-1])
        prev = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
        first = float(hist["Cours"].iloc[0])
        change = (last / first - 1) * 100 if first else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Dernier cours", f"{last:.4f}")
        c2.metric("Dernière séance", f"{(last / prev - 1) * 100:+.2f}%" if prev else "—")
        c3.metric(f"Variation {period_label}", f"{change:+.2f}%")
        fig = px.line(hist, x="Date", y="Cours", title=f"{name} — {period_label}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        if name == "EUR/CHF":
            if change >= 2.0:
                st.success(f"Signal CHF : EUR/CHF a progressé de {change:+.2f}% sur {period_label}. Un euro achète davantage de CHF qu’au début de la période ; une conversion fractionnée peut être étudiée.")
            elif change <= -2.0:
                st.warning(f"Signal CHF : EUR/CHF a reculé de {change:+.2f}% sur {period_label}. Le CHF s’est renforcé face à l’euro ; convertir des euros en CHF coûte davantage.")
            else:
                st.info(f"Pas de signal de change marqué : variation EUR/CHF de {change:+.2f}% sur {period_label} (seuil indicatif ±2%).")
            st.info("Pour acheter des CHF : une baisse d'EUR/CHF signifie que chaque euro achète moins de francs ; une hausse signifie que l'euro achète davantage. Des conversions fractionnées limitent le risque de chercher le point bas.")
        else:
            st.caption("L'AED est arrimé au dollar américain ; EUR/AED dépend donc largement de l'évolution de l'euro face au dollar.")
    st.subheader("Répartition CHF de référence — scénarios discutés")
    allocations = pd.DataFrame([
        {"Support": "Liquidités CHF", "10k CHF": 20, "25k CHF": 15, "50k CHF": 12},
        {"Support": "Obligations souveraines CHF", "10k CHF": 20, "25k CHF": 15, "50k CHF": 15},
        {"Support": "Obligations entreprises CHF", "10k CHF": 15, "25k CHF": 15, "50k CHF": 13},
        {"Support": "Actions suisses SPI", "10k CHF": 15, "25k CHF": 20, "50k CHF": 20},
        {"Support": "Or CHF", "10k CHF": 15, "25k CHF": 15, "50k CHF": 15},
        {"Support": "Immobilier suisse SIMA", "10k CHF": 10, "25k CHF": 10, "50k CHF": 10},
        {"Support": "ETF mondial coté en CHF", "10k CHF": 5, "25k CHF": 10, "50k CHF": 10},
        {"Support": "Diversification complémentaire", "10k CHF": 0, "25k CHF": 0, "50k CHF": 5},
    ])
    st.dataframe(allocations, use_container_width=True, hide_index=True)
    st.caption("Répartitions de travail, non recommandations d'achat. Vérifier frais, fiscalité française, risque de taux/crédit, liquidité et modalités de détention. SIMA est moins liquide qu'un ETF coté.")

def war_risk_page(df):
    st.header("🌍 Guerres & risques géopolitiques")
    st.caption("Flux médias et GDELT : un titre ou un mot-clé ne confirme pas seul un événement.")
    war = df[df["Thèmes"].apply(lambda x: "Géopolitique / conflit" in x)].copy() if not df.empty else df
    days = st.selectbox("Période", [1, 3, 7, 14, 30], index=2, key="wels_war_days")
    keyword = st.text_input("Zone / mot-clé", placeholder="Ukraine, Iran, mer Rouge…", key="wels_war_keyword")
    if not war.empty:
        cutoff = datetime.utcnow() - timedelta(days=days)
        war = war[war["Date"].isna() | (war["Date"] >= cutoff)]
        if keyword.strip():
            war = war[(war["Titre"].fillna("") + " " + war["Résumé"].fillna("")).str.contains(re.escape(keyword.strip()), case=False, na=False)]
    st.subheader("Transmission potentielle aux actifs")
    st.markdown(
        "- **Pétrole / gaz :** risque d'offre, routes maritimes, primes d'assurance et inflation.\n"
        "- **Transport maritime :** détours, fret et délais ; exposition indirecte de Frontline.\n"
        "- **Or / CHF :** demande refuge possible, sans garantie de hausse.\n"
        "- **Énergie / défense :** réactions potentiellement positives mais dépendantes des valorisations, marges, coûts et marchés.\n"
        "- **Bitcoin :** actif volatil, pas un substitut garanti aux liquidités."
    )
    render_alert_articles(war, 40)

def france_supply_page(df):
    st.header("🇫🇷 France — approvisionnement & continuité")
    st.caption("Les flux repèrent des articles, pas les stocks réels des commerces ou pharmacies. Ne pas extrapoler une pénurie locale depuis une information nationale.")
    supply = df[df["Thèmes"].apply(lambda x: "Approvisionnement" in x)].copy() if not df.empty else df
    days = st.selectbox("Période", [1, 3, 7, 14, 30], index=2, key="wels_supply_days")
    if not supply.empty:
        cutoff = datetime.utcnow() - timedelta(days=days)
        supply = supply[supply["Date"].isna() | (supply["Date"] >= cutoff)]
    render_alert_articles(supply, 30)
    st.markdown("**Précautions**")
    st.markdown("- Confirmer la zone, la durée et l'opérateur auprès de la préfecture ou de l'autorité compétente.\n- Pour les médicaments, suivre les conseils du pharmacien ; éviter le stockage spéculatif.\n- Garder des réserves domestiques raisonnables et tournantes, sans stockage dangereux de carburants.")

def enhanced_newsletter_page(df, errors):
    st.header("📰 Newsletter — veille filtrée")
    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1: days = st.selectbox("Période", [1, 3, 7, 14, 30], index=2, key="wels_news_days")
    with c2: topic = st.selectbox("Thème", ["Tous", "Géopolitique / conflit", "Approvisionnement", "Suisse / CHF", "Marchés"], key="wels_news_topic")
    with c3: keyword = st.text_input("Mot-clé", placeholder="CHF, pétrole, Ukraine…", key="wels_news_keyword")
    if st.button("Actualiser les actualités", key="wels_news_refresh"):
        fetch_news.clear()
        st.rerun()
    selected = df.copy()
    if not selected.empty:
        cutoff = datetime.utcnow() - timedelta(days=days)
        selected = selected[selected["Date"].isna() | (selected["Date"] >= cutoff)]
        if topic != "Tous":
            selected = selected[selected["Thèmes"].apply(lambda tags: topic in tags)]
        if keyword.strip():
            selected = selected[(selected["Titre"].fillna("") + " " + selected["Résumé"].fillna("")).str.contains(re.escape(keyword.strip()), case=False, na=False)]
    st.write(f"**{len(selected)} article(s)**")
    render_alert_articles(selected, 50)
    if errors:
        with st.expander(f"Sources indisponibles ({len(errors)})"):
            for error in errors: st.write("• " + error)

def enhanced_sources_page():
    st.header("Sources, méthode et limites")
    st.markdown(
        "**Flux utilisés**\n- Franceinfo Monde, Économie et Société.\n- BBC World et Business.\n- Boursorama.\n- GDELT : agrégateur de presse, utile pour repérer des signaux mais pas une autorité de confirmation.\n\n"
        "**Sources officielles pour confirmer**\n- [Gouvernement français](https://www.gouvernement.fr/) · [Service-Public](https://www.service-public.fr/) · [Préfecture de police de Paris](https://www.prefecturedepolice.interieur.gouv.fr/)\n"
        "- [Banque nationale suisse](https://www.snb.ch/) · [Banque de France](https://www.banque-france.fr/) · [AIE](https://www.iea.org/)\n\n"
        "**Limites**\n- Les flux RSS et cours peuvent être indisponibles ou différés.\n- Les alertes utilisent des mots-clés : elles ne valident ni gravité, ni localisation, ni causalité.\n- Pas de surveillance intrajournalière continue, pas d'ordre automatique, pas de conseil financier."
    )



def executive_dashboard(df):
    """Compact, article-free executive risk dashboard for the Overview page."""
    st.markdown("""
    <style>
    .risk-shell {background:#050505;border:1px solid #242424;border-radius:18px;padding:18px 16px 14px;margin:4px 0 14px}
    .risk-head {display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
    .risk-title {font-size:21px;font-weight:750;letter-spacing:.2px;color:#eee}
    .risk-sub {font-size:14px;color:#bdbdbd;margin-top:4px}
    .risk-pill {border-radius:20px;padding:5px 11px;font-size:13px;font-weight:700;background:#321706;color:#ff8a32}
    .risk-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:16px}
    .risk-card {border-radius:16px;padding:13px 12px 10px;min-height:118px}
    .risk-card.red {background:#1b0708}.risk-card.amber {background:#1c1507}.risk-card.green {background:#07180e}
    .risk-icon {font-size:21px}.risk-label {font-size:16px;font-weight:700;color:#eee;margin-top:7px}
    .risk-level {font-size:21px;font-weight:800;margin-top:7px}
    .risk-detail {font-size:13px;color:#d0d0d0;margin-top:8px}
    .risk-divider {border-top:1px solid #242424;margin:18px 0 16px}
    @media(max-width:600px){.risk-grid{grid-template-columns:1fr}.risk-card{min-height:95px}}
    </style>
    """, unsafe_allow_html=True)

    def topic_rows(topic):
        if df.empty or "Thèmes" not in df.columns:
            return df
        return df[df["Thèmes"].apply(lambda tags: topic in tags)]

    war = topic_rows("Géopolitique / conflit")
    supply = topic_rows("Approvisionnement")
    market = topic_rows("Marchés")
    chf = topic_rows("Suisse / CHF")

    def level_for(rows, default="MODÉRÉ"):
        if rows.empty: return default
        if "Priorité" in rows.columns and (rows["Priorité"] == "À examiner en priorité").any():
            return "ÉLEVÉ"
        if len(rows) >= 5: return "ÉLEVÉ"
        return "À SURVEILLER" if len(rows) >= 2 else "MODÉRÉ"

    war_level = level_for(war, "NON ÉVALUÉ")
    france_level = level_for(supply, "NON ÉVALUÉ")
    market_level = "VOLATIL" if len(market) >= 4 else ("À SURVEILLER" if len(market) >= 1 else "CALME")
    levels = [war_level, france_level]
    global_level = "RISQUE ÉLEVÉ" if "ÉLEVÉ" in levels else ("VIGILANCE" if any(x == "À SURVEILLER" for x in levels) or market_level == "VOLATIL" else "RISQUE MODÉRÉ")
    stamp = datetime.now().astimezone().strftime("%d %b %Y")
    pill_class = "background:#321706;color:#ff8a32" if global_level == "RISQUE ÉLEVÉ" else ("background:#2a2107;color:#ffc04d" if global_level == "VIGILANCE" else "background:#092313;color:#43d17a")
    st.markdown(f"""
    <div class="risk-shell">
      <div class="risk-head">
        <div><div class="risk-title">GLOBAL RISK MONITOR</div>
        <div class="risk-sub">Daily Intelligence Dashboard · {stamp}</div></div>
        <div class="risk-pill" style="{pill_class}">{global_level}</div>
      </div>
      <div class="risk-grid">
        <div class="risk-card {'red' if war_level == 'ÉLEVÉ' else 'amber'}">
          <div class="risk-icon">◎</div><div class="risk-label">Géopolitique</div>
          <div class="risk-level" style="color:{'#ff5d61' if war_level == 'ÉLEVÉ' else '#ffad48'}">{war_level}</div>
          <div class="risk-detail">{len(war)} signal(aux) détecté(s) · flux presse</div>
        </div>
        <div class="risk-card {'amber' if market_level != 'CALME' else 'green'}">
          <div class="risk-icon">▥</div><div class="risk-label">Marchés</div>
          <div class="risk-level" style="color:{'#ffad48' if market_level != 'CALME' else '#43d17a'}">{market_level}</div>
          <div class="risk-detail">{len(market)} actualité(s) marché · volatilité à confirmer par les cours</div>
        </div>
        <div class="risk-card {'red' if france_level == 'ÉLEVÉ' else 'amber'}">
          <div class="risk-icon">♧</div><div class="risk-label">France</div>
          <div class="risk-level" style="color:{'#ff5d61' if france_level == 'ÉLEVÉ' else '#ffad48'}">{france_level}</div>
          <div class="risk-detail">{len(supply)} signal(aux) approvisionnement · portée locale à confirmer</div>
        </div>
      </div>
      <div class="risk-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Market snapshot")
    instruments = [
        ("Brent", "BZ=F", "$"),
        ("Or", "GC=F", "$"),
        ("EUR / CHF", "EURCHF=X", ""),
        ("EUR / AED", "EURAED=X", ""),
    ]
    cols = st.columns(2)
    for i, (label, symbol, suffix) in enumerate(instruments):
        hist = fetch_history(symbol, "1mo")
        with cols[i % 2]:
            st.caption(label)
            if hist.empty:
                st.metric(label, "Indisponible")
                continue
            last = float(hist["Cours"].iloc[-1])
            previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
            delta = ((last / previous) - 1) * 100 if previous else 0
            if label in ("EUR / CHF", "EUR / AED"):
                value = f"{last:.4f}"
            else:
                value = f"{last:,.2f} {suffix}"
            st.metric(label, value, f"{delta:+.2f} %")
    if not chf.empty:
        st.caption(f"Veille CHF : {len(chf)} actualité(s) détectée(s). Le sens de conversion est présenté dans l'onglet « CHF & devises ».")
    st.caption("Indicateurs issus de flux presse et de cours indicatifs (potentiellement différés). Les niveaux sont des signaux de veille, pas une évaluation officielle du risque.")
    if not df.empty:
        with st.expander("État technique des sources"):
            st.write(f"{len(df)} articles analysés en arrière-plan ; aucun article n'est affiché sur cette page.")

def main():
    # Do not call set_page_config here: app.py configures the parent Streamlit page.
    st.title("📰 WELS Actions — Market, War & CHF Monitor")
    st.caption("Briefing financier et veille des risques patrimoniaux · cache des données : 15 minutes")
    with st.spinner("Collecte des flux d'actualité…"):
        df, errors = fetch_news()
    df = enrich_news_frame(df) if not df.empty else df
    with st.sidebar:
        st.title("WELS Actions")
        page = st.radio("Navigation", [
            "Vue d'ensemble", "Marchés & portefeuille", "Guerres & géopolitique",
            "CHF & devises", "France — approvisionnement", "Newsletter", "Sources & limites"
        ], key="wels_navigation_v2")
        if st.button("Actualiser toutes les données", key="wels_refresh_all"):
            fetch_news.clear()
            fetch_history.clear()
            st.rerun()
        st.caption("Actualités et cours : cache 15 min")
    if page == "Vue d'ensemble":
        executive_dashboard(df)
    elif page == "Marchés & portefeuille": market_watch_page()
    elif page == "Guerres & géopolitique": war_risk_page(df)
    elif page == "CHF & devises": fx_chf_page()
    elif page == "France — approvisionnement": france_supply_page(df)
    elif page == "Newsletter": enhanced_newsletter_page(df, errors)
    else: enhanced_sources_page()

def newsletter_dashboard():
    """Point d’entrée compatible avec app.py sur Streamlit Cloud."""
    main()


if __name__ == "__main__":
    main()
