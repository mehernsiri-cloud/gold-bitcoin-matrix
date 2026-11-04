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
import plotly.graph_objects as go
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
    "Veille France — infrastructures vitales": "https://api.gdeltproject.org/api/v2/doc/doc?query=(France%20OR%20French)%20AND%20(fuel%20shortage%20OR%20fuel%20strike%20OR%20refinery%20OR%20electricity%20outage%20OR%20water%20supply%20OR%20cyberattack%20OR%20port%20blockade%20OR%20transport%20strike%20OR%20food%20shortage%20OR%20hospital%20crisis)&mode=artlist&format=rss&maxrecords=50&sort=datedesc",
    "Veille Émirats — Dubaï": "https://api.gdeltproject.org/api/v2/doc/doc?query=(UAE%20OR%20Emirates%20OR%20Dubai%20OR%20Abu%20Dhabi)%20AND%20(missile%20OR%20drone%20OR%20airspace%20OR%20airport%20closure%20OR%20port%20OR%20Jebel%20Ali%20OR%20water%20OR%20desalination%20OR%20power%20outage%20OR%20banking%20OR%20cyberattack%20OR%20food%20supply)&mode=artlist&format=rss&maxrecords=50&sort=datedesc",
    "Veille Europe — stabilité et économie": "https://api.gdeltproject.org/api/v2/doc/doc?query=(Europe%20OR%20European%20Union%20OR%20EU)%20AND%20(energy%20shortage%20OR%20gas%20supply%20OR%20electricity%20grid%20OR%20banking%20crisis%20OR%20sovereign%20debt%20OR%20capital%20controls%20OR%20mass%20protests%20OR%20general%20strike%20OR%20cyberattack%20OR%20infrastructure%20attack)&mode=artlist&format=rss&maxrecords=50&sort=datedesc",
    "Veille Moyen-Orient — sécurité et approvisionnement": "https://api.gdeltproject.org/api/v2/doc/doc?query=(Middle%20East%20OR%20Iran%20OR%20Israel%20OR%20Gulf%20OR%20Yemen%20OR%20Saudi%20Arabia)%20AND%20(military%20attack%20OR%20missile%20OR%20drone%20OR%20escalation%20OR%20Hormuz%20OR%20Red%20Sea%20OR%20oil%20supply%20OR%20gas%20supply%20OR%20water%20crisis%20OR%20power%20outage%20OR%20civilian%20casualties)&mode=artlist&format=rss&maxrecords=60&sort=datedesc",
    "Veille mondiale — risques systémiques": "https://api.gdeltproject.org/api/v2/doc/doc?query=(global%20OR%20international)%20AND%20(major%20cyberattack%20OR%20pandemic%20OR%20epidemic%20OR%20food%20crisis%20OR%20energy%20crisis%20OR%20shipping%20disruption%20OR%20financial%20crisis%20OR%20nuclear%20incident%20OR%20critical%20infrastructure)&mode=artlist&format=rss&maxrecords=40&sort=datedesc",
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
        st.caption(f"{date_label} · {article['Thème']}")
        summary = article["Résumé"] or "Le flux RSS ne fournit pas de résumé."
        st.write(summary)
        if full_text:
            with st.expander("Lire le contenu disponible"):
                article_text = fetch_article_text(article["Lien"])
                if article_text:
                    st.write(article_text[:12000])
                else:
                    st.info("Le site ne permet pas d'extraire automatiquement le texte intégral. Le résumé RSS ci-dessus reste disponible.")

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

def render_alert_articles(df, limit=15):
    """Do not display article-level content; retain the collected data for risk indicators."""
    if df is None or df.empty:
        st.info("Aucun signal détecté sur la période sélectionnée.")
        return
    st.info(f"{min(len(df), limit)} élément(s) détecté(s). Les articles individuels sont masqués.")

def classify_tags(title, summary, source):
    text = f"{title} {summary} {source}".lower()
    tags = []
    if any(w in text for w in WAR_TERMS):
        tags.append("Géopolitique / conflit")
    if any(w in text for w in SUPPLY_TERMS):
        tags.append("Approvisionnement")
    if any(w in text for w in CHF_TERMS):
        tags.append("Suisse / CHF")
    if any(w in text for w in TOPICS["Marchés et économie"]):
        tags.append("Marchés")
    return tags or ["Autres"]

def triage_priority(tags, title, summary):
    text = f"{title} {summary}".lower()
    priority_words = ["invasion", "missile attack", "airstrike", "strait of hormuz closed",
                      "national emergency", "rationing", "pénurie nationale", "national shortage"]
    watch_words = ["escalation", "sanctions", "strike", "blockade", "shortage", "disruption",
                   "escalade", "grève", "blocage", "rupture de stock"]
    if any(w in text for w in priority_words):
        return "À examiner en priorité"
    if any(w in text for w in watch_words) or "Géopolitique / conflit" in tags:
        return "À surveiller"
    return "Information"

def enrich_news_frame(df):
    if df.empty:
        return df
    df = df.copy()
    df["Thèmes"] = df.apply(
        lambda x: classify_tags(x.get("Titre", ""), x.get("Résumé", ""), x.get("Source", "")),
        axis=1
    )
    df["Priorité"] = df.apply(
        lambda x: triage_priority(x["Thèmes"], x.get("Titre", ""), x.get("Résumé", "")),
        axis=1
    )
    return df


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

    st.divider()
    st.header("Composition du portefeuille réel — ETF et actions")
    st.caption(
        "Vue des positions détenues, exprimée en nombre de lignes et non en poids financier. "
        "Les quantités et valorisations récentes ne sont pas toutes disponibles dans les données enregistrées."
    )

    holdings = pd.DataFrame([
        {"Actif": "Vanguard FTSE All-World UCITS ETF", "Type": "ETF", "Exposition": "Actions mondiales"},
        {"Actif": "WELS — World Health Care", "Type": "ETF", "Exposition": "Santé mondiale"},
        {"Actif": "XDWU — World Utilities", "Type": "ETF", "Exposition": "Services aux collectivités / électricité"},
        {"Actif": "TotalEnergies", "Type": "Action", "Exposition": "Énergie"},
        {"Actif": "ExxonMobil", "Type": "Action", "Exposition": "Énergie"},
        {"Actif": "Air Liquide", "Type": "Action", "Exposition": "Industrie"},
        {"Actif": "FTAI Aviation", "Type": "Action", "Exposition": "Aviation"},
        {"Actif": "Frontline", "Type": "Action", "Exposition": "Transport maritime"},
        {"Actif": "Thales", "Type": "Action", "Exposition": "Défense / technologie"},
        {"Actif": "Lockheed Martin", "Type": "Action", "Exposition": "Défense"},
    ])
    by_type = holdings.groupby("Type").size().reset_index(name="Nombre de positions")
    by_exposure = holdings.groupby("Exposition").size().reset_index(name="Nombre de positions").sort_values("Nombre de positions")

    m1, m2, m3 = st.columns(3)
    m1.metric("Positions détenues", len(holdings))
    m2.metric("ETF", int((holdings["Type"] == "ETF").sum()))
    m3.metric("Actions", int((holdings["Type"] == "Action").sum()))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_type = go.Figure(go.Pie(
            labels=by_type["Type"], values=by_type["Nombre de positions"],
            hole=0.48, textinfo="label+percent+value"
        ))
        fig_type.update_layout(title="ETF vs actions — nombre de lignes", height=390)
        st.plotly_chart(fig_type, use_container_width=True)
    with chart_right:
        fig_exposure = go.Figure(go.Bar(
            x=by_exposure["Nombre de positions"],
            y=by_exposure["Exposition"],
            orientation="h",
            text=by_exposure["Nombre de positions"],
            textposition="auto"
        ))
        fig_exposure.update_layout(
            title="Répartition par type d'exposition — nombre de lignes",
            height=390, xaxis_title="Nombre de positions", yaxis_title=""
        )
        st.plotly_chart(fig_exposure, use_container_width=True)

    st.dataframe(holdings, use_container_width=True, hide_index=True)
    st.caption(
        "Important : ces graphiques décrivent le nombre de lignes détenues, pas leur poids dans le portefeuille. "
        "Pour calculer les poids en euros et les expositions financières, il faudra disposer des quantités ou des valorisations actualisées."
    )

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
    st.caption("Indicateurs de veille : un titre ou un mot-clé ne confirme pas seul un événement.")
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

def qualify_situation(title, summary):
    """Rule-based event qualification. It explains plausible pathways, not confirmed outcomes."""
    text = f"{title} {summary}".lower()
    rules = [
        {
            "id": "fuel_logistics_strike",
            "terms": ["fuel tanker strike", "fuel transport strike", "fuel truck strike", "grève des transporteurs de carburant", "grève des chauffeurs", "blocage des dépôts", "blocage dépôt pétrolier", "raffinerie en grève", "fuel depot blockade", "fuel distribution strike", "oil refinery strike"],
            "event": "Perturbation du transport ou de la distribution de carburants",
            "impact": "Risque de retards de livraison et de tensions locales dans les stations. Une pénurie nationale n'est pas automatique : elle dépend de la durée, des dépôts touchés et des stocks disponibles.",
            "market": "Possible prime de risque sur les produits raffinés et les coûts logistiques ; effet sur le pétrole brut incertain.",
            "action": "Vérifier les dépôts, régions et durée concernés ; suivre les communications préfectorales et opérateurs. Pour le foyer : garder un niveau de carburant raisonnable, sans achats de panique ni stockage dangereux.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "À confirmer selon la zone et la durée"
        },
        {
            "id": "strategic_reserves_release",
            "terms": ["release of strategic reserves", "strategic petroleum reserve release", "release oil reserves", "drawdown of reserves", "mobilisation des réserves stratégiques", "mobilisation des stocks stratégiques", "déblocage des réserves pétrolières", "libération des stocks stratégiques"],
            "event": "Mobilisation de réserves stratégiques de pétrole",
            "impact": "Augmente temporairement l'offre disponible et peut réduire la prime de pénurie ou calmer les prix. L'effet reste limité si la perturbation physique dure ou si les volumes libérés sont faibles.",
            "market": "Signal potentiellement baissier à court terme pour le pétrole, mais pas une garantie de baisse à la pompe.",
            "action": "Vérifier les volumes, le calendrier et les pays participants ; comparer avec l'évolution du Brent et des prix des carburants avant toute décision.",
            "level": "ATTÉNUATION", "score": 0, "polarity": "positive", "confidence": "Effet dépendant des volumes et de la durée"
        },
        {
            "id": "fuel_shortage",
            "terms": ["fuel shortage", "fuel shortages", "petrol shortage", "diesel shortage", "pénurie de carburant", "pénurie d'essence", "pénurie de diesel", "rationnement du carburant", "fuel rationing"],
            "event": "Tension ou pénurie de carburants signalée",
            "impact": "Risque de files d'attente, restrictions locales et difficultés pour les déplacements et la logistique. Distinguer annonce nationale, tension régionale et stations isolées.",
            "market": "Risque de hausse des prix locaux et de tensions sur les produits raffinés.",
            "action": "Vérifier les cartes officielles et annonces préfectorales ; différer les déplacements non essentiels si la zone est touchée ; ne pas constituer de stock de carburant à domicile.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "Portée géographique à vérifier"
        },
        {
            "id": "refinery_outage",
            "terms": ["refinery outage", "refinery shutdown", "raffinerie à l'arrêt", "arrêt de raffinerie", "incendie raffinerie", "refinery fire", "refinery disruption"],
            "event": "Arrêt ou incident dans une raffinerie",
            "impact": "Peut réduire la production de carburants et tendre l'approvisionnement régional, surtout si plusieurs sites ou dépôts sont touchés.",
            "market": "Pression possible sur les marges de raffinage et les prix des produits raffinés ; effet variable sur le brut.",
            "action": "Identifier le site, sa capacité, la durée annoncée et les autres raffineries disponibles dans la zone ; surveiller les communications de l'exploitant.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "Impact dépendant de la capacité arrêtée"
        },
        {
            "id": "power_disruption",
            "terms": ["blackout", "power outage", "electricity outage", "coupure d'électricité", "panne électrique", "réseau électrique", "grid failure", "power grid attack"],
            "event": "Perturbation de l'alimentation électrique",
            "impact": "Selon l'étendue et la durée, peut affecter commerces, télécommunications, paiements, eau et chaîne du froid.",
            "market": "Risque opérationnel pour les entreprises et possible tension sur les prix de l'électricité.",
            "action": "Vérifier la zone et l'opérateur ; recharger téléphones et batteries, conserver de l'eau potable et suivre les consignes officielles. Ne pas utiliser de générateur thermique en intérieur.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "Durée et périmètre à confirmer"
        },
        {
            "id": "shipping_chokepoint",
            "terms": ["strait of hormuz", "bab el-mandeb", "suez canal", "red sea shipping", "shipping disruption", "maritime blockade", "détroit d'ormuz", "mer rouge", "canal de suez", "blocage maritime", "attaque de navire", "tanker attack"],
            "event": "Risque sur une route maritime stratégique",
            "impact": "Détours, délais, hausse des coûts de fret et d'assurance ; risque de perturbation des flux d'énergie et de marchandises si la situation se prolonge.",
            "market": "Prime de risque potentielle sur pétrole, transport maritime et assurance ; forte volatilité possible.",
            "action": "Suivre le trafic réel, les avis aux navires et la durée des perturbations. Éviter de conclure à une pénurie immédiate sans preuve de baisse des flux livrés.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "Risque de transmission, pas rupture automatique"
        },
        {
            "id": "war_escalation",
            "terms": ["invasion", "missile attack", "airstrike", "military escalation", "escalade militaire", "frappes aériennes", "attaque de missiles", "offensive militaire", "mobilisation militaire"],
            "event": "Escalade militaire ou attaque majeure",
            "impact": "Risque de sanctions, perturbation d'infrastructures ou de routes commerciales, hausse de l'incertitude et réactions de marché.",
            "market": "Possible hausse de la volatilité, demande de valeurs refuges et prime sur l'énergie ; réactions non garanties.",
            "action": "Vérifier confirmation et localisation auprès de plusieurs sources fiables ; surveiller sanctions, infrastructures énergétiques et routes commerciales avant d'évaluer l'impact économique.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "Événement et conséquences à confirmer"
        },
        {
            "id": "sanctions_energy",
            "terms": ["oil sanctions", "energy sanctions", "sanctions pétrolières", "sanctions énergétiques", "embargo pétrolier", "oil embargo", "gas embargo"],
            "event": "Sanctions ou embargo visant l'énergie",
            "impact": "Peut déplacer les flux commerciaux, réduire l'offre accessible à certains marchés et augmenter les coûts de transport ou de conformité.",
            "market": "Risque de hausse de la prime de risque énergétique, mais réorientation des flux possible.",
            "action": "Identifier les pays, produits et dates d'entrée en vigueur ; distinguer annonce politique, texte adopté et mise en œuvre effective.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Effet dépendant de l'application"
        },
        {
            "id": "transport_strike",
            "terms": ["transport strike", "rail strike", "port strike", "grève des transports", "grève ferroviaire", "grève portuaire", "blocage des ports", "truckers strike"],
            "event": "Grève ou blocage des transports",
            "impact": "Peut ralentir l'approvisionnement des commerces, des usines et des dépôts ; l'impact dépend des axes, plateformes et stocks disponibles.",
            "market": "Risque de coûts logistiques supplémentaires et de retards de production.",
            "action": "Identifier les réseaux et régions concernés, les dates et les alternatives logistiques ; ne pas extrapoler à une pénurie nationale.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Impact local à documenter"
        },
        {
            "id": "medicine_shortage",
            "terms": ["medicine shortage", "drug shortage", "pénurie de médicaments", "rupture de stock médicament", "pharmaceutical shortage"],
            "event": "Tension sur l'approvisionnement en médicaments",
            "impact": "Peut affecter certains produits ou dosages sans signifier une rupture générale des médicaments.",
            "market": "Risque principalement opérationnel et sanitaire ; impact boursier sectoriel non déductible d'un signal isolé.",
            "action": "Demander conseil au pharmacien et au médecin pour une alternative ; suivre l'ANSM. Ne pas constituer de stock spéculatif.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Médicament et zone à préciser"
        },
    ]
    rules.extend([
        {"id":"airspace_airport","terms":["airspace closure","airport closure","flights suspended","flight cancellations","fermeture de l'espace aérien","aéroport fermé","vols suspendus"],"event":"Fermeture d'espace aérien ou perturbation majeure des vols","impact":"Voyageurs bloqués, déroutements et perturbation du fret aérien ; effet économique local si la fermeture dure.","market":"Surcoûts pour les compagnies, le tourisme et la logistique.","action":"Vérifier les avis de l'aéroport, de la compagnie et des autorités de l'aviation civile avant tout déplacement.","level":"CRITIQUE","score":4,"polarity":"negative","confidence":"Durée et périmètre à confirmer"},
        {"id":"water_desalination","terms":["water shortage","water supply disruption","desalination plant","water rationing","pénurie d'eau","coupure d'eau","usine de dessalement","restriction d'eau"],"event":"Perturbation de l'accès à l'eau potable","impact":"Risque vital en cas de perturbation prolongée, particulièrement dans les territoires dépendants du dessalement ; conséquences sanitaires et économiques.","market":"Impact opérationnel sur ménages, hôtels, industrie et services.","action":"Vérifier les communiqués des opérateurs et autorités ; suivre les consignes concernant l'eau potable.","level":"CRITIQUE","score":4,"polarity":"negative","confidence":"Installation, zone et durée à confirmer"},
        {"id":"cyber_critical","terms":["cyberattack","cyber attack","ransomware attack","critical infrastructure cyber","cyberattaque","attaque informatique","rançongiciel"],"event":"Cyberattaque visant une infrastructure essentielle","impact":"Peut interrompre les services publics, hôpitaux, banques, énergie, eau ou télécommunications ; cible et portée à confirmer.","market":"Coûts de remise en service et interruption d'activité.","action":"Identifier l'opérateur et les services réellement indisponibles ; suivre les communications officielles.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Cible et conséquences à confirmer"},
        {"id":"banking_capital","terms":["bank run","bank failure","bank collapse","capital controls","withdrawal limits","deposit freeze","contrôle des capitaux","limitation des retraits","gel des dépôts","faillite bancaire"],"event":"Crise bancaire ou restriction d'accès aux liquidités","impact":"Risque direct pour l'accès aux dépôts, paiements et transferts ; distinguer rumeur, difficulté d'une banque et mesure officielle.","market":"Stress financier et risque de contagion si plusieurs établissements sont concernés.","action":"Vérifier les communications du régulateur et de la banque centrale ; ne pas agir sur des rumeurs.","level":"CRITIQUE","score":4,"polarity":"negative","confidence":"Confirmation réglementaire requise"},
        {"id":"food_supply","terms":["food shortage","food supply disruption","food rationing","grain export ban","food crisis","pénurie alimentaire","rupture d'approvisionnement alimentaire","rationnement alimentaire"],"event":"Risque majeur sur l'approvisionnement alimentaire","impact":"Hausse des prix ou disponibilité réduite de produits essentiels ; conséquences sociales si la crise dure ou s'étend.","market":"Pression possible sur les prix agricoles et l'inflation.","action":"Vérifier les produits, territoires et volumes concernés auprès des autorités et opérateurs ; éviter les achats de panique.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Produits et territoires à préciser"},
        {"id":"sovereign_financial","terms":["sovereign debt crisis","sovereign default","debt restructuring","crise de la dette souveraine","défaut souverain","restructuration de la dette publique"],"event":"Crise de dette souveraine","impact":"Risque de hausse des taux, mesures budgétaires et pression sur les banques et l'activité ; conséquences dépendantes du pays.","market":"Tension possible sur obligations, banques, euro et financement public.","action":"Suivre décisions gouvernementales, banque centrale et régulateurs ; distinguer dégradation de notation et défaut effectif.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Indicateurs à recouper"},
        {"id":"mass_social_disruption","terms":["nationwide general strike","national general strike","mass riots","state of emergency","grève générale nationale","émeutes généralisées","état d'urgence"],"event":"Troubles sociaux majeurs ou paralysie nationale","impact":"Risque de perturbation des transports, services publics, commerces et accès aux soins ; impact dépendant des villes et de la durée.","market":"Pertes d'activité, retards logistiques et incertitude pour les entreprises.","action":"Vérifier les zones et services touchés ; anticiper les déplacements et suivre les consignes officielles.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Échelle et conséquences à confirmer"}
    ]);
    matches = []
    for rule in rules:
        if any(term in text for term in rule["terms"]):
            matches.append(rule)
    return matches


def geopolitics_supply_dashboard(df):
    """Situation dashboard: event highlights, causal impact chains and immediate actions."""
    st.header("🌍 Géopolitique, énergie & approvisionnement")
    st.caption("Veille ciblée sur les événements ayant des conséquences humaines, vitales, sociales, économiques ou logistiques. Priorité : France et Dubaï/Émirats ; puis Europe et Moyen-Orient.")

    if st.button("🔄 Actualiser les flux de veille", key="wels_geo_refresh"):
        fetch_news.clear()
        st.rerun()

    days = st.selectbox("Fenêtre de veille", [1, 3, 7, 14, 30], index=2, key="wels_geo_supply_days")
    keyword = st.text_input("Filtrer par zone ou sujet (facultatif)", placeholder="France, Iran, carburant, électricité…", key="wels_geo_supply_keyword")

    if df is None or df.empty:
        st.warning("Aucun flux exploitable. Le statut est indéterminé, pas automatiquement rassurant.")
        return

    selected = df.copy()
    if "Date" in selected.columns:
        cutoff = datetime.utcnow() - timedelta(days=days)
        selected = selected[selected["Date"].isna() | (selected["Date"] >= cutoff)]
    if keyword.strip():
        title_text = selected.get("Titre", pd.Series("", index=selected.index)).fillna("").astype(str)
        summary_text = selected.get("Résumé", pd.Series("", index=selected.index)).fillna("").astype(str)
        selected = selected[(title_text + " " + summary_text).str.contains(re.escape(keyword.strip()), case=False, na=False)]

    # Diagnostic éditorial : thèmes et sujets, sans exposer les noms des flux.
    with st.expander("Thèmes et résumés des actualités récupérées", expanded=True):
        st.write(f"**{len(df)} articles récupérés** · **{len(selected)}** dans la période sélectionnée ({days} jour(s)).")
        if not selected.empty:
            theme_terms = {
                "Sécurité, conflits et tensions militaires": ["missile", "drone", "attack", "war", "military", "conflict", "iran", "israel", "gulf", "hormuz", "red sea", "strike", "attaque", "guerre", "conflit"],
                "Énergie, pétrole et carburants": ["oil", "fuel", "petrol", "gas", "energy", "refinery", "electricity", "pétrole", "carburant", "gaz", "énergie", "raffinerie", "électricité"],
                "Eau, alimentation et produits essentiels": ["water", "desalination", "food", "shortage", "supply", "medicine", "eau", "dessal", "alimentation", "pénurie", "approvisionnement", "médicament"],
                "Transport, ports et aviation": ["airport", "airspace", "port", "shipping", "tanker", "flight", "logistics", "aviation", "aéroport", "espace aérien", "maritime", "transport", "logistique"],
                "Économie, banques et dette": ["bank", "banking", "financial", "debt", "inflation", "market", "currency", "liquidity", "capital control", "banque", "dette", "inflation", "finance", "liquidité"],
                "Cyberattaques et infrastructures critiques": ["cyber", "hack", "ransomware", "infrastructure", "power grid", "telecom", "critical", "cyberattaque", "réseau électrique", "infrastructure critique"],
                "Santé et risques sanitaires": ["health", "hospital", "pandemic", "epidemic", "virus", "disease", "medical", "santé", "hôpital", "pandémie", "épidémie"],
                "Stabilité sociale et mouvements majeurs": ["protest", "riot", "unrest", "general strike", "national strike", "emergency", "manifestation", "émeute", "troubles", "grève générale", "état d'urgence"],
            }
            tagged = []
            for _, article in selected.iterrows():
                article_text = f"{article.get('Titre', '')} {article.get('Résumé', '')}".lower()
                matched_themes = [theme for theme, terms in theme_terms.items() if any(term in article_text for term in terms)]
                if not matched_themes:
                    matched_themes = ["Autres sujets à examiner"]
                for theme in matched_themes:
                    tagged.append((theme, article))
            theme_counts = pd.DataFrame([(theme, article) for theme, article in tagged], columns=["Thème", "Article"])
            counts = theme_counts.groupby("Thème").size().sort_values(ascending=False)
            st.markdown("**Répartition par thème**")
            st.dataframe(counts.rename("Articles").reset_index(), hide_index=True, use_container_width=True)
            st.markdown("**Sujets repérés**")
            for theme in counts.index:
                st.markdown(f"**{theme}**")
                seen_titles = set()
                shown = 0
                for _, article in theme_counts[theme_counts["Thème"] == theme]["Article"].items():
                    title = str(article.get("Titre", "")).strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    summary = str(article.get("Résumé", "")).strip()
                    st.markdown(f"- {title}")
                    if summary:
                        st.caption(summary[:350] + ("…" if len(summary) > 350 else ""))
                    shown += 1
                    if shown >= 3:
                        break
        else:
            st.info("Aucun article dans la période ou les filtres sélectionnés.")
        if "Date" in df.columns and df["Date"].notna().any():
            st.caption(f"Actualité la plus récente récupérée : {df['Date'].max().strftime('%d/%m/%Y %H:%M UTC')}")
        st.caption("Les thèmes sont attribués automatiquement à partir des mots-clés des titres et résumés. Ils servent à organiser la veille, pas à confirmer la réalité ou la gravité d'un événement.")

    events = []
    unqualified = []
    for _, row in selected.iterrows():
        title = str(row.get("Titre", "") or "")
        summary = str(row.get("Résumé", "") or "")
        matches = qualify_situation(title, summary)
        if not matches:
            continue
        date_value = row.get("Date")
        date_label = date_value.strftime("%d/%m/%Y %H:%M UTC") if isinstance(date_value, datetime) else "Date non précisée"
        for match in matches:
            events.append({
                **match,
                "title": title,
                "summary": summary,
                "source": str(row.get("Source", "Source non précisée")),
                "url": str(row.get("Lien", "")),
                "date": date_label,
                "region": ("France" if "france" in str(row.get("Source", "")).lower() else "Dubaï / Émirats" if "émirats" in str(row.get("Source", "")).lower() else "Europe" if "europe" in str(row.get("Source", "")).lower() else "Moyen-Orient" if "moyen-orient" in str(row.get("Source", "")).lower() else "International"),
            })
        if len(unqualified) < 12:
            unqualified.append(row)

    # Collapse repeated articles describing the same event family; retain the strongest instance.
    unique_events = {}
    for event in events:
        key = (event["id"], re.sub(r"[^a-z0-9]+", " ", event["title"].lower())[:90])
        if key not in unique_events or event["score"] > unique_events[key]["score"]:
            unique_events[key] = event
    events = sorted(unique_events.values(), key=lambda item: (item["score"], item["date"]), reverse=True)

    negative_events = [event for event in events if event["polarity"] == "negative"]
    positive_events = [event for event in events if event["polarity"] == "positive"]
    high_events = [event for event in negative_events if event["score"] >= 3]
    medium_events = [event for event in negative_events if event["score"] == 2]
    distinct_negative_families = {event["id"] for event in negative_events}
    if len(distinct_negative_families) >= 2:
        overall = "TENSIONS MULTIPLES"
        overall_detail = "Plusieurs types d'événements défavorables sont détectés : vérifier les cumuls et les effets en chaîne."
    elif high_events:
        overall = "ALERTE — SIGNAL FORT"
        overall_detail = "Au moins un événement potentiellement perturbateur nécessite une vérification rapide."
    elif medium_events:
        overall = "VIGILANCE RENFORCÉE"
        overall_detail = "Des perturbations possibles sont signalées, sans preuve suffisante d'une rupture majeure."
    elif positive_events:
        overall = "MESURES D'ATTÉNUATION DÉTECTÉES"
        overall_detail = "Des mesures susceptibles de réduire la tension sont signalées ; leur effet réel reste à vérifier."
    elif events:
        overall = "VEILLE"
        overall_detail = "Des événements sont détectés, sans qualification de risque fort par les règles actuelles."
    else:
        overall = "SITUATION NON QUALIFIÉE"
        overall_detail = "Aucun événement couvert par les règles n'a été reconnu. Cela ne signifie pas qu'il n'existe aucun risque."

    st.markdown("### Situation générale")
    st.subheader(overall)
    st.write(overall_detail)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Événements qualifiés", len(events))
    k2.metric("Signaux défavorables", len(negative_events))
    k3.metric("Signaux forts", len(high_events))
    k4.metric("Mesures d'atténuation", len(positive_events))

    if len(distinct_negative_families) >= 2:
        st.error("CUMUL DE RISQUES : plusieurs familles d'événements défavorables sont présentes. Examiner les dépendances communes : énergie, transport, logistique et disponibilité.")
    elif high_events:
        st.warning("Un signal à impact potentiellement élevé est présent. Vérifier sa réalité, son périmètre et sa durée avant d'agir.")

    st.markdown("### Événements à impact concret — faits et conséquences")
    if not events:
        st.info("Aucun événement n'a pu être qualifié automatiquement sur cette période. Les nouvelles non reconnues ne sont pas interprétées comme positives.")
    for index, event in enumerate(events[:15], start=1):
        with st.expander(f"{'🟢' if event['polarity'] == 'positive' else ('🔴' if event['score'] >= 3 else '🟠')} {event['event']} — {event['level']}", expanded=(index <= 3)):
            st.markdown(f"**Actualité :** {event['title']}")
            st.caption(f"{event.get('region', 'Zone à confirmer')} · {event['source']} · {event['date']} · Qualification : {event['confidence']}")
            if event["summary"]:
                st.write(event["summary"][:700])
            st.markdown(f"**Ce que cela peut entraîner :** {event['impact']}")
            st.markdown(f"**Transmission économique / marchés :** {event['market']}")
            st.markdown(f"**Plan d'action immédiat :** {event['action']}")
            if event["url"].startswith("http"):
                st.markdown(f"[Consulter la source]({event['url']})")

    if positive_events:
        st.markdown("### Signaux qui peuvent réduire la tension")
        for event in positive_events:
            st.success(f"**{event['event']}** — {event['impact']}")

    if negative_events:
        st.markdown("### Plan d'action prioritaire")
        actions = []
        if any(event["id"] in {"fuel_logistics_strike", "fuel_shortage", "refinery_outage"} for event in negative_events):
            actions.append("Carburants : vérifier les départements et dépôts réellement touchés, les stocks communiqués et la durée prévue ; maintenir une réserve d'usage normale, sans stockage excessif.")
        if any(event["id"] in {"power_disruption"} for event in negative_events):
            actions.append("Électricité : recharger les équipements essentiels, vérifier les informations du gestionnaire de réseau et appliquer les consignes officielles.")
        if any(event["id"] in {"shipping_chokepoint", "sanctions_energy"} for event in negative_events):
            actions.append("Énergie et transport : surveiller les flux physiques, les annonces réglementaires et le Brent ; ne pas assimiler une hausse de risque à une pénurie déjà constatée.")
        if any(event["id"] in {"medicine_shortage"} for event in negative_events):
            actions.append("Médicaments : vérifier le produit précis avec le pharmacien et les informations de l'ANSM ; ne pas faire de réserves non nécessaires.")
        if any(event["id"] in {"water_desalination"} for event in negative_events):
            actions.append("Eau potable : vérifier les consignes de l'opérateur et des autorités locales ; appliquer les recommandations officielles de réserve si une perturbation est confirmée.")
        if any(event["id"] in {"airspace_airport"} for event in negative_events):
            actions.append("Voyages : contrôler les statuts de vol et les avis des aéroports/compagnies avant tout déplacement, en particulier dans le Golfe.")
        if any(event["id"] in {"cyber_critical"} for event in negative_events):
            actions.append("Services essentiels : identifier les opérateurs touchés et les services indisponibles ; suivre les canaux officiels, sans relayer de rumeurs.")
        if any(event["id"] in {"banking_capital", "sovereign_financial"} for event in negative_events):
            actions.append("Banques et liquidités : distinguer rumeurs, tensions de marché et mesures officielles ; vérifier les communications des autorités financières.")
        if any(event["id"] in {"food_supply"} for event in negative_events):
            actions.append("Alimentation : vérifier les produits et zones réellement touchés ; conserver des réserves domestiques raisonnables, sans achats de panique.")
        if any(event["id"] in {"mass_social_disruption"} for event in negative_events):
            actions.append("Continuité quotidienne : vérifier les transports, écoles, services de santé et consignes locales avant les déplacements.")
        if any(event["id"] in {"war_escalation", "military_direct_threat"} for event in negative_events):
            actions.append("Géopolitique : recouper les faits auprès de sources fiables et suivre les avis officiels ; distinguer risque régional et impact direct en France.")
        for action in actions:
            st.markdown(f"- {action}")
        st.caption("Ces actions sont des mesures de vérification et de préparation proportionnées, pas des consignes d'achat ni des prédictions.")

    st.caption("Les articles hors périmètre sont volontairement exclus de cette page afin de limiter le bruit. Une absence d'événement qualifié ne signifie pas une absence de risque : les flux et les règles de détection ont des limites.")


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


def executive_dashboard(df):
    """Article-free executive dashboard: risk indicators and four market quotes."""
    st.markdown("""
    <style>
    .risk-shell{background:#050505;border:1px solid #242424;border-radius:18px;padding:18px 16px 14px;margin:4px 0 14px}
    .risk-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
    .risk-title{font-size:21px;font-weight:750;color:#eee}
    .risk-sub{font-size:14px;color:#bdbdbd;margin-top:4px}
    .risk-pill{border-radius:20px;padding:5px 11px;font-size:13px;font-weight:700}
    .risk-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:16px}
    .risk-card{border-radius:16px;padding:13px 12px 10px;min-height:118px}
    .risk-card.red{background:#1b0708}.risk-card.amber{background:#1c1507}.risk-card.green{background:#07180e}
    .risk-icon{font-size:21px}.risk-label{font-size:16px;font-weight:700;color:#eee;margin-top:7px}
    .risk-level{font-size:21px;font-weight:800;margin-top:7px}
    .risk-detail{font-size:13px;color:#d0d0d0;margin-top:8px}
    @media(max-width:600px){.risk-grid{grid-template-columns:1fr}.risk-card{min-height:95px}}
    </style>
    """, unsafe_allow_html=True)

    def rows_for(topic):
        if df.empty or "Thèmes" not in df.columns:
            return df
        return df[df["Thèmes"].apply(lambda tags: topic in tags)]

    war = rows_for("Géopolitique / conflit")
    supply = rows_for("Approvisionnement")
    chf = rows_for("Suisse / CHF")

    def signal_level(rows):
        if rows.empty:
            return "AUCUN SIGNAL"
        if "Priorité" in rows.columns and (rows["Priorité"] == "À examiner en priorité").any():
            return "ÉLEVÉ"
        if len(rows) >= 4:
            return "ÉLEVÉ"
        if len(rows) >= 1:
            return "À SURVEILLER"
        return "AUCUN SIGNAL"

    war_level = signal_level(war)
    france_level = signal_level(supply)

    # Market risk is based on observed daily moves, not the number of news articles.
    market_moves = {}
    for label, symbol in [("Brent", "BZ=F"), ("Or", "GC=F"), ("S&P 500", "^GSPC")]:
        hist = fetch_history(symbol, "1mo")
        if not hist.empty and len(hist) > 1:
            last = float(hist["Cours"].iloc[-1])
            prev = float(hist["Cours"].iloc[-2])
            market_moves[label] = (last / prev - 1) * 100 if prev else 0.0
    volatile = any(abs(v) >= 2.0 for v in market_moves.values())
    market_level = "VOLATILE" if volatile else ("À SURVEILLER" if market_moves else "INDISPONIBLE")

    # A missing feed is not interpreted as a zero-risk situation.
    global_high = war_level == "ÉLEVÉ" or france_level == "ÉLEVÉ"
    global_watch = war_level == "À SURVEILLER" or france_level == "À SURVEILLER" or market_level == "VOLATILE"
    if global_high:
        global_level, pill_style = "RISQUE ÉLEVÉ", "background:#321706;color:#ff8a32"
    elif global_watch:
        global_level, pill_style = "VIGILANCE", "background:#2a2107;color:#ffc04d"
    elif market_level == "INDISPONIBLE":
        global_level, pill_style = "DONNÉES INCOMPLÈTES", "background:#252525;color:#ddd"
    else:
        global_level, pill_style = "RISQUE MODÉRÉ", "background:#092313;color:#43d17a"

    stamp = datetime.now().astimezone().strftime("%d %b %Y")
    def card_class(level):
        return "red" if level == "ÉLEVÉ" else ("green" if level == "AUCUN SIGNAL" else "amber")
    def level_color(level):
        return "#ff5d61" if level == "ÉLEVÉ" else ("#43d17a" if level == "AUCUN SIGNAL" else "#ffad48")

    st.markdown(f"""
    <div class="risk-shell">
      <div class="risk-head">
        <div><div class="risk-title">GLOBAL RISK MONITOR</div>
        <div class="risk-sub">Daily Intelligence Dashboard · {stamp}</div></div>
        <div class="risk-pill" style="{pill_style}">{global_level}</div>
      </div>
      <div class="risk-grid">
        <div class="risk-card {card_class(war_level)}">
          <div class="risk-icon">◎</div><div class="risk-label">Géopolitique</div>
          <div class="risk-level" style="color:{level_color(war_level)}">{war_level}</div>
          <div class="risk-detail">Conflits · sanctions · routes maritimes</div>
        </div>
        <div class="risk-card {'amber' if market_level == 'VOLATILE' else 'green'}">
          <div class="risk-icon">▥</div><div class="risk-label">Marchés</div>
          <div class="risk-level" style="color:{'#ffad48' if market_level == 'VOLATILE' else '#43d17a'}">{market_level}</div>
          <div class="risk-detail">Pétrole · Or · Actions</div>
        </div>
        <div class="risk-card {card_class(france_level)}">
          <div class="risk-icon">♧</div><div class="risk-label">France</div>
          <div class="risk-level" style="color:{level_color(france_level)}">{france_level}</div>
          <div class="risk-detail">Carburant · Énergie · Approvisionnement</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Market snapshot")
    instruments = [
        ("Brent", "BZ=F", "USD"),
        ("Or", "GC=F", "USD"),
        ("EUR / CHF", "EURCHF=X", ""),
        ("EUR / AED", "EURAED=X", ""),
    ]
    cols = st.columns(2)
    for i, (label, symbol, unit) in enumerate(instruments):
        hist = fetch_history(symbol, "1mo")
        with cols[i % 2]:
            st.caption(label)
            if hist.empty:
                st.metric(label, "Indisponible")
                continue
            last = float(hist["Cours"].iloc[-1])
            previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
            delta = (last / previous - 1) * 100 if previous else 0
            value = f"{last:.4f}" if not unit else f"{last:,.2f} {unit}"
            st.metric(label, value, f"{delta:+.2f} %")

    if not chf.empty:
        st.caption(f"CHF : {len(chf)} signal(aux) de presse détecté(s) — voir l'onglet CHF & devises.")
    st.caption("Cours indicatifs potentiellement différés. Les niveaux de risque sont des indicateurs de veille, pas une évaluation officielle.")

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
            "Vue d'ensemble", "Marchés & portefeuille", "Géopolitique & approvisionnement",
            "CHF & devises"
        ], key="wels_navigation_v3")
        if st.button("Actualiser toutes les données", key="wels_refresh_all"):
            fetch_news.clear()
            fetch_history.clear()
            st.rerun()
        st.caption("Actualités et cours : cache 15 min")
    if page == "Vue d'ensemble":
        executive_dashboard(df)
    elif page == "Marchés & portefeuille": market_watch_page()
    elif page == "Géopolitique & approvisionnement": geopolitics_supply_dashboard(df)
    elif page == "CHF & devises": fx_chf_page()

def newsletter_dashboard():
    """Point d’entrée compatible avec app.py sur Streamlit Cloud."""
    main()


if __name__ == "__main__":
    main()
