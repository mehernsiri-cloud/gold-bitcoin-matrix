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
    "Reuters World": "https://feeds.reuters.com/Reuters/worldNews",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
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


def newsletter_dashboard():
    st.set_page_config(page_title=APP_TITLE, page_icon="📰", layout="wide")
    st.title("📰 WELS Actions")
    st.caption("Newsletter financière, suivi des marchés et veille des risques patrimoniaux")
    if feedparser is None:
        st.error("Le module feedparser est absent. Installez les dépendances avec : pip install -r requirements.txt")
    with st.sidebar:
        st.title("WELS Actions")
        page = st.radio("Navigation", ["🏠 Vue d’ensemble", "📰 Newsletter", "📈 Marchés & portefeuille", "💱 Devises", "⚠️ Sources & risques"], key="navigation")
        if st.button("🔄 Vider le cache et actualiser"):
            fetch_news.clear()
            fetch_history.clear()
            fetch_article_text.clear()
            st.rerun()
        st.caption("Cache des données : 15 minutes")
    if page == "🏠 Vue d’ensemble":
        home_page()
    elif page == "📰 Newsletter":
        newsletter_page()
    elif page == "📈 Marchés & portefeuille":
        markets_page()
    elif page == "💱 Devises":
        currencies_page()
    else:
        sources_page()


if __name__ == "__newsletter_dashboard__":
    newsletter_dashboard()
