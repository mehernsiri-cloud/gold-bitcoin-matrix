
# newsletter_app.py
# WELS Actions — Newsletter financiere et tableau de bord
#
# Installation :
# pip install streamlit pandas requests plotly feedparser yfinance
#
# Lancement :
# streamlit run newsletter_app.py

import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

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


# =========================================================
# 1. CONFIGURATION
# =========================================================

APP_TITLE = "WELS Actions - Newsletter"

RSS_FEEDS = {
    "France Info Economie":
        "https://www.francetvinfo.fr/economie.rss",
    "France Info Monde":
        "https://www.francetvinfo.fr/monde.rss",
    "France Info Societe":
        "https://www.francetvinfo.fr/societe.rss",
    "Boursorama":
        "https://www.boursorama.com/rss/actualites/",
    "Reuters World":
        "https://feeds.reuters.com/Reuters/worldNews",
    "Reuters Business":
        "https://feeds.reuters.com/reuters/businessNews",
}

TOPICS = {
    "Tous les themes": [],
    "Marches et economie": [
        "market", "markets", "bourse", "marche", "actions",
        "inflation", "fed", "bce", "banque centrale", "taux",
        "obligation", "economy", "economie", "recession", "dette"
    ],
    "Or et Bitcoin": [
        "gold", "or ", "bitcoin", "crypto", "btc", "xau"
    ],
    "Energie et transport maritime": [
        "oil", "petrole", "energy", "energie", "gas", "gaz",
        "shipping", "maritime", "tanker", "totalenergies", "exxon"
    ],
    "Sante": [
        "health", "sante", "pharma", "medical", "biotech",
        "healthcare", "medicament"
    ],
    "Electricite et utilities": [
        "electricity", "electricite", "utilities", "grid",
        "power", "data centre", "data center",
        "schneider", "siemens"
    ],
    "Defense et geopolitique": [
        "defence", "defense", "guerre", "war", "military",
        "militaire", "conflict", "conflit", "otan", "nato",
        "sanction", "geopolitique"
    ],
    "France : approvisionnement": [
        "france", "ile-de-france", "carburant", "fuel",
        "water", "eau", "food", "alimentation", "strike",
        "greve", "blocage", "supply", "approvisionnement",
        "medicament", "baby", "electricite"
    ],
    "Suisse et devises": [
        "swiss", "suisse", "switzerland", "franc suisse",
        "chf", "eur/chf", "eur/aed", "banque nationale suisse"
    ],
}

# Symboles indicatifs Yahoo Finance.
# Certains symboles peuvent changer ou ne pas etre disponibles.
MARKET_SYMBOLS = {
    "Or (USD/once)": "GC=F",
    "Bitcoin (USD)": "BTC-USD",
    "TotalEnergies": "TTE.PA",
    "Exxon Mobil": "XOM",
    "ETF Vanguard FTSE All-World": "VWRL.AS",
    "Frontline": "FRO",
    "FTAI Aviation": "FTAI",
    "Thales": "HO.PA",
    "Lockheed Martin": "LMT",
    "Air Liquide": "AI.PA",
    "ETF utilities XDWU (a verifier)": "XDWU.L",
    "Schneider Electric": "SU.PA",
    "Siemens": "SIE.DE",
    "S&P 500": "^GSPC",
    "ETF MSCI World ACWI": "ACWI",
}

FX_SYMBOLS = {
    "EUR/CHF": "EURCHF=X",
    "EUR/AED": "EURAED=X",
}


# =========================================================
# 2. OUTILS
# =========================================================

def clean_html(value):
    """Nettoie les balises HTML et les espaces."""
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_date(value):
    """Convertit une date RSS en datetime."""
    if not value:
        return None

    try:
        result = parsedate_to_datetime(value)

        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)

        return result.astimezone(timezone.utc).replace(tzinfo=None)

    except Exception:
        try:
            return pd.to_datetime(
                value, utc=True
            ).tz_convert(None).to_pydatetime()
        except Exception:
            return None


def get_domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


# =========================================================
# 3. RECUPERATION DES ACTUALITES RSS
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(max_per_feed=30):
    """
    Recupere les actualites depuis les flux RSS.
    Le cache dure 15 minutes.
    """

    if feedparser is None:
        return pd.DataFrame()

    articles = []

    for source, url in RSS_FEEDS.items():

        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 WELSNewsletter/1.0"
                },
                timeout=12,
            )

            response.raise_for_status()
            parsed = feedparser.parse(response.content)

            for item in parsed.entries[:max_per_feed]:

                title = clean_html(item.get("title", ""))
                link = item.get("link", "")

                if not title or not link:
                    continue

                summary = clean_html(
                    item.get(
                        "summary",
                        item.get("description", "")
                    )
                )

                date = parse_date(
                    item.get(
                        "published",
                        item.get("updated", "")
                    )
                )

                articles.append({
                    "Date": date,
                    "Source": source,
                    "Domaine": get_domain(link),
                    "Titre": title,
                    "Resume": summary[:1200],
                    "Lien": link,
                })

        except Exception:
            # Un flux en erreur ne bloque pas les autres.
            continue

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=["Lien"])

    df["Date_tri"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    return df.sort_values(
        "Date_tri",
        ascending=False,
        na_position="last",
    )


def identify_topic(text):
    """Classe les articles selon les mots-cles."""
    text = (text or "").lower()

    for topic, keywords in TOPICS.items():

        if topic == "Tous les themes":
            continue

        if any(word in text for word in keywords):
            return topic

    return "Autres actualites"


def filter_news(df, topic, days, keyword):
    if df.empty:
        return df.copy()

    result = df.copy()

    cutoff = datetime.utcnow() - timedelta(days=days)

    result = result[
        result["Date"].isna()
        | (result["Date"] >= cutoff)
    ]

    result["Theme"] = (
        result["Titre"].fillna("")
        + " "
        + result["Resume"].fillna("")
    ).apply(identify_topic)

    if topic != "Tous les themes":
        result = result[result["Theme"] == topic]

    if keyword.strip():
        search_text = (
            result["Titre"].fillna("")
            + " "
            + result["Resume"].fillna("")
            + " "
            + result["Source"].fillna("")
        ).str.lower()

        result = result[
            search_text.str.contains(
                re.escape(keyword.strip().lower()),
                na=False,
            )
        ]

    return result.sort_values(
        "Date_tri",
        ascending=False,
        na_position="last",
    )


def display_news(df, limit=50):
    """Affiche les articles sous forme de cartes."""

    if df.empty:
        st.info(
            "Aucune actualite trouvee pour ces filtres. "
            "Essayez une autre periode ou un autre theme."
        )
        return

    for _, article in df.head(limit).iterrows():

        date = article["Date"]

        if isinstance(date, datetime):
            date_text = date.strftime("%d/%m/%Y %H:%M UTC")
        else:
            date_text = "Date non disponible"

        title = html.escape(str(article["Titre"]))
        summary = html.escape(str(article["Resume"]))
        source = html.escape(str(article["Source"]))
        theme = html.escape(str(article["Theme"]))
        link = html.escape(str(article["Lien"]), quote=True)

        st.markdown(
            f"""
            <div style="
                border:1px solid #d5dce5;
                border-radius:10px;
                padding:14px;
                margin:10px 0;
                background-color:rgba(128,128,128,0.06);
            ">
                <div style="font-size:12px;color:gray">
                    {date_text} | {source} | {theme}
                </div>
                <div style="font-size:17px;font-weight:650;
                            margin:8px 0">
                    <a href="{link}" target="_blank"
                       style="text-decoration:none">
                        {title}
                    </a>
                </div>
                <div style="font-size:14px;line-height:1.5">
                    {summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 4. HISTORIQUE DES COURS
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol, period="1y"):
    """Recupere l'historique quotidien d'un actif."""

    if yf is None or not symbol:
        return pd.DataFrame()

    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Compatibilite avec les colonnes MultiIndex.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        date_col = (
            "Date" if "Date" in df.columns
            else df.columns[0]
        )

        if "Close" not in df.columns:
            return pd.DataFrame()

        result = df[[date_col, "Close"]].copy()
        result.columns = ["Date", "Cours"]

        result["Cours"] = pd.to_numeric(
            result["Cours"], errors="coerce"
        )

        return result.dropna(subset=["Cours"])

    except Exception:
        return pd.DataFrame()


def get_period():
    label = st.selectbox(
        "Periode des graphiques",
        ["1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans"],
        index=3,
    )

    mapping = {
        "1 mois": "1mo",
        "3 mois": "3mo",
        "6 mois": "6mo",
        "1 an": "1y",
        "2 ans": "2y",
        "5 ans": "5y",
    }

    return label, mapping[label]


# =========================================================
# 5. PAGE NEWSLETTER
# =========================================================

def newsletter_page():

    st.header("📰 Briefing du matin")

    now = datetime.now().astimezone()

    st.caption(
        "Consultation : "
        + now.strftime("%d/%m/%Y %H:%M %Z")
    )

    st.info(
        "Les titres et resumes proviennent de flux RSS. "
        "Ils ne constituent pas a eux seuls une verification "
        "des faits ni une recommandation d'investissement."
    )

    col1, col2, col3 = st.columns([1.3, 1, 1.5])

    with col1:
        topic = st.selectbox(
            "Theme",
            list(TOPICS.keys()) + ["Autres actualites"],
        )

    with col2:
        days = st.selectbox(
            "Periode",
            [1, 3, 7, 14, 30],
            index=2,
            format_func=lambda x: f"{x} jour(s)",
        )

    with col3:
        keyword = st.text_input(
            "Mot-cle",
            placeholder="CHF, petrole, Bitcoin...",
        )

    if feedparser is None:
        st.error(
            "Module feedparser manquant. "
            "Ajoutez-le au fichier requirements.txt."
        )
        return

    with st.spinner("Chargement des actualites..."):
        df = fetch_news()

    if df.empty:
        st.warning(
            "Aucune actualite disponible. Les flux RSS peuvent "
            "etre indisponibles ou bloquer les requetes."
        )
    else:
        filtered = filter_news(
            df, topic, days, keyword
        )

        st.write(
            f"**{len(filtered)} article(s)** correspondant aux filtres"
        )

        display_news(filtered)

    if st.button("🔄 Actualiser les actualites"):
        fetch_news.clear()
        st.rerun()


# =========================================================
# 6. PAGE MARCHES ET PORTEFEUILLE
# =========================================================

def markets_page():

    st.header("📈 Marches et portefeuille")

    period_label, period = get_period()

    st.subheader("Or et Bitcoin")

    col1, col2 = st.columns(2)

    for col, label, symbol in [
        (col1, "Or - USD par once", "GC=F"),
        (col2, "Bitcoin - USD", "BTC-USD"),
    ]:

        with col:
            hist = fetch_history(symbol, period)

            if hist.empty:
                st.warning(f"Donnees indisponibles : {label}")
                continue

            last = float(hist["Cours"].iloc[-1])

            previous = (
                float(hist["Cours"].iloc[-2])
                if len(hist) > 1 else last
            )

            change = (
                (last / previous - 1) * 100
                if previous else 0
            )

            st.metric(
                label,
                f"{last:,.2f}",
                f"{change:+.2f}% vs derniere seance",
            )

            fig = px.line(
                hist,
                x="Date",
                y="Cours",
                title=label,
            )

            fig.update_layout(height=300)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.divider()
    st.subheader("Actions et ETF")

    selected = st.multiselect(
        "Selectionner les actifs a comparer",
        options=list(MARKET_SYMBOLS.keys()),
        default=[
            "TotalEnergies",
            "Exxon Mobil",
            "Frontline",
            "FTAI Aviation",
            "Thales",
            "Lockheed Martin",
            "Air Liquide",
            "ETF Vanguard FTSE All-World",
        ],
    )

    histories = []
    summary = []

    for name in selected:

        symbol = MARKET_SYMBOLS[name]
        hist = fetch_history(symbol, period)

        if hist.empty:
            continue

        first = float(hist["Cours"].iloc[0])
        last = float(hist["Cours"].iloc[-1])

        if first == 0:
            continue

        hist = hist.copy()
        hist["Performance"] = hist["Cours"] / first * 100
        hist["Actif"] = name

        histories.append(
            hist[["Date", "Performance", "Actif"]]
        )

        summary.append({
            "Actif": name,
            "Symbole": symbol,
            "Dernier cours": last,
            "Variation (%)": (last / first - 1) * 100,
            "Date du cours": hist["Date"].iloc[-1],
        })

    if histories:

        combined = pd.concat(
            histories, ignore_index=True
        )

        fig = px.line(
            combined,
            x="Date",
            y="Performance",
            color="Actif",
            title=f"Performance comparee - {period_label} (base 100)",
        )

        fig.update_layout(
            height=450,
            yaxis_title="Indice base 100",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "Aucune serie disponible pour cette selection."
        )

    st.caption(
        "Les cours peuvent etre differes et exprimes dans "
        "des devises differentes. Les performances passees "
        "ne prejugent pas des performances futures. "
        "Verifiez les symboles des ETF avant toute utilisation."
    )


# =========================================================
# 7. PAGE DEVISES
# =========================================================

def currencies_page():

    st.header("💱 Change EUR/CHF et EUR/AED")

    label, period = get_period()

    col1, col2 = st.columns(2)

    for col, (name, symbol) in zip(
        [col1, col2], FX_SYMBOLS.items()
    ):

        with col:

            hist = fetch_history(symbol, period)

            if hist.empty:
                st.warning(
                    f"Cours indisponible pour {name}. "
                    "Le fournisseur ne prend peut-etre pas "
                    "en charge ce symbole."
                )
                continue

            last = float(hist["Cours"].iloc[-1])

            st.metric(name, f"{last:.4f}")

            fig = px.line(
                hist,
                x="Date",
                y="Cours",
                title=f"{name} - {label}",
            )

            fig.update_layout(height=300)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.caption(
        "L'AED est arrime au dollar americain. "
        "Si EUR/AED n'est pas disponible, il faut obtenir "
        "EUR/USD et USD/AED aupres de sources fiables."
    )


# =========================================================
# 8. SOURCES ET RISQUES
# =========================================================

def sources_page():

    st.header("🧭 Sources officielles et risques")

    st.markdown("""
    Consulte les sources officielles pour confirmer les informations :

    - [Gouvernement francais](https://www.gouvernement.fr/)
    - [Prefecture de police de Paris](https://www.prefecturedepolice.interieur.gouv.fr/)
    - [Service-public.fr](https://www.service-public.fr/)
    - [Vigilance Meteo-France](https://vigilance.meteofrance.fr/)
    - [Banque de France](https://www.banque-france.fr/)
    - [Banque nationale suisse](https://www.snb.ch/)
    - [Agence internationale de l'energie](https://www.iea.org/)
    """)

    st.warning(
        "Cette page ne detecte pas automatiquement les urgences "
        "et ne verifie pas independamment chaque information. "
        "Pour les alertes de securite et d'approvisionnement, "
        "privilegiez les communications officielles locales."
    )

    st.subheader("Principes de suivi")

    st.markdown("""
    - Distinguer les faits confirmes des hypotheses.
    - Verifier la date, la zone geographique et la source.
    - Ne pas prendre de decision financiere sur un titre seul.
    - Eviter les achats de panique et les changements brusques.
    - Suivre les risques de concentration et de liquidite.
    """)


# =========================================================
# 9. APPLICATION PRINCIPALE
# =========================================================

def newsletter_dashboard():

    st.title("📰 WELS Actions")

    st.markdown(
        "Newsletter de marche, graphiques financiers et "
        "suivi des risques patrimoniaux."
    )

    st.sidebar.title("WELS Actions")

    page = st.sidebar.radio(
    "Navigation",
    [
        "📰 Newsletter",
        "📈 Marchés & portefeuille",
        "💱 Devises",
        "⚠️ Sources & risques",
    ],
    key="wels_newsletter_navigation",
    )
    if st.sidebar.button("🔄 Vider le cache"):

        fetch_news.clear()
        fetch_history.clear()

        st.rerun()

    if page == "Newsletter du jour":
        newsletter_page()

    elif page == "Marches et portefeuille":
        markets_page()

    elif page == "Change EUR/CHF et EUR/AED":
        currencies_page()

    else:
        sources_page()


if __name__ == "__main__":

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📰",
        layout="wide",
    )

    newsletter_dashboard()
