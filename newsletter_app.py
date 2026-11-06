# WELS Actions — tableau de bord financier et veille
# Installation : pip install -r requirements.txt
# Lancement : streamlit run newsletter_app.py

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
import html
import hashlib
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # Curated publisher feeds replace GDELT, which was rate-limiting the Streamlit app.
    "Franceinfo — France": "https://www.franceinfo.fr/france.rss",
    "Franceinfo — Monde": "https://www.franceinfo.fr/monde.rss",
    "France 24 — International": "https://www.france24.com/fr/rss",
    "BBC — World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC — Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Deutsche Welle — World": "https://rss.dw.com/rdf/rss-en-all",
    "The Guardian — World": "https://www.theguardian.com/world/rss",
    "Al Jazeera — International": "https://www.aljazeera.com/xml/rss/all.xml",
    "BleepingComputer — Cyber": "https://www.bleepingcomputer.com/feed/",
    "The Hacker News — Cyber": "https://feeds.feedburner.com/TheHackersNews",
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
    "Vanguard FTSE All-World UCITS ETF Acc (VWCE)": "VWCE.DE",
    "Frontline": "FRO",
    "FTAI Aviation": "FTAI",
    "Thales": "HO.PA",
    "Lockheed Martin": "LMT",
    "Air Liquide": "AI.PA",
    "Amundi Global Health Care (WELS)": "WELS.DE",
    "iShares World Utilities (XDWU)": "XDWU.L",
    "SPDR World Health Care (WHEA)": "WHEA.AS",
    "VanEck Electrification (PIKA)": "PIKA.DE",
    "Schneider Electric": "SU.PA",
    "Siemens": "SIE.DE",
    "S&P 500": "^GSPC",
    "Nasdaq Composite": "^IXIC",
    "Bitcoin": "BTC-USD",
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


ENGLISH_STOPWORDS = {"the", "and", "of", "to", "in", "for", "on", "by", "from", "with", "after", "over", "as", "at", "is", "are", "was", "were", "says", "said", "will", "its"}
FRENCH_STOPWORDS = {"le", "la", "les", "des", "du", "de", "un", "une", "et", "pour", "dans", "sur", "par", "avec", "après", "contre", "selon", "est", "sont", "au", "aux", "en"}

def supported_news_language(title, summary=""):
    """Keep French/English reporting; reject unrelated language noise and uncertain short titles."""
    text = f"{title} {summary}".lower()
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0370-\u03ff\u0400-\u04ff]", text):
        return False
    words = set(re.findall(r"[a-zà-ÿ']+", text))
    return max(len(words & ENGLISH_STOPWORDS), len(words & FRENCH_STOPWORDS)) >= 2

def topic_for(text):
    text = (text or "").lower()
    for topic, words in TOPICS.items():
        if topic != "Tous les thèmes" and any(word in text for word in words):
            return topic
    return "Autres actualités"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(max_per_feed=25):
    """Fetch independent publisher RSS feeds concurrently with strict timeouts."""
    if feedparser is None:
        return pd.DataFrame(), ["Le module feedparser n'est pas installé."]

    def fetch_one(source, url):
        try:
            response = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=(3, 6),
            )
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            if not parsed.entries:
                return [], f"{source} : flux vide ou format RSS non reconnu."

            rows = []
            for item in parsed.entries[:max_per_feed]:
                title = clean_html(item.get("title", ""))
                link = item.get("link", "")
                if not title or not link:
                    continue
                summary = clean_html(item.get("summary", item.get("description", "")))
                if not supported_news_language(title, summary):
                    continue
                published = item.get("published", item.get("updated", ""))
                rows.append({
                    "Date": parse_date(published),
                    "Source": source,
                    "Domaine": urlparse(link).netloc.replace("www.", ""),
                    "Titre": title,
                    "Résumé": summary[:2500],
                    "Lien": link,
                })
            return rows, None if rows else f"{source} : aucun article FR/EN exploitable."
        except requests.exceptions.Timeout:
            return [], f"{source} : délai dépassé (6 secondes)."
        except requests.exceptions.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return [], f"{source} : HTTP {status or 'erreur'}."
        except requests.exceptions.RequestException as exc:
            return [], f"{source} : erreur réseau {type(exc).__name__}."
        except Exception as exc:
            return [], f"{source} : {type(exc).__name__}."

    articles, errors = [], []
    # Parallel requests prevent one slow publisher from blocking the other sources.
    with ThreadPoolExecutor(max_workers=len(RSS_FEEDS)) as executor:
        futures = {
            executor.submit(fetch_one, source, url): source
            for source, url in RSS_FEEDS.items()
        }
        for future in as_completed(futures):
            try:
                rows, error = future.result()
                articles.extend(rows)
                if error:
                    errors.append(error)
            except Exception as exc:
                errors.append(f"{futures[future]} : erreur inattendue {type(exc).__name__}.")

    if not articles:
        return pd.DataFrame(), errors or ["Aucun flux éditeur n'a retourné d'article."]
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gold_eur_gram(period="1y"):
    """Estimation du cours EUR/g depuis les futures or USD/once et EUR/USD."""
    gold = fetch_history("GC=F", period)
    eurusd = fetch_history("EURUSD=X", period)  # USD pour 1 EUR
    if gold.empty or eurusd.empty:
        return pd.DataFrame()
    gold = gold.rename(columns={"Cours": "Or_USD_oz"})
    eurusd = eurusd.rename(columns={"Cours": "USD_par_EUR"})
    merged = pd.merge(gold, eurusd, on="Date", how="inner")
    merged = merged[(merged["USD_par_EUR"] > 0) & (merged["Or_USD_oz"] > 0)].copy()
    merged["Cours"] = merged["Or_USD_oz"] / merged["USD_par_EUR"] / 31.1034768
    return merged[["Date", "Cours"]].dropna()


def fx_conversion_assessment(symbol, amount_eur=1000):
    """Un taux EUR/devise élevé signifie davantage de devise étrangère reçue par euro."""
    hist = fetch_history(symbol, "1y")
    if hist.empty or len(hist) < 20:
        return None
    rates = pd.to_numeric(hist["Cours"], errors="coerce").dropna()
    if len(rates) < 20:
        return None
    current = float(rates.iloc[-1])
    percentile = float((rates <= current).mean() * 100)
    if percentile >= 75:
        label, color, bg = "FAVORABLE", "#237A45", "#E5F4E9"
        explanation = "L'euro achète plus de cette devise que lors d'au moins 75 % des séances de l'année."
    elif percentile <= 25:
        label, color, bg = "PEU FAVORABLE", "#B42318", "#FCE8E8"
        explanation = "L'euro achète moins de cette devise que lors d'au moins 75 % des séances de l'année."
    else:
        label, color, bg = "INTERMÉDIAIRE", "#A86100", "#FFF3D9"
        explanation = "Le taux se situe dans la zone intermédiaire de sa fourchette annuelle."
    return {"current": current, "amount": amount_eur * current, "percentile": percentile,
            "avg30": float(rates.tail(30).mean()), "avg1y": float(rates.mean()),
            "low": float(rates.min()), "high": float(rates.max()),
            "label": label, "color": color, "bg": bg, "explanation": explanation}


def render_fx_assessment(symbol, name, amount_eur=1000):
    a = fx_conversion_assessment(symbol, amount_eur)
    if a is None:
        st.info(f"Évaluation indisponible pour {name} : historique de change insuffisant.")
        return
    st.markdown(f"""
    <div style="background:{a['bg']};border:1px solid {a['color']}80;border-radius:12px;padding:13px;margin:8px 0">
      <div style="font-size:17px;font-weight:800;color:{a['color']}">{a['label']} pour convertir des euros en {name}</div>
      <div style="font-size:14px;color:#344054;margin:5px 0">{a['explanation']}</div>
      <div style="font-size:22px;font-weight:900;color:#263445">1 000 € ≈ {a['amount']:,.2f} {name}</div>
      <div style="font-size:12px;color:#596579">Taux actuel {a['current']:.4f} · percentile annuel {a['percentile']:.0f}% · moyenne 30 séances {a['avg30']:.4f} · moyenne 1 an {a['avg1y']:.4f}</div>
      <div style="font-size:12px;color:#596579">Fourchette 1 an : {a['low']:.4f} – {a['high']:.4f}</div>
    </div>""", unsafe_allow_html=True)


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
        gold_hist = fetch_gold_eur_gram(period)
        if gold_hist.empty:
            st.warning("Cours de l'or en EUR/g indisponible.")
        else:
            last_gold = float(gold_hist["Cours"].iloc[-1])
            prev_gold = float(gold_hist["Cours"].iloc[-2]) if len(gold_hist) > 1 else last_gold
            st.metric("🥇 Or — euro / gramme", f"{last_gold:,.2f} €", f"{(last_gold / prev_gold - 1) * 100:+.2f}% dernière séance")
            st.caption("Estimation indicative depuis les futures or COMEX en USD/once convertis en EUR via EUR/USD ; hors prime et frais.")
            fig = px.line(gold_hist, x="Date", y="Cours", title=f"Or — EUR par gramme · {period_label}")
            fig.update_layout(height=260, yaxis_title="EUR / gramme")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        market_summary_card("Bitcoin (USD)", "BTC-USD", period)

    st.subheader("Actions et ETF")
    defaults = ["TotalEnergies", "Exxon Mobil", "Vanguard FTSE All-World UCITS ETF Acc (VWCE)", "Frontline", "FTAI Aviation", "Thales", "Lockheed Martin", "Air Liquide", "Amundi Global Health Care (WELS)", "iShares World Utilities (XDWU)", "Schneider Electric", "Siemens"]
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
    st.caption("Cours quotidiens Yahoo Finance : date du dernier point affichée dans le tableau. Les devises de cotation diffèrent ; les performances sont calculées dans la devise de chaque ligne et ne sont pas directement comparables en euros.")


def currencies_page():
    st.header("💱 Change EUR/CHF et EUR/AED")
    st.caption("Lecture du taux pour savoir combien de devises tu reçois en échange de tes euros.")
    period_label = st.selectbox("Période", list(PERIODS.keys()), index=3, key="fx_period")
    period = PERIODS[period_label]
    for name, symbol in FX_SYMBOLS.items():
        hist = fetch_history(symbol, period)
        st.subheader("🇨🇭 EUR/CHF" if name == "EUR/CHF" else "🇦🇪 EUR/AED")
        if hist.empty:
            st.warning(f"Cours indisponible pour {name} ({symbol}).")
            continue
        last = float(hist["Cours"].iloc[-1])
        previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
        c1, c2 = st.columns(2)
        c1.metric("Dernier cours", f"{last:.4f}")
        c2.metric("Dernière séance", f"{(last / previous - 1) * 100:+.2f}%")
        render_fx_assessment(symbol, "CHF" if name == "EUR/CHF" else "AED")
        fig = px.line(hist, x="Date", y="Cours", title=f"{name} — {period_label}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Un taux EUR/CHF ou EUR/AED plus élevé est plus favorable à l'achat de devises avec des euros. Le signal compare le cours au percentile de son historique sur un an ; il ne prédit pas son évolution et n'intègre pas les frais bancaires. L'AED est arrimé au dollar américain.")


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
    st.caption("Source unique des cours : Yahoo Finance. Cache 5 min ; dernier point quotidien disponible (pas un flux intraday). Chaque ligne indique sa date et sa devise.")
    render_drawdown_alerts()
    period_label = st.selectbox("Période", list(PERIODS.keys()), index=3, key="wels_market_period")
    period = PERIODS[period_label]
    watch = dict(MARKET_SYMBOLS)
    watch.update({"Schneider Electric": "SU.PA", "Siemens": "SIE.DE"})
    selected = st.multiselect("Actifs suivis", list(watch), default=[
        "TotalEnergies", "Exxon Mobil", "Vanguard FTSE All-World UCITS ETF Acc (VWCE)",
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
        summary.append({"Actif": name, "Ticker Yahoo": watch[name], "Dernier point": round(last, 4),
                        "Devise": "EUR" if watch[name].endswith((".PA", ".DE", ".AS")) else ("GBP" if watch[name].endswith(".L") else "USD"),
                        "Variation dernière séance (%)": round((last / prev - 1) * 100, 2) if prev else None,
                        "Variation période (%)": round((last / first - 1) * 100, 2) if first else None,
                        "Date du point": pd.to_datetime(hist["Date"].iloc[-1]).strftime("%d/%m/%Y")})
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
        {"Actif": "Vanguard FTSE All-World UCITS ETF Acc (VWCE)", "Type": "ETF", "Exposition": "Actions mondiales"},
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
        render_fx_assessment(symbol, "CHF" if name == "EUR/CHF" else "AED")
        if name == "EUR/AED":
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

def investment_flows_page():
    st.header("💸 Follow the Money — Global Capital Flows")
    st.caption(
        "Cartographie des grandes masses de capitaux qui orientent les capacités industrielles, "
        "technologiques, militaires et alimentaires. Priorité aux volumes systémiques, pas aux petites levées de fonds."
    )
    st.info(
        "Lecture essentielle : les montants ci-dessous ne sont pas additionnables. "
        "Certains sont des dépenses annuelles, d'autres des levées de fonds, des investissements annoncés "
        "ou une capacité de financement estimée. Les chevauchements entre secteurs sont signalés."
    )

    flows = [
        {
            "sector": "IA, data centers & semi-conducteurs", "icon": "🧠",
            "headline": "≈ 900 Md$ en 2026", "sub": "Investissement mondial lié à l'IA estimé par J.P. Morgan",
            "extra": "Projection : 5 500 Md$ cumulés d'ici 2030 selon J.P. Morgan. Les dépenses des grands hyperscalers sont le moteur principal.",
            "horizon": "2026–2030 · immédiat à 4 ans",
            "status": "Estimation 2026 + projection", "kind": "Capex / infrastructures",
            "impact": "Accélération des centres de données, GPU, mémoire, réseaux, électricité et refroidissement. Pression sur le financement, le réseau électrique et les chaînes de semi-conducteurs.",
            "watch": "Capex annoncé vs réalisé, commandes GPU/mémoire, raccordements électriques, dette et cash-flow des hyperscalers.",
            "source": "J.P. Morgan — 19 octobre 2026", "url": "https://www.jpmorgan.com/insights/global-research/artificial-intelligence/ai-capex"
        },
        {
            "sector": "Énergie, réseaux & stockage", "icon": "⚡",
            "headline": "3 400 Md$ en 2026", "sub": "Investissement mondial attendu dans l'énergie (IEA)",
            "extra": "Environ 2 200 Md$ vers les renouvelables, le nucléaire, les réseaux, le stockage, l'efficacité et l'électrification ; 1 200 Md$ vers pétrole, gaz et charbon.",
            "horizon": "2026–2035 · 5 à 10 ans et plus",
            "status": "Prévision annuelle 2026", "kind": "Investissement énergétique",
            "impact": "Renforce la sécurité énergétique et la capacité électrique. Les réseaux, transformateurs, batteries et centrales deviennent des goulets d'étranglement stratégiques.",
            "watch": "Dépenses réseau, stockage, nucléaire, contrats d'approvisionnement, retards de raccordement et investissements fossiles.",
            "source": "IEA — World Energy Investment 2026", "url": "https://www.iea.org/topics/investment"
        },
        {
            "sector": "Défense & industrie militaire", "icon": "🛡️",
            "headline": "2 887 Md$ en 2025", "sub": "Dépenses militaires mondiales mesurées par SIPRI (+2,9 % réels)",
            "extra": "11e année consécutive de hausse. Europe : 864 Md$ (+14 %) ; Asie-Océanie : 681 Md$ (+8,1 %). Il s'agit de dépenses militaires totales, pas uniquement de commandes d'armement.",
            "horizon": "2025–2030 · budgets pluriannuels",
            "status": "Dépense observée 2025", "kind": "Budgets publics récurrents",
            "impact": "Soutient la montée en cadence des industriels, les stocks de munitions, la défense aérienne, les drones, le cyber et les capacités spatiales.",
            "watch": "Budgets votés, commandes fermes, délais de livraison, capacités industrielles et reconstitution des stocks.",
            "source": "SIPRI — 27 avril 2026", "url": "https://www.sipri.org/media/press-release/2026/global-military-spending-rise-continues-european-and-asian-expenditures-surge"
        },
        {
            "sector": "Capital-risque & startups", "icon": "🚀",
            "headline": "427,1 Md$ en 2025", "sub": "Investissement mondial en venture capital (OCDE)",
            "extra": "L'IA a capté 258,7 Md$, soit 61 % du total. Les financements se concentrent dans les méga-tours et les entreprises déjà avancées.",
            "horizon": "2025–2028 · 2 à 5 ans",
            "status": "Investissement observé 2025", "kind": "Capital privé / VC",
            "impact": "Oriente les futurs leaders technologiques et la création de capacités. Mais une levée de fonds n'est pas encore une capacité industrielle rentable.",
            "watch": "Part de l'IA, taille des méga-tours, financement des phases précoces, sorties/IPO et taux de survie des startups.",
            "source": "OCDE — 17 février 2026", "url": "https://www.oecd.org/en/about/news/announcements/2026/02/ai-firms-capture-61-percent-of-global-venture-capital-in-2025.html"
        },
        {
            "sector": "Entreprises mondiales & investissements directs", "icon": "🏭",
            "headline": "1 600 Md$ en 2025", "sub": "Investissements directs étrangers mondiaux (UNCTAD)",
            "extra": "Les projets greenfield annoncés dans cinq secteurs stratégiques ont atteint 576 Md$ en 2025, contre 109 Md$ en 2020. L'IA et les infrastructures numériques dominent ; les 20 premiers pays hôtes captent plus de 80 % des flux.",
            "horizon": "2025–2030 · 3 à 7 ans",
            "status": "Estimation FDI 2025 / projets annoncés", "kind": "Investissement productif transfrontalier",
            "impact": "Révèle où les groupes et investisseurs construisent de nouvelles capacités : usines, data centers, semi-conducteurs, énergie et technologies stratégiques.",
            "watch": "Annonces converties en chantiers, décisions finales d'investissement, pays hôtes, annulations et délais de construction.",
            "source": "UNCTAD — World Investment Report 2026", "url": "https://unctad.org/publication/world-investment-report-2026-international-investment-turbulent-era"
        },
        {
            "sector": "Santé, biotech & pharmacie", "icon": "🧬",
            "headline": "60 Md$ en 2025", "sub": "Investissement mondial de venture capital dans la santé (HSBC)",
            "extra": "Hausse depuis 45,4 Md$ en 2024. HSBC anticipait 65–70 Md$ en 2026 ; les financements se concentrent sur les biotechs, les dispositifs médicaux et les entreprises proches de la commercialisation.",
            "horizon": "2025–2035+ · 5 à 15 ans",
            "status": "Observé 2025 + prévision 2026", "kind": "VC / innovation médicale",
            "impact": "Peut accélérer les traitements, diagnostics, plateformes biotech et capacités de production pharmaceutique. Les retours sont longs et le risque clinique/réglementaire élevé.",
            "watch": "Financements late-stage, essais cliniques, autorisations, capacités de production, acquisitions et ruptures d'approvisionnement.",
            "source": "HSBC Innovation Banking — 13 janvier 2026", "url": "https://www.about.us.hsbc.com/newsroom/press-releases/healthcare-venture-markets-regain-momentum-as-capital-returns"
        },
        {
            "sector": "Food, agriculture & sécurité alimentaire", "icon": "🌾",
            "headline": "≈ 630 Md$ / an", "sub": "Capacité potentielle estimée de financement privé de l'agroalimentaire",
            "extra": "Ce chiffre correspond à une capacité de financement estimée, pas à un flux annuel effectivement investi. Le financement climatique nécessaire aux systèmes agroalimentaires est estimé à 1 100 Md$ par an.",
            "horizon": "2026–2035 · 5 à 10 ans",
            "status": "Capacité estimée / besoin de financement", "kind": "Potentiel privé et besoin structurel",
            "impact": "Détermine la résilience des chaînes alimentaires : irrigation, stockage, engrais, semences, transformation, logistique et adaptation climatique.",
            "watch": "Investissements réellement déployés, eau/irrigation, stocks céréaliers, engrais, capacités de stockage et financement des producteurs.",
            "source": "Food and Land Use Coalition — rapport 2026", "url": "https://www.foodandlandusecoalition.org/knowledge-hub/re-wire-agricultural-resilience/"
        },
    ]

    # Valeurs en Md$ utilisées pour le tri et la longueur relative des barres.
    value_map = {
        "IA, data centers & semi-conducteurs": 900,
        "Énergie, réseaux & stockage": 3400,
        "Défense & industrie militaire": 2887,
        "Capital-risque & startups": 427.1,
        "Entreprises mondiales & investissements directs": 1600,
        "Santé, biotech & pharmacie": 60,
        "Food, agriculture & sécurité alimentaire": 630,
    }
    sorted_flows = sorted(flows, key=lambda x: value_map.get(x["sector"], 0), reverse=True)
    # Palette pastel : une teinte douce et distincte par secteur.
    colors = ["#A9C9E8", "#F2C98A", "#C5CBD3", "#C9B6E4", "#A8D5C2", "#E8B8CC", "#C9D99B"]
    max_value = max(value_map.values())
    row_count = len(sorted_flows)
    y_positions = list(range(row_count))
    # Les barres de volume sont volontairement plafonnées à 62 % de la piste.
    ratios = [value_map[item["sector"]] / max_value * 62 for item in sorted_flows]
    sectors = [item["sector"] for item in sorted_flows]

    st.subheader("Les grandes masses de capitaux — du plus élevé au plus faible")
    st.caption(
        "Vue en deux panneaux : sélectionnez une barre à gauche pour afficher la fiche complète à droite. "
        "Les barres sont volontairement raccourcies pour laisser respirer les libellés."
    )

    left_panel, right_panel = st.columns([1, 1], gap="medium")

    with left_panel:
        st.markdown("**Répartition par secteur**")
        fig = go.Figure()
        # Piste cliquable de largeur constante, y compris pour les petits montants.
        fig.add_trace(go.Bar(
            x=[72] * row_count, y=y_positions, orientation="h",
            width=0.82,
            marker=dict(color=colors, opacity=0.24, line=dict(color=colors, width=1)),
            customdata=sectors, hoverinfo="skip", showlegend=False,
            selected=dict(marker=dict(opacity=0.45)),
            unselected=dict(marker=dict(opacity=0.24)),
        ))
        # Barre pastel représentant le montant relatif.
        fig.add_trace(go.Bar(
            x=ratios, y=y_positions, orientation="h",
            width=0.82,
            marker=dict(color=colors, line=dict(color=colors, width=1)),
            customdata=sectors,
            hovertemplate="<b>%{customdata}</b><br>Longueur relative : %{x:.1f} %<extra>Cliquez pour le détail</extra>",
            showlegend=False,
        ))

        for y, item in zip(y_positions, sorted_flows):
            detail = (
                f'<b>{item["icon"]} {item["sector"]}</b><br>'
                f'<b>{item["headline"]}</b><br>'
                f'{item["source"]} · {item["horizon"]}'
            )
            fig.add_annotation(
                x=1.5, y=y, xref="x", yref="y", text=detail,
                showarrow=False, xanchor="left", yanchor="middle",
                align="left", font=dict(family="Arial, sans-serif", size=16, color="#26384A")
            )

        fig.update_layout(
            barmode="overlay", height=850,
            margin=dict(l=4, r=4, t=8, b=8),
            bargap=0.22,
            xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
            yaxis=dict(
                range=[row_count - 0.5, -0.5], visible=False, fixedrange=True,
                tickmode="array", tickvals=y_positions, ticktext=[""] * row_count
            ),
            plot_bgcolor="white", paper_bgcolor="white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            clickmode="event+select", dragmode=False,
        )
        chart_event = st.plotly_chart(
            fig, use_container_width=True, config={"displayModeBar": False},
            key="capital_flows_chart", on_select="rerun", selection_mode="points"
        )

    # Le clic sur une barre sélectionne le secteur via sa donnée personnalisée.
    selected_sector = None
    try:
        selected_points = chart_event.selection.points
        if selected_points:
            point = selected_points[-1]
            selected_sector = point.get("customdata")
            if isinstance(selected_sector, (list, tuple)) and selected_sector:
                selected_sector = selected_sector[0]
    except Exception:
        selected_sector = None

    selected_item = next(
        (item for item in sorted_flows if item["sector"] == selected_sector), None
    )

    with right_panel:
        st.markdown("**Détail de l'investissement sélectionné**")
        if selected_item:
            st.markdown(f"### {selected_item['icon']} {selected_item['sector']}")
            st.markdown(f"## {selected_item['headline']}")
            st.markdown(f"**{selected_item['sub']}**")
            st.markdown("---")
            st.markdown(
                f'**Nature du chiffre :** {selected_item["status"]}  \n'
                f'**Type :** {selected_item["kind"]}'
            )
            st.markdown(f'**Horizon :** {selected_item["horizon"]}')
            st.markdown("**Précision / projection**")
            st.write(selected_item["extra"])
            st.markdown("**Pourquoi c'est important**")
            st.write(selected_item["impact"])
            st.markdown("**Indicateurs à surveiller**")
            st.write(selected_item["watch"])
            st.markdown(f'**Source :** [{selected_item["source"]}]({selected_item["url"]})')
        else:
            st.info(
                "Cliquez sur l'une des barres du graphique pour afficher ici "
                "le montant, la source, l'horizon, les projections, les impacts "
                "et les indicateurs à surveiller."
            )

    st.caption(
        "Les longueurs sont proportionnelles aux montants indicatifs en Md$, avec une échelle visuelle "
        "raccourcie. Les chiffres ne sont pas additionnables : dépenses publiques, investissements, "
        "capital-risque et capacités potentielles sont des catégories différentes qui peuvent se chevaucher."
    )

    st.divider()
    st.subheader("Lecture stratégique — ce que les flux de capitaux annoncent")
    st.markdown(
        """
        - **Course aux capacités physiques :** l'IA ne finance pas seulement des logiciels. Elle tire les centres de données, les semi-conducteurs, l'électricité, les réseaux et le refroidissement.
        - **Réarmement durable :** les budgets militaires sont récurrents et les commandes industrielles s'étalent sur plusieurs années ; les annonces politiques ne deviennent toutefois des revenus industriels qu'après contractualisation.
        - **Capital concentré :** les investissements directs et le capital-risque se concentrent dans un nombre limité de pays, de secteurs et de grandes opérations.
        - **Santé et alimentation :** les cycles sont plus longs, mais ils touchent des besoins fondamentaux. Il faut suivre les capacités de production et les infrastructures, pas seulement les valorisations boursières.
        - **Risque de surinvestissement :** un volume de capitaux élevé peut annoncer une capacité future, mais aussi une bulle, un excès d'endettement ou des actifs qui ne seront pas rentables.
        """
    )
    st.warning(
        "Méthode : les indicateurs sont volontairement séparés et non additionnés. "
        "Les montants de VC santé et IA sont inclus dans le total VC ; les projets IA peuvent aussi apparaître dans les FDI et les capex des entreprises. "
        "Les horizons sont des fenêtres d'impact indicatives, pas des dates certaines de rendement."
    )
    st.caption("Base documentaire initiale : publications 2026 disponibles au 6 novembre 2026. Actualisation recommandée à chaque nouvelle publication annuelle ou trimestrielle des organismes cités ; les montants ne sont pas un flux temps réel.")


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
            "terms": ["invasion", "missile attack", "missile strike", "missile strikes", "airstrike", "air strike", "military strike", "military attack", "military escalation", "attack on iran", "attack on israel", "iran attack", "israel attack", "escalade militaire", "frappes aériennes", "frappe militaire", "attaque de missiles", "attaque militaire", "offensive militaire", "mobilisation militaire"],
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
        {"id":"cyber_critical","terms":[
            "cyberattack","cyber attack","cyber incident","cybersecurity incident","ransomware","ransomware attack",
            "hackers breach","data breach","computer attack","critical infrastructure cyber",
            "cyberattaque","cyber attaque","attaque informatique","attaque cyber","incident cyber",
            "rançongiciel","piratage informatique","système frigorifique","frigos","chambres froides",
            "cold storage","refrigeration systems","cooling systems","Danfoss","Copeland xweb",
            "Etablissement français du sang","Établissement français du sang","EFS cyber",
            "Crédit Mutuel Arkéa","Credit Mutuel Arkea","Arkea cyberattack","Binance hack",
            "Binance security incident","crypto exchange hack","exchange hack",
            "bank cyberattack","banking cyberattack","hospital cyberattack","blood bank cyberattack",
            "food supply cyberattack","supermarket cyberattack","industrial control system attack",
            "operational technology cyber","OT cyberattack","SCADA attack"
        ],"event":"Cyberattaque ou compromission numérique","impact":"Peut perturber les paiements, les services de santé, les chaînes du froid, l'alimentation, l'énergie, l'eau ou les télécommunications ; l'impact réel doit être qualifié par secteur et territoire.","market":"Risque opérationnel, pertes financières, interruption de service et perturbation logistique selon la cible.","action":"Identifier la cible, le système compromis, les services effectivement interrompus, la durée et les mesures de confinement ; recouper avec CERT-FR, ANSSI, ENISA ou les communiqués de l'organisation.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Incident confirmé ou signalé ; conséquences à qualifier"},
        {"id":"banking_capital","terms":["bank run","bank failure","bank collapse","capital controls","withdrawal limits","deposit freeze","contrôle des capitaux","limitation des retraits","gel des dépôts","faillite bancaire"],"event":"Crise bancaire ou restriction d'accès aux liquidités","impact":"Risque direct pour l'accès aux dépôts, paiements et transferts ; distinguer rumeur, difficulté d'une banque et mesure officielle.","market":"Stress financier et risque de contagion si plusieurs établissements sont concernés.","action":"Vérifier les communications du régulateur et de la banque centrale ; ne pas agir sur des rumeurs.","level":"CRITIQUE","score":4,"polarity":"negative","confidence":"Confirmation réglementaire requise"},
        {"id":"food_supply","terms":["food shortage","food supply disruption","food rationing","grain export ban","food crisis","pénurie alimentaire","rupture d'approvisionnement alimentaire","rationnement alimentaire"],"event":"Risque majeur sur l'approvisionnement alimentaire","impact":"Hausse des prix ou disponibilité réduite de produits essentiels ; conséquences sociales si la crise dure ou s'étend.","market":"Pression possible sur les prix agricoles et l'inflation.","action":"Vérifier les produits, territoires et volumes concernés auprès des autorités et opérateurs ; éviter les achats de panique.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Produits et territoires à préciser"},
        {"id":"sovereign_financial","terms":["sovereign debt crisis","sovereign default","debt restructuring","crise de la dette souveraine","défaut souverain","restructuration de la dette publique"],"event":"Crise de dette souveraine","impact":"Risque de hausse des taux, mesures budgétaires et pression sur les banques et l'activité ; conséquences dépendantes du pays.","market":"Tension possible sur obligations, banques, euro et financement public.","action":"Suivre décisions gouvernementales, banque centrale et régulateurs ; distinguer dégradation de notation et défaut effectif.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Indicateurs à recouper"},
        {"id":"mass_social_disruption","terms":["nationwide general strike","national general strike","mass riots","state of emergency","grève générale nationale","émeutes généralisées","état d'urgence"],"event":"Troubles sociaux majeurs ou paralysie nationale","impact":"Risque de perturbation des transports, services publics, commerces et accès aux soins ; impact dépendant des villes et de la durée.","market":"Pertes d'activité, retards logistiques et incertitude pour les entreprises.","action":"Vérifier les zones et services touchés ; anticiper les déplacements et suivre les consignes officielles.","level":"ÉLEVÉ","score":3,"polarity":"negative","confidence":"Échelle et conséquences à confirmer"}
    ]);
    # Economic and social transmission signals: detect pressure before a formal shortage.
    rules.extend([
        {
            "id": "gas_market_tightness",
            "terms": ["gas market tightness", "low storage buffers", "low gas storage", "gas storage levels", "gas storage", "gas inventories", "gas stocks", "gas supply tightness", "faibles stocks de gaz", "stocks de gaz", "réserves de gaz", "stockage de gaz", "tension sur le marché du gaz", "approvisionnement gazier"],
            "event": "Tension sur les réserves ou l'approvisionnement de gaz",
            "impact": "Des stocks faibles ou une offre tendue peuvent renchérir le prix de gros du gaz. Une transmission aux factures de chauffage et de gaz des ménages est possible selon les contrats, les mécanismes tarifaires et le calendrier de révision ; elle n'est ni immédiate ni automatique.",
            "market": "Risque de volatilité ou de hausse du gaz de gros (notamment TTF), avec effets possibles sur l'électricité, l'industrie et l'inflation.",
            "action": "Suivre le niveau des stocks européens, les prix TTF, les flux de gaz et de GNL, la météo et les annonces de la Commission européenne. Pour la France, vérifier les évolutions tarifaires et les conditions de son contrat.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Risque de tension identifié ; évolution des prix et répercussion sur les factures à confirmer"
        },
        {
            "id": "gas_price_increase",
            "terms": ["gas prices rise", "gas price rises", "gas price surge", "gas price spike", "gas prices jump", "gas prices climb", "gas prices increase", "hausse du prix du gaz", "hausse des prix du gaz", "prix du gaz en hausse", "flambée du gaz", "envolée du gaz"],
            "event": "Hausse des prix de marché du gaz",
            "impact": "Une hausse durable des prix de gros peut augmenter les coûts de chauffage et de cuisson des ménages, directement ou lors d'une révision contractuelle, et peser sur les entreprises. L'ampleur dépend du pays, du contrat et des protections tarifaires.",
            "market": "Pression potentielle sur l'électricité, les coûts industriels et l'inflation ; vérifier le TTF et la durée du mouvement.",
            "action": "Distinguer mouvement journalier et hausse durable ; suivre TTF, prix de détail et annonces tarifaires françaises avant de conclure à une hausse des factures.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Hausse de marché à recouper avec les prix de détail"
        },
        {
            "id": "oil_price_increase",
            "terms": ["oil rises", "oil prices rise", "crude rises", "brent rises", "oil prices jump", "oil prices climb", "surge in oil prices", "pétrole en hausse", "hausse du pétrole", "prix du pétrole en hausse", "brent en hausse", "flambée du pétrole"],
            "event": "Hausse du pétrole et risque de transmission aux carburants",
            "impact": "Une hausse durable du brut peut se répercuter sur les prix des carburants, les transports et certains coûts de production. La répercussion dépend aussi du raffinage, des marges, des taxes et du taux de change.",
            "market": "Pression potentielle sur Brent, diesel, essence, fret et inflation ; la hausse du brut ne se traduit pas mécaniquement ni immédiatement à la pompe.",
            "action": "Suivre le Brent sur plusieurs séances, les cours des produits raffinés et les prix à la pompe en France ; distinguer réaction ponctuelle et tendance durable.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Transmission aux prix de détail possible, non automatique"
        },
        {
            "id": "electricity_price_increase",
            "terms": ["electricity prices rise", "electricity price surge", "power prices rise", "power price spike", "electricity prices increase", "hausse du prix de l'électricité", "hausse des prix de l'électricité", "prix de l'électricité en hausse"],
            "event": "Hausse des prix de marché de l'électricité",
            "impact": "Une hausse durable des prix de gros peut se répercuter sur les factures des ménages et les coûts des entreprises, selon les contrats, les mécanismes tarifaires et le calendrier de révision.",
            "market": "Pression possible sur les coûts industriels et l'inflation ; les prix de gros ne correspondent pas directement à la facture finale.",
            "action": "Suivre les prix de gros, les annonces tarifaires françaises et les mesures publiques ; distinguer prix spot, contrats à terme et tarifs de détail.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Répercussion sur les factures dépendante des contrats et mécanismes tarifaires"
        },
        {
            "id": "diplomatic_negotiations",
            "terms": ["mediate talks", "peace talks", "negotiations between", "talks with kyiv", "talks with moscow", "ceasefire talks", "pourparlers", "négociations de paix", "médiation diplomatique"],
            "event": "Ouverture ou reprise de discussions diplomatiques",
            "impact": "Une médiation peut ouvrir une voie de désescalade, mais l'annonce de discussions ne signifie ni accord ni cessez-le-feu. Les risques sécuritaires et économiques restent à suivre.",
            "market": "Effet de marché incertain ; une baisse durable de la prime de risque dépendrait de résultats concrets et vérifiables.",
            "action": "Suivre la tenue effective des discussions, les déclarations des parties et tout accord formel ; ne pas assimiler une annonce de médiation à une résolution du conflit.",
            "level": "INFORMATION", "score": 1, "polarity": "neutral", "confidence": "Processus diplomatique annoncé ; résultat non acquis"
        },
        {
            "id": "measles_health",
            "terms": ["measles", "morbillo", "rougeole", "morbilli"],
            "event": "Risque sanitaire lié à la rougeole",
            "impact": "La rougeole est très contagieuse. Des cas ou foyers peuvent entraîner des chaînes de transmission, des consultations et hospitalisations, surtout chez les personnes non immunisées.",
            "market": "Impact principalement sanitaire et organisationnel ; ne pas déduire une épidémie d'un article général ou d'un cas isolé.",
            "action": "Vérifier le nombre de cas, les régions et les recommandations des autorités sanitaires ; consulter les recommandations vaccinales officielles et un professionnel de santé en cas de doute.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Cas, foyer et diffusion géographique à confirmer"
        },
        {
            "id": "homecare_workforce",
            "terms": ["private equity takeover of homecare", "private equity homecare", "home care workers", "homecare workers", "home care services", "domiciliary care", "soins à domicile", "aide à domicile"],
            "event": "Tensions sociales dans les services de soins à domicile",
            "impact": "Des changements de propriété ou de modèle économique peuvent affecter les conditions de travail, la continuité et l'accès aux services. L'effet dépend des opérateurs et des territoires concernés.",
            "market": "Risque social et opérationnel pour un secteur essentiel ; impact local à documenter.",
            "action": "Identifier les opérateurs, les territoires, les effets documentés sur les salariés et la continuité des prestations ; rechercher les rapports et réponses des autorités.",
            "level": "À SURVEILLER", "score": 2, "polarity": "negative", "confidence": "Constats à vérifier selon les opérateurs et territoires"
        },
        {
            "id": "humanitarian_crisis",
            "terms": ["civilian killings", "civilian casualties", "siege", "gaza killings", "west bank raids", "west bank siege", "mass displacement", "forced displacement", "morts de civils", "victimes civiles", "siège", "déplacements forcés"],
            "event": "Aggravation d'une crise humanitaire ou de violences contre les civils",
            "impact": "Risque de victimes, déplacements de population, accès réduit aux soins et à l'aide humanitaire, et extension possible des tensions régionales.",
            "market": "Effets indirects possibles via l'instabilité régionale, les routes commerciales, l'énergie et les dépenses humanitaires ; pas d'effet de marché automatique.",
            "action": "Recouper les bilans et l'accès humanitaire auprès d'organisations reconnues ; suivre les avis aux voyageurs et les évolutions régionales.",
            "level": "ÉLEVÉ", "score": 3, "polarity": "negative", "confidence": "Bilan humain et périmètre à recouper auprès de sources indépendantes"
        },
        {
            "id": "flight_normalization",
            "terms": ["resumes flights", "resume flights", "return to uae flights", "return to the uae", "flights resume", "restarting flights", "reintroduces flights", "reprise des vols", "reprend ses vols", "retour des vols"],
            "event": "Reprise ou rétablissement de liaisons aériennes",
            "impact": "Peut signaler une amélioration de la connectivité et une réduction de certaines perturbations opérationnelles. Cela ne prouve pas à lui seul la disparition du risque sécuritaire régional.",
            "market": "Effet potentiellement favorable pour le tourisme, les compagnies et les déplacements ; portée limitée à la compagnie, aux routes et aux dates concernées.",
            "action": "Vérifier les routes et dates effectivement rétablies, les avis officiels et les conditions des compagnies avant tout déplacement.",
            "level": "ATTÉNUATION", "score": 0, "polarity": "positive", "confidence": "Reprise annoncée ; périmètre opérationnel à vérifier"
        }
    ])
    matches = []
    for rule in rules:
        if any(term in text for term in rule["terms"]):
            matches.append(rule)
    return matches




@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ecb_ciss():
    """ECB New CISS: daily systemic-stress series; return percentile vs its own 5y history."""
    url = "https://data-api.ecb.europa.eu/service/data/CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX"
    response = requests.get(url, params={"startPeriod": (datetime.utcnow() - timedelta(days=1825)).strftime("%Y-%m-%d"), "format": "csvdata"}, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    from io import StringIO
    data = pd.read_csv(StringIO(response.text))
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(data.columns):
        raise ValueError("Format inattendu de l'API BCE")
    data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
    data = data.dropna(subset=["OBS_VALUE"]).sort_values("TIME_PERIOD")
    if data.empty:
        raise ValueError("Série CISS vide")
    current = float(data["OBS_VALUE"].iloc[-1])
    history = data["OBS_VALUE"].tail(1260)
    percentile = float((history <= current).mean() * 100)
    previous = float(data["OBS_VALUE"].iloc[-6]) if len(data) > 5 else current
    change = current - previous
    return {"value": current, "percentile": percentile, "change_5d": change,
            "date": str(data["TIME_PERIOD"].iloc[-1]), "observations": len(data)}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rte_grid_data():
    """Latest French national RTE/ODRE éCO2mix observations, with API-domain fallback."""
    from io import StringIO
    import json

    endpoints = [
        "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/records",
        "https://reseaux-energies-rte.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/records",
    ]
    failures = []
    records = []
    for url in endpoints:
        try:
            response = requests.get(
                url,
                params={"limit": 100, "order_by": "date_heure DESC"},
                headers=REQUEST_HEADERS,
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            records = payload.get("results", [])
            if records:
                break
            failures.append(f"{url.split('/')[2]}: réponse sans enregistrements")
        except Exception as exc:
            failures.append(f"{url.split('/')[2]}: {type(exc).__name__} {str(exc)[:160]}")

    # CSV fallback: ODRÉ also publishes the same national real-time dataset as CSV.
    if not records:
        csv_urls = [
            "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/csv",
            "https://reseaux-energies-rte.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/csv",
        ]
        for url in csv_urls:
            try:
                response = requests.get(url, params={"limit": 100}, headers=REQUEST_HEADERS, timeout=30)
                response.raise_for_status()
                data = pd.read_csv(StringIO(response.text), sep=None, engine="python")
                if not data.empty:
                    records = data.to_dict("records")
                    break
            except Exception as exc:
                failures.append(f"{url.split('/')[2]} CSV: {type(exc).__name__} {str(exc)[:160]}")

    if not records:
        raise ValueError("Aucune donnée reçue depuis les deux API ODRÉ/RTE. " + " | ".join(failures))

    # Flatten alternate Opendatasoft payloads where fields are nested.
    records = [
        row.get("fields", row.get("record", {}).get("fields", row))
        if isinstance(row, dict) else {}
        for row in records
    ]

    data = pd.DataFrame(records)
    required = {"date_heure", "consommation"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Champs absents de la réponse ODRÉ : {', '.join(sorted(missing))}. Champs reçus : {', '.join(map(str, data.columns[:20]))}")

    # Numeric conversion is deliberately deferred until locale separators are normalized.

    # Normalize timestamps and numeric values: CSV exports may contain local date/time,
    # decimal commas, non-breaking spaces or a separate date + heure pair.
    # Parse ISO timestamps first without dayfirst: setting dayfirst globally can
    # silently swap month/day in ISO dates such as 2026-11-06.
    data["date_heure"] = pd.to_datetime(data["date_heure"], utc=True, errors="coerce")
    invalid_dates = data["date_heure"].isna()
    if invalid_dates.any() and {"date", "heure"}.issubset(data.columns):
        rebuilt = data.loc[invalid_dates, "date"].astype(str).str.strip() + " " + data.loc[invalid_dates, "heure"].astype(str).str.strip()
        data.loc[invalid_dates, "date_heure"] = pd.to_datetime(rebuilt, utc=True, errors="coerce", dayfirst=True)

    def _rte_number(series):
        cleaned = series.astype(str).str.replace(r"[\s\u00a0\u202f]", "", regex=True).str.replace(",", ".", regex=False)
        cleaned = cleaned.replace({"": None, "nan": None, "None": None, "null": None, "nd": None, "ND": None})
        return pd.to_numeric(cleaned, errors="coerce")

    data["consommation"] = _rte_number(data["consommation"])
    for col in ["prevision_j", "prevision_j1", "ech_physiques", "nucleaire", "gaz", "eolien", "solaire", "hydraulique", "taux_co2"]:
        if col in data.columns:
            data[col] = _rte_number(data[col])

    data = data.dropna(subset=["date_heure", "consommation"]).sort_values("date_heure")
    if data.empty:
        raise ValueError(
            "Aucune mesure exploitable après normalisation : "
            + str(len(records)) + " lignes reçues ; dates reconnues="
            + str(int(pd.to_datetime(pd.DataFrame(records).get("date_heure"), utc=True, errors="coerce").notna().sum()))
            + ". Vérifier le format date/heure et consommation de la réponse ODRÉ."
        )

    latest = data.iloc[-1]
    age_hours = max(0, (pd.Timestamp.now(tz="UTC") - latest["date_heure"]).total_seconds() / 3600)
    demand = float(latest["consommation"])
    forecast_value = latest.get("prevision_j")
    if pd.isna(forecast_value):
        forecast_value = latest.get("prevision_j1")
    forecast = float(forecast_value) if forecast_value is not None and pd.notna(forecast_value) and float(forecast_value) > 0 else None
    deviation = ((demand / forecast) - 1) * 100 if forecast else None
    physical_exchange = float(latest["ech_physiques"]) if "ech_physiques" in data.columns and pd.notna(latest.get("ech_physiques")) else None

    # In éCO2mix, negative physical exchanges mean net exports; positive means net imports.
    import_share = max(0, physical_exchange) / demand * 100 if physical_exchange is not None and demand > 0 else None
    carbon = float(latest["taux_co2"]) if "taux_co2" in data.columns and pd.notna(latest.get("taux_co2")) else None
    return {
        "date": latest["date_heure"].strftime("%d/%m/%Y %H:%M UTC"),
        "age_hours": age_hours, "demand": demand, "forecast": forecast,
        "deviation": deviation, "exchange": physical_exchange,
        "import_share": import_share, "carbon": carbon,
        "records": len(data), "api_host": url.split("/")[2],
    }

def operational_risk_panel():
    st.subheader("📡 Données opérationnelles — indicateurs mesurés")
    st.caption("Sources directes, indépendantes des articles. Les seuils affichés sont des repères de veille, pas des seuils officiels de déclenchement de crise.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏦 🇪🇺 Stress financier — zone euro**")
        try:
            ciss = fetch_ecb_ciss()
            if ciss["percentile"] >= 95 and ciss["change_5d"] > 0:
                st.error("Stress financier très élevé vs historique récent")
            elif ciss["percentile"] >= 80:
                st.warning("Stress financier élevé vs historique récent")
            elif ciss["percentile"] >= 60:
                st.info("Stress financier à surveiller")
            else:
                st.success("Pas de stress exceptionnel détecté par cet indicateur")
            st.metric("BCE New CISS", f'{ciss["value"]:.3f}', f'{ciss["change_5d"]:+.3f} sur 5 obs.')
            st.caption(f"Percentile {ciss['percentile']:.0f} sur l'historique 5 ans · dernière observation : {ciss['date']} · série quotidienne, publication avec décalage.")
            st.caption("Le percentile compare le niveau courant à l'historique de la série ; il ne représente pas une probabilité de crise.")
        except Exception as exc:
            st.warning(f"BCE CISS indisponible : {type(exc).__name__}. Indicateur non qualifié.")
    with col2:
        st.markdown("**⚡ 🇫🇷 Réseau électrique — France**")
        try:
            rte = fetch_rte_grid_data()
            if rte["age_hours"] > 3:
                st.warning("Données RTE trop anciennes pour qualifier la situation actuelle")
            elif rte["deviation"] is not None and rte["deviation"] >= 5:
                st.error("Consommation supérieure de plus de 5 % à la prévision")
            elif rte["import_share"] is not None and rte["import_share"] >= 20:
                st.warning("Dépendance aux importations élevée à cet instant")
            else:
                st.success("Aucun signal de tension simple détecté sur ces mesures")
            st.metric("Consommation", f'{rte["demand"]:,.0f} MW',
                      f'{rte["deviation"]:+.1f}% vs prévision J' if rte["deviation"] is not None else "Prévision indisponible")
            m1, m2 = st.columns(2)
            m1.metric("Part importée estimée", f'{rte["import_share"]:.1f}%' if rte["import_share"] is not None else "—")
            m2.metric("Intensité carbone", f'{rte["carbon"]:.0f} gCO₂/kWh' if rte["carbon"] is not None else "—")
            st.caption(f"Dernier point : {rte['date']} · âge des données : {rte['age_hours']:.1f} h · RTE / ODRÉ, données temps réel estimées.")
            st.caption("La part importée est calculée à partir des échanges physiques nets. Ce n'est pas une mesure de marge de réserve ni une prévision de blackout.")
        except Exception as exc:
            st.warning(f"Données RTE indisponibles : {type(exc).__name__} — {str(exc)[:350]}. Indicateur non qualifié.")
            st.caption("Diagnostic technique affiché pour distinguer une API inaccessible, un quota dépassé et un changement de format des données.")
    st.caption("Couverture opérationnelle actuelle : finance zone euro (BCE CISS) et réseau électrique français (RTE). Les indicateurs opérationnels directs pour carburants, médicaments, eau, gaz, tensions sociales et opérations militaires restent à intégrer ; ces domaines ne doivent pas être considérés comme couverts par les seules actualités.")


def geopolitics_supply_dashboard(df, errors=None):
    """Regional risk dashboard: compact regional blocks, severity colors and click-to-open details."""
    st.header("🌍 Regional Risk Monitor")
    st.caption("Vue synthétique des événements ayant un impact humain, vital, social, économique ou logistique. Les articles non qualifiés ne sont pas assimilés à une absence de risque.")

    if st.button("🔄 Actualiser les flux de veille", key="wels_geo_refresh"):
        fetch_news.clear()
        st.rerun()

    c1, c2 = st.columns([1, 2])
    with c1:
        days = st.selectbox("Période", [1, 3, 7, 14, 30], index=2, key="wels_geo_supply_days")
    with c2:
        keyword = st.text_input("Filtrer (facultatif)", placeholder="Iran, carburant, eau, santé…", key="wels_geo_supply_keyword")

    data_available = df is not None and not df.empty
    if not data_available:
        st.error("Aucun article n'a été récupéré : la collecte des flux a échoué ou les flux sont vides.")
        st.warning("Le statut des régions est INCONNU. L'absence de données ne signifie pas une situation normale.")
        if errors:
            with st.expander("Diagnostic de la collecte — erreurs des flux", expanded=False):
                for error in errors:
                    st.write(f"- {error}")
        else:
            st.caption("Aucun détail d'erreur n'a été transmis par la collecte. Réessayez l'actualisation.")

    selected = df.copy() if data_available else pd.DataFrame(columns=["Date", "Titre", "Résumé", "Source", "Lien"])
    if "Date" in selected.columns:
        cutoff = datetime.utcnow() - timedelta(days=days)
        selected = selected[selected["Date"].isna() | (selected["Date"] >= cutoff)]
    if keyword.strip():
        title_text = selected.get("Titre", pd.Series("", index=selected.index)).fillna("").astype(str)
        summary_text = selected.get("Résumé", pd.Series("", index=selected.index)).fillna("").astype(str)
        selected = selected[(title_text + " " + summary_text).str.contains(re.escape(keyword.strip()), case=False, na=False)]

    events, unqualified = [], []
    for _, row in selected.iterrows():
        title = str(row.get("Titre", "") or "")
        summary = str(row.get("Résumé", "") or "")
        matches = qualify_situation(title, summary)
        if not matches:
            if len(unqualified) < 20:
                unqualified.append(row)
            continue

        date_value = row.get("Date")
        date_label = date_value.strftime("%d/%m/%Y %H:%M UTC") if isinstance(date_value, datetime) else "Date non précisée"
        article_text = f"{title} {summary}".lower()
        source_text = str(row.get("Source", "")).lower()

        # Affectation géographique à partir du contenu de l'article, puis du flux RSS.
        if any(term in article_text for term in ["france", "french", "paris", "lyon", "marseille", "français", "française"]):
            region = "France"
        elif any(term in article_text for term in ["dubai", "dubaï", "uae", "united arab emirates", "emirates", "abu dhabi", "jebel ali", "émirats"]):
            region = "Dubaï"
        elif any(term in article_text for term in ["europe", "european union", "eu member", "eurozone", "européen", "européenne", "cyprus", "italy", "germany", "poland", "ukraine", "united kingdom"]):
            region = "Europe"
        elif any(term in article_text for term in ["middle east", "iran", "iraq", "israel", "palestine", "gaza", "west bank", "yemen", "lebanon", "syria", "saudi", "qatar", "bahrain", "oman", "hormuz", "red sea"]):
            region = "Moyen-Orient"
        elif any(term in article_text for term in ["global", "worldwide", "international", "world economy", "global supply", "across countries", "pandemic"]):
            region = "International"
        elif "france" in source_text:
            region = "France"
        elif "émirats" in source_text or "dubai" in source_text:
            region = "Dubaï"
        elif "europe" in source_text:
            region = "Europe"
        elif "moyen-orient" in source_text:
            region = "Moyen-Orient"
        else:
            region = "International"

        for match in matches:
            events.append({
                **match,
                "title": title,
                "summary": summary,
                "source": str(row.get("Source", "Source non précisée")),
                "url": str(row.get("Lien", "")),
                "date": date_label,
                "date_raw": pd.to_datetime(date_value, utc=True, errors="coerce") if date_value is not None else pd.NaT,
                "region": region,
            })

    # Regrouper les reprises d'un même événement : la diversité des sources
    # apporte un élément de corroboration, sans transformer un article en preuve.
    unique_events = {}
    for event in events:
        key = (event["id"], re.sub(r"[^a-z0-9]+", " ", event["title"].lower())[:100])
        if key not in unique_events:
            event["sources"] = [event["source"]] if event["source"] else []
            event["source_links"] = [{"source": event["source"], "url": event["url"]}] if event["url"].startswith("http") else []
            unique_events[key] = event
        else:
            current = unique_events[key]
            if event["source"] and event["source"] not in current["sources"]:
                current["sources"].append(event["source"])
            if event["url"].startswith("http") and not any(link["url"] == event["url"] for link in current.get("source_links", [])):
                current.setdefault("source_links", []).append({"source": event["source"], "url": event["url"]})
            if event["score"] > current["score"]:
                current["score"] = event["score"]
            if event.get("date_raw") and (not current.get("date_raw") or event["date_raw"] > current["date_raw"]):
                current["date_raw"] = event["date_raw"]
    events = list(unique_events.values())
    for event in events:
        event["source_count"] = len(event.get("sources", []))

    # Familles de risques : catégories opérationnelles, pas prédictions.
    risk_domains = {
        "Guerre / escalade": {"war_escalation", "shipping_chokepoint", "sanctions_energy", "airspace_airport", "humanitarian_crisis"},
        "Crise financière": {"banking_capital", "sovereign_financial"},
        "Cyberattaques / services essentiels": {"cyber_critical"},
        "Blackout / infrastructures": {"power_disruption", "cyber_critical", "water_desalination"},
        "Pénuries / approvisionnement": {"fuel_logistics_strike", "fuel_shortage", "refinery_outage", "transport_strike", "medicine_shortage", "food_supply", "gas_market_tightness"},
        "Tensions sociales": {"mass_social_disruption", "transport_strike", "homecare_workforce", "fuel_logistics_strike", "fuel_shortage"},
    }
    for event in events:
        event["domains"] = [name for name, ids in risk_domains.items() if event["id"] in ids]

    def pressure_index(items):
        """Transparent 0-100 signal-pressure index; explicitly NOT a probability."""
        if not items:
            return 0, 0, 0
        now = pd.Timestamp.now(tz="UTC")
        contributions = []
        sources = set()
        for item in items:
            raw_date = item.get("date_raw")
            if raw_date is None or pd.isna(raw_date):
                age_days = 30
            else:
                if raw_date.tzinfo is None:
                    raw_date = raw_date.tz_localize("UTC")
                age_days = max(0, (now - raw_date).total_seconds() / 86400)
            freshness = 1.0 if age_days <= 1 else (0.8 if age_days <= 3 else (0.55 if age_days <= 7 else 0.25))
            base = min(48, int(item.get("score", 0)) * 12) * freshness
            corroboration = min(12, max(0, int(item.get("source_count", 1)) - 1) * 6)
            contributions.append(base + corroboration)
            sources.update(item.get("sources", []))
        contributions.sort(reverse=True)
        persistence = min(15, max(0, len(items) - 1) * 5)
        value = min(100, round(contributions[0] + sum(contributions[1:3]) * 0.30 + persistence))
        return value, len(items), len(sources)

    def pressure_label(value):
        if value >= 80: return "TRÈS ÉLEVÉ"
        if value >= 60: return "ÉLEVÉ"
        if value >= 40: return "À SURVEILLER"
        if value >= 20: return "FAIBLE"
        return "BAS — PEU DE SIGNAUX"

    # International ne conserve que les événements qualifiés par une règle d'impact ;
    # les articles mondiaux sans lien concret restent dans la file de revue.
    severity_order = ["CRITIQUE", "ÉLEVÉ", "VIGILANCE", "INFORMATION", "ATTÉNUATION"]
    severity_meta = {
        "CRITIQUE": ("🔴", "#dc2626", "Impact majeur / rupture confirmée à vérifier"),
        "ÉLEVÉ": ("🟠", "#ea580c", "Perturbation potentiellement importante"),
        "VIGILANCE": ("🟡", "#ca8a04", "Pression émergente ou impact indirect possible"),
        "INFORMATION": ("🔵", "#2563eb", "Évolution à suivre, impact encore limité ou incertain"),
        "ATTÉNUATION": ("🟢", "#16a34a", "Évolution pouvant réduire les tensions"),
    }

    def severity(event):
        if event["polarity"] == "positive":
            return "ATTÉNUATION"
        score = int(event.get("score", 0))
        if score >= 4:
            return "CRITIQUE"
        if score >= 3:
            return "ÉLEVÉ"
        if score == 2:
            return "VIGILANCE"
        return "INFORMATION"

    for event in events:
        event["severity"] = severity(event)

    regions = ["France", "Dubaï", "Europe", "Moyen-Orient", "International"]
    region_icons = {"France": "🇫🇷", "Dubaï": "🇦🇪", "Europe": "🇪🇺", "Moyen-Orient": "🌍", "International": "🌐"}
    domain_icons = {
        "Guerre / escalade": "⚔️",
        "Crise financière": "🏦",
        "Cyberattaques / services essentiels": "🛡️",
        "Blackout / infrastructures": "⚡",
        "Pénuries / approvisionnement": "🛢️",
        "Tensions sociales": "👥",
    }
    source_coverage = selected["Source"].nunique() if data_available and "Source" in selected.columns else 0
    st.markdown(f"**{len(events)} signaux qualifiés** · {len(selected)} articles analysés · {source_coverage} source(s) distincte(s) · fenêtre de {days} jour(s)")
    if errors:
        st.caption(f"Collecte partielle : {len(errors)} flux en erreur. Les jauges doivent être lues avec cette limite.")
    st.caption("CRITIQUE = événement majeur à vérifier ; ÉLEVÉ = risque important ; VIGILANCE = pression émergente ; INFORMATION = suivi ; ATTÉNUATION = signal positif.")

    # Indices par famille de risque. Ils synthétisent uniquement les signaux RSS
    # qualifiés et leur recoupement ; ils ne sont jamais des probabilités de crise.
    operational_risk_panel()

    st.subheader("Jauges de pression — signaux précurseurs observés")
    st.caption("Chaque jauge est accompagnée des événements qui expliquent son niveau. Le score 0–100 mesure la pression documentaire (gravité, fraîcheur, corroboration et répétition), pas une probabilité de crise.")

    def event_pressure_contribution(item):
        raw_date = item.get("date_raw")
        if raw_date is None or pd.isna(raw_date):
            age_days = 30
        else:
            if raw_date.tzinfo is None:
                raw_date = raw_date.tz_localize("UTC")
            age_days = max(0, (pd.Timestamp.now(tz="UTC") - raw_date).total_seconds() / 86400)
        freshness = 1.0 if age_days <= 1 else (0.8 if age_days <= 3 else (0.55 if age_days <= 7 else 0.25))
        base = min(48, int(item.get("score", 0)) * 12) * freshness
        corroboration = min(12, max(0, int(item.get("source_count", 1)) - 1) * 6)
        return round(base + corroboration), age_days

    gauge_cols = st.columns(3)
    for index, domain in enumerate(risk_domains):
        domain_events = [event for event in events if domain in event.get("domains", [])]
        value, event_count, source_count = pressure_index(domain_events)
        if not data_available:
            bg, fg, label = "#EEF1F5", "#667085", "INCONNU"
        elif not domain_events:
            bg, fg, label = "#EEF1F5", "#667085", "AUCUN SIGNAL QUALIFIÉ"
        elif value >= 60:
            bg, fg, label = "#FCE8E8", "#B42318", pressure_label(value)
        elif value >= 40:
            bg, fg, label = "#FFF3D9", "#A86100", pressure_label(value)
        elif value >= 20:
            bg, fg, label = "#EAF1FB", "#315B91", pressure_label(value)
        else:
            bg, fg, label = "#E5F4E9", "#237A45", pressure_label(value)

        ranked_events = []
        for event in domain_events:
            contribution, age_days = event_pressure_contribution(event)
            ranked_events.append((contribution, age_days, event))
        ranked_events.sort(key=lambda item: (item[0], item[2].get("score", 0), item[2].get("source_count", 1)), reverse=True)

        with gauge_cols[index % 3]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {fg}55;border-radius:13px;padding:13px;min-height:145px">
              <div style="font-size:14px;font-weight:800;color:#344054;min-height:35px"><span style="font-size:22px;margin-right:7px">{domain_icons.get(domain, "📌")}</span>{domain}</div>
              <div style="font-size:29px;font-weight:900;color:{fg};margin:5px 0">{value if data_available else "—"}<span style="font-size:13px">/100</span></div>
              <div style="height:9px;background:#DDE3EB;border-radius:8px;overflow:hidden"><div style="height:9px;width:{value if data_available else 0}%;background:{fg};border-radius:8px"></div></div>
              <div style="font-size:12px;font-weight:700;color:{fg};margin-top:9px">{label}</div>
              <div style="font-size:11px;color:#667085">{event_count} événement(s) · {source_count} source(s)</div>
            </div>""", unsafe_allow_html=True)

            # Top events are visible without opening a panel.
            if not data_available:
                st.warning("Données indisponibles : score non qualifiable.")
            elif not ranked_events:
                st.caption("Aucun événement qualifié dans cette famille sur la période. Cela ne signifie pas une absence de risque.")
            else:
                st.markdown("**Événements principaux à l'origine du score**")
                for contribution, age_days, event in ranked_events[:2]:
                    st.markdown(f"**{str(event.get('title', 'Événement sans titre'))[:150]}**")
                    age_label = "âge inconnu" if age_days >= 29 else (f"il y a {age_days * 24:.0f} h" if age_days < 1 else f"il y a {age_days:.1f} j")
                    st.caption(f"{event.get('date', 'Date inconnue')} · {event.get('severity', 'Non classé')} · {age_label} · {event.get('source_count', 1)} source(s) · poids indicatif {contribution}/60")
                    if event.get("summary"):
                        st.write(str(event["summary"])[:420])
                    if event.get("impact"):
                        st.markdown(f"**Pourquoi c'est important :** {event['impact']}")
                    for link in event.get("source_links", [])[:4]:
                        if str(link.get("url", "")).startswith(("https://", "http://")):
                            st.markdown(f"[{link.get('source', 'Ouvrir la source')}]({link['url']})")
                    st.divider()

                if len(ranked_events) > 2:
                    with st.expander(f"Voir les {len(ranked_events) - 2} autres événements et le détail du calcul", expanded=False):
                        st.caption("Le poids indicatif combine gravité, fraîcheur et corroboration. Le score global ajoute aussi un bonus de répétition : les poids individuels ne sont donc pas des parts exactes du total.")
                        for contribution, age_days, event in ranked_events[2:]:
                            st.markdown(f"**{event.get('title', 'Événement sans titre')}**")
                            st.caption(f"{event.get('date', 'Date inconnue')} · {event.get('severity', 'Non classé')} · poids indicatif {contribution}/60")
                            if event.get("summary"):
                                st.write(str(event["summary"])[:700])
                            if event.get("impact"):
                                st.markdown(f"**Impact potentiel :** {event['impact']}")
                            for link in event.get("source_links", [])[:4]:
                                if str(link.get("url", "")).startswith(("https://", "http://")):
                                    st.markdown(f"[{link.get('source', 'Ouvrir la source')}]({link['url']})")
                            st.divider()

    st.caption("Lecture : 0–19 bas, 20–39 faible, 40–59 à surveiller, 60–79 élevé, 80–100 très élevé. Les seuils sont des conventions de veille. Le poids indicatif affiché par événement n'est pas une part exacte du score total : le score inclut aussi un bonus de répétition.")
    
    # Regional widgets are the default view. Event titles are only rendered after a region is selected.
    selected_region = st.session_state.get("wels_selected_risk_region")
    if selected_region not in regions:
        selected_region = None

    region_cols = st.columns(2)
    for index, region in enumerate(regions):
        regional = [event for event in events if event["region"] == region]
        counts = {level: sum(1 for event in regional if event["severity"] == level) for level in severity_order}
        highest = next((level for level in severity_order if counts[level] > 0), None)
        regional_score, regional_count, regional_sources = pressure_index(regional)

        if not data_available:
            status = "INCONNU"
            status_icon = "⚪"
        elif highest:
            status = highest
            status_icon = severity_meta[highest][0]
        else:
            status = "AUCUN SIGNAL QUALIFIÉ"
            status_icon = "⚪"

        count_label = f"{len(regional)} événement(s)" if data_available else "Données indisponibles"
        breakdown = " · ".join(f"{level}: {counts[level]}" for level in severity_order if counts[level])
        gauge_label = f"Indice signaux {regional_score}/100" if data_available else "Indice INCONNU"
        button_label = f"{region_icons.get(region, '🌐')}  {status_icon}  {region}\n{status} · {count_label}\n{gauge_label}"
        with region_cols[index % 2]:
            st.button(
                button_label,
                key=f"wels_region_widget_{index}",
                use_container_width=True,
                on_click=lambda r=region: st.session_state.update({"wels_selected_risk_region": r}),
            )
            if breakdown:
                st.caption(breakdown)
            elif data_available:
                st.caption("Aucun événement répondant aux critères de qualification. Cela ne prouve pas l'absence de risque.")

    if selected_region is None:
        st.info("Sélectionne une région pour consulter les événements qualifiés et leurs conséquences.")
    else:
        st.divider()
        back_col, title_col = st.columns([1, 3])
        with back_col:
            if st.button("← Toutes les régions", key="wels_back_to_regions"):
                st.session_state["wels_selected_risk_region"] = None
                st.rerun()
        with title_col:
            st.subheader(f"{region_icons.get(selected_region, '🌐')} Événements — {selected_region}")

        regional = [event for event in events if event["region"] == selected_region]
        if not data_available:
            st.warning("INCONNU — aucune donnée disponible pour évaluer cette région.")
        elif not regional:
            st.info("Aucun événement qualifié pour cette région sur la période. Cela ne garantit pas l'absence de risque.")
        else:
            for level in severity_order:
                level_events = sorted(
                    [event for event in regional if event["severity"] == level],
                    key=lambda item: (item["date"], item["score"]),
                    reverse=True
                )
                if not level_events:
                    continue
                icon, level_color, description = severity_meta[level]
                st.markdown(
                    f"<div style='color:{level_color};font-weight:700;margin:14px 0 6px'>{icon} {level} — {description}</div>",
                    unsafe_allow_html=True
                )
                for event in level_events:
                    short_title = event["title"]
                    if len(short_title) > 115:
                        short_title = short_title[:112].rstrip() + "…"
                    with st.expander(short_title, expanded=False):
                        st.caption(f"{event['event']} · {event['date']} · {event['source']} · Fiabilité : {event['confidence']}")
                        if event["summary"]:
                            st.write(event["summary"][:900])
                        st.markdown(f"**Impact humain / quotidien :** {event['impact']}")
                        st.markdown(f"**Transmission économique / marchés :** {event['market']}")
                        st.markdown(f"**À faire / à vérifier :** {event['action']}")
                        if event["url"].startswith("http"):
                            st.markdown(f"[Ouvrir la source]({event['url']})")

    # La file de revue n'est affichée qu'après l'ouverture d'une région.
    # Les expandeurs Streamlit ne peuvent pas être imbriqués : les articles
    # sont donc présentés dans des conteneurs simples à l'intérieur du bloc.
    if unqualified and selected_region is not None:
        with st.expander(f"Articles à examiner — non qualifiés automatiquement ({len(unqualified)})", expanded=False):
            st.caption("Articles récupérés mais non reconnus par les règles actuelles. Ils ne sont pas considérés comme sans importance ; cette zone permet de repérer les angles morts.")
            for index, row in enumerate(unqualified):
                title = str(row.get("Titre", "Titre indisponible"))
                with st.container():
                    st.markdown(f"**{index + 1}. {title[:140]}**")
                    summary = str(row.get("Résumé", "") or "").strip()
                    if summary:
                        st.write(summary[:700])
                    url = str(row.get("Lien", "") or "")
                    if url.startswith("http"):
                        st.markdown(f"[Ouvrir la source]({url})")
                    if index < len(unqualified) - 1:
                        st.divider()

    st.caption("Les couleurs indiquent une priorité de veille, pas une probabilité statistique. Les jauges sont des indices internes de pression documentaire calculés sur les articles qualifiés ; elles ne prédisent pas la survenue d'une guerre, d'un blackout, d'une pénurie ou d'une crise financière. Un article ne confirme pas à lui seul la réalité, l'étendue ou la durée d'un impact. Les événements internationaux ne sont affichés que lorsqu'une règle d'impact les qualifie.")


def _newsletter_api_key():
    """Read the AI provider key from Streamlit secrets or the hosting environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return str(key or os.environ.get("GROQ_API_KEY", "")).strip()


@st.cache_data(ttl=5400, show_spinner=False)
def _generate_ai_newsletter(articles_json, api_key, refresh_id):
    """Generate a source-grounded French briefing; cache by exact source corpus."""
    articles = json.loads(articles_json)
    if not articles:
        return ""
    system_prompt = (
        "Tu es analyste senior macroéconomie, marchés, géopolitique et risques systémiques. "
        "Rédige en français un briefing stratégique précis, nuancé et utile à un investisseur particulier. "
        "Utilise uniquement les éléments fournis. N'invente aucun cours, chiffre, date, citation, causalité "
        "ou événement. Distingue explicitement faits rapportés, interprétations et hypothèses. "
        "Quand les sources divergent ou sont insuffisantes, dis-le. Un titre ou un résumé RSS ne constitue "
        "pas une confirmation. Ne transforme jamais une absence d'information en absence de risque. "
        "Ne donne pas d'ordre automatique d'achat ou de vente."
    )
    user_prompt = (
        "À partir des articles ci-dessous, produis un briefing avec :\n"
        "1. Synthèse exécutive (5 points maximum, importance décroissante).\n"
        "2. Analyse approfondie des événements majeurs : ce qui s'est passé, mécanismes économiques/financiers, "
        "éléments confirmés et inconnues.\n"
        "3. Transmission aux marchés : actions, obligations/taux, dollar, EUR/CHF, pétrole/énergie, or et Bitcoin "
        "(uniquement si les sources permettent une analyse pertinente).\n"
        "4. Gagnants/perdants potentiels par secteurs, avec les conditions et risques associés.\n"
        "5. Scénarios à 1-4 semaines : central, dégradation, détente ; déclencheurs observables, sans probabilités "
        "chiffrées non justifiées.\n"
        "6. Implications pour un portefeuille diversifié exposé aux actions mondiales, énergie, défense, santé, "
        "utilities/électricité, or et liquidités. Pas de conseil personnalisé ni d'ordre de transaction.\n"
        "7. Tableau de surveillance : indicateur, pourquoi il compte, signal à surveiller.\n"
        "8. Qualité documentaire : nombre de sources distinctes, corroborations, limites et niveau de confiance "
        "(élevé/moyen/faible, justifié).\n\n"
        "Référence chaque affirmation importante avec les identifiants [S1], [S2]… fournis. "
        "N'affirme pas qu'une séance boursière a chuté ou qu'un mouvement est historique sans données de marché "
        "chiffrées dans les sources. Si le corpus est trop faible, indique clairement les limites et réduis la portée.\n\n"
        + json.dumps(articles, ensure_ascii=False)
    )
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "openai/gpt-oss-120b",
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def ai_newsletter_page(df, errors):
    """Automatically create a deep AI briefing from the latest qualified news corpus."""
    st.header("🧠 Newsletter stratégique — analyse IA")
    st.caption(
        "Briefing généré automatiquement à partir des actualités collectées : analyse macroéconomique, "
        "géopolitique, marchés, scénarios et implications patrimoniales."
    )
    if df is None or df.empty:
        st.warning("Aucune actualité exploitable : le briefing ne peut pas être généré.")
        if errors:
            with st.expander("Diagnostic des flux"):
                for error in errors:
                    st.write(f"• {error}")
        return

    api_key = _newsletter_api_key()
    if not api_key:
        st.error(
            "La génération IA gratuite n'est pas encore activée. Crée une clé Groq gratuite et ajoute GROQ_API_KEY dans "
            "les secrets de l'application Streamlit (ou comme variable d'environnement), puis actualise le dashboard."
        )
        st.caption("La clé n'est jamais affichée dans le dashboard. Modèle utilisé : openai/gpt-oss-120b (offre gratuite Groq, soumise à quotas).")
        return

    recent = df.copy()
    if "Date" in recent.columns:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=72)
        recent = recent[recent["Date"].isna() | (recent["Date"] >= cutoff)]
    if recent.empty:
        st.info("Pas de nouvel article sur les dernières 72 heures. Aucun événement ne sera inventé.")
        return

    # Prioritize distinct publishers and deduplicate near-identical headlines.
    recent = recent.sort_values("Date_tri", ascending=False, na_position="last") if "Date_tri" in recent else recent
    recent = recent.drop_duplicates(subset=["Lien"]).head(6)
    articles = []
    with st.spinner(f"Lecture des sources et préparation du briefing ({len(recent)} articles)…"):
        for index, (_, row) in enumerate(recent.iterrows(), start=1):
            url = str(row.get("Lien", "") or "")
            body = fetch_article_text(url) if url.startswith("http") else ""
            summary = str(row.get("Résumé", "") or "")
            content = body[:2200] if body else summary[:1200]
            articles.append({
                "source_id": f"S{index}",
                "date": str(row.get("Date", "")),
                "publisher": str(row.get("Source", row.get("Domaine", "Source inconnue"))),
                "title": str(row.get("Titre", ""))[:400],
                "content": content,
                "content_quality": "texte intégral extrait" if body else "résumé RSS uniquement",
                "url": url,
            })

    distinct_sources = len({a["publisher"] for a in articles})
    source_digest = hashlib.sha256(
        json.dumps(articles, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    st.caption(
        f"Corpus : {len(articles)} articles · {distinct_sources} sources éditoriales · "
        f"fenêtre : 72 h · Référence du corpus : {source_digest[:10]}"
    )
    try:
        with st.spinner("L'IA recoupe les informations et rédige l'analyse stratégique…"):
            briefing = _generate_ai_newsletter(
                json.dumps(articles, ensure_ascii=False, sort_keys=True), api_key,
                f"{int(time.time() // 600)}:{st.session_state.get('wels_ai_refresh_id', 'initial')}"
            )
        if briefing:
            st.markdown(briefing)
        else:
            st.warning("Le modèle n'a pas retourné de contenu exploitable.")
    except requests.exceptions.Timeout:
        st.error("Le service IA n'a pas répondu dans le délai prévu. Le briefing sera retenté à la prochaine actualisation.")
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", "inconnu")
        st.error(f"Erreur du service IA (HTTP {status}). Vérifie la clé Groq et les quotas gratuits. Ne passe pas à une offre payante.")
    except Exception as exc:
        st.error(f"Génération IA indisponible : {type(exc).__name__}. Le dashboard et les autres modules restent accessibles.")

    with st.expander("Sources utilisées et qualité des contenus", expanded=False):
        for article in articles:
            quality = article["content_quality"]
            st.markdown(
                f"**[{article['source_id']}] {article['publisher']}** · {article['date']} · {quality}  \\n"
                f"[{article['title']}]({article['url']})"
            )
        st.caption(
            "Les éditeurs ne permettent pas tous l'extraction du texte intégral. "
            "Les éléments fondés uniquement sur un résumé RSS sont donc moins solides. "
            "Les liens ci-dessus permettent de vérifier les articles originaux."
        )

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


def render_drawdown_alerts():
    """Highlight sharp portfolio drawdowns using the latest three daily closes."""
    st.subheader("🚨 Alertes de baisse — positions à surveiller")
    st.caption("Détection automatique sur les deux dernières séances disponibles. Rouge : baisse ≥ 8 % · orange : baisse de 4 à 8 %. Une alerte signale un mouvement, pas un ordre de vente.")
    assets = [
        {"name": "FTAI Aviation", "ticker": "FTAI", "currency": "USD", "pe": "32,93", "forward_pe": "15,07",
         "fundamentals_date": "05/11/2026",
         "note": "Le forward P/E repose sur des estimations bénéficiaires futures, pas sur un bénéfice déjà réalisé."},
        {"name": "Lockheed Martin", "ticker": "LMT", "currency": "USD", "pe": "≈ 20,5", "forward_pe": "≈ 18",
         "fundamentals_date": "05/11/2026",
         "note": "Valorisation indicative rapportée au 05/11/2026 ; à revalider avec les résultats et le cours du jour."}
    ]
    cols = st.columns(2)
    for idx, asset in enumerate(assets):
        hist = fetch_history(asset["ticker"], "1mo")
        year = fetch_history(asset["ticker"], "1y")
        if hist.empty or len(hist) < 3:
            with cols[idx]:
                st.warning(f"{asset['name']} : historique de cours insuffisant pour calculer la variation sur deux séances.")
            continue
        recent = hist.tail(3).copy()
        prices = [float(x) for x in recent["Cours"].tolist()]
        two_day = (prices[-1] / prices[0] - 1) * 100 if prices[0] else 0
        if two_day <= -8:
            bg, color, status = "#FCE4E4", "#C62828", "BAISSE FORTE"
        elif two_day <= -4:
            bg, color, status = "#FFF0D5", "#A86100", "BAISSE À SURVEILLER"
        elif two_day >= 4:
            bg, color, status = "#E1F3E7", "#237A45", "REBOND"
        else:
            bg, color, status = "#EEF1F5", "#536273", "MOUVEMENT LIMITÉ"
        low52 = float(year["Cours"].min()) if not year.empty else None
        high52 = float(year["Cours"].max()) if not year.empty else None
        proximity = ((prices[-1] - low52) / (high52 - low52) * 100) if low52 is not None and high52 and high52 > low52 else None
        rows_html = ""
        for _, row in recent.iterrows():
            date_label = pd.to_datetime(row["Date"]).strftime("%d %b")
            rows_html += f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #D8DEE8;color:#344054'><span>{date_label}</span><b>{float(row['Cours']):,.2f} {asset['currency']}</b></div>"
        range_html = f"{low52:,.2f} – {high52:,.2f} {asset['currency']}" if low52 is not None and high52 is not None else "Indisponible"
        proximity_html = f"Position dans la fourchette : {proximity:.0f} % au-dessus du plus bas" if proximity is not None else ""
        with cols[idx]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {color}80;border-radius:14px;padding:15px;margin:4px 0 10px">
              <div style="font-size:17px;font-weight:800;color:#263445">{asset['name']}</div>
              <div style="font-size:28px;font-weight:900;color:{color};margin:6px 0">{two_day:+.1f} %</div>
              <div style="font-size:13px;font-weight:800;color:{color}">{status} · 2 dernières séances</div>
              <div style="margin-top:12px">{rows_html}</div>
              <div style="margin-top:10px;color:#344054;font-size:13px"><b>Fourchette 52 semaines :</b> {range_html}<br>{proximity_html}</div>
              <div style="margin-top:10px;color:#344054;font-size:13px"><b>P/E :</b> {asset['pe']} · <b>Forward P/E :</b> {asset['forward_pe']}<br><span style="color:#667085">{asset['note']}</span></div>
              <div style="margin-top:8px;color:#667085;font-size:11px">Cours : Yahoo Finance · fondamentaux indicatifs : Stock Analysis, référence {asset['fundamentals_date']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Les variations, dates, cours et fourchettes 52 semaines sont recalculés à chaque actualisation depuis Yahoo Finance. Les multiples P/E affichés sont des références datées et doivent être actualisés séparément ; ils ne sont pas présentés comme des données temps réel.")

def executive_dashboard(df):
    """Executive view: common quote source, portfolio watch and candidate supports."""
    st.markdown("""
    <style>
    .risk-shell{background:#F4F6FA;border:1px solid #DCE3ED;border-radius:18px;padding:18px 16px 14px;margin:4px 0 14px}
    .risk-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
    .risk-title{font-size:22px;font-weight:800;color:#263445}.risk-sub{font-size:14px;color:#667085;margin-top:4px}
    .risk-pill{border-radius:20px;padding:6px 12px;font-size:13px;font-weight:700}
    .risk-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:16px}
    .risk-card{border-radius:16px;padding:14px;min-height:112px}
    .risk-card.red{background:#FCE8E8}.risk-card.amber{background:#FFF3D9}.risk-card.green{background:#E5F4E9}
    .risk-label{font-size:16px;font-weight:700;color:#344054}.risk-level{font-size:20px;font-weight:800;margin-top:8px}
    .risk-detail{font-size:13px;color:#596579;margin-top:8px}
    @media(max-width:650px){.risk-grid{grid-template-columns:1fr}}
    </style>
    """, unsafe_allow_html=True)

    def rows_for(topic):
        if df is None or df.empty or "Thèmes" not in df.columns:
            return pd.DataFrame()
        return df[df["Thèmes"].apply(lambda tags: topic in tags)]

    war = rows_for("Géopolitique / conflit")
    supply = rows_for("Approvisionnement")
    chf = rows_for("Suisse / CHF")

    def signal_level(rows):
        if rows.empty:
            return "AUCUN SIGNAL QUALIFIÉ"
        # Article volume alone must never create a high-risk status.
        if "Priorité" in rows.columns and (rows["Priorité"] == "À examiner en priorité").any():
            return "À QUALIFIER EN PRIORITÉ"
        return "À SURVEILLER"

    war_level = signal_level(war)
    france_level = signal_level(supply)
    snapshot = [
        ("Brent", "BZ=F", "USD"), ("Or (€/g)", "GC=F", "EUR/g"),
        ("Nasdaq Composite", "^IXIC", "Index"), ("S&P 500", "^GSPC", "Index"),
        ("Dow Jones", "^DJI", "Index"), ("EUR/CHF", "EURCHF=X", "FX"),
        ("EUR/AED", "EURAED=X", "FX"), ("Bitcoin", "BTC-USD", "USD")
    ]
    quote_rows, market_moves, kpi_rows = [], {}, []
    for label, symbol, currency in snapshot:
        hist = fetch_gold_eur_gram("1mo") if label == "Or (€/g)" else fetch_history(symbol, "1mo")
        if hist.empty:
            quote_rows.append({"Indicateur": label, "Dernier point": "Indisponible", "Variation séance": "—", "Variation 5 séances": "—", "Date": "—", "Source": "Yahoo Finance"})
            kpi_rows.append({"label": label, "value": "Indisponible", "daily": None, "weekly": None, "date": "—", "currency": currency})
            continue
        last = float(hist["Cours"].iloc[-1])
        previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
        week_base = float(hist["Cours"].iloc[-6]) if len(hist) > 5 else float(hist["Cours"].iloc[0])
        delta = (last / previous - 1) * 100 if previous else 0.0
        weekly = (last / week_base - 1) * 100 if week_base else 0.0
        market_moves[label] = delta
        date_label = pd.to_datetime(hist["Date"].iloc[-1]).strftime("%d/%m/%Y")
        value = f"{last:,.4f}" if currency == "FX" else (f"{last:,.2f} € /g" if currency == "EUR/g" else f"{last:,.2f}" + (f" {currency}" if currency == "USD" else ""))
        quote_rows.append({"Indicateur": label, "Dernier point": value, "Variation séance": f"{delta:+.2f}%", "Variation 5 séances": f"{weekly:+.2f}%", "Date": date_label, "Source": "Yahoo Finance"})
        kpi_rows.append({"label": label, "value": value, "daily": delta, "weekly": weekly, "date": date_label, "currency": currency})

    market_level = "VOLATIL" if any(abs(v) >= 2.0 for v in market_moves.values()) else ("CALME / MODÉRÉ" if market_moves else "INDISPONIBLE")
    if war_level == "À QUALIFIER EN PRIORITÉ" or france_level == "À QUALIFIER EN PRIORITÉ":
        global_level, pill_style = "ÉVÉNEMENT À QUALIFIER", "background:#FCE8DC;color:#9A3412"
    elif war_level == "À SURVEILLER" or france_level == "À SURVEILLER" or market_level == "VOLATIL":
        global_level, pill_style = "VIGILANCE RENFORCÉE", "background:#FFF1D6;color:#8A5200"
    elif not market_moves:
        global_level, pill_style = "DONNÉES INCOMPLÈTES", "background:#EEF1F5;color:#475467"
    else:
        global_level, pill_style = "RISQUE MODÉRÉ", "background:#E3F3E8;color:#216E3A"

    stamp = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M %Z")
    def card_class(level):
        return "red" if level == "À QUALIFIER EN PRIORITÉ" else ("green" if level == "AUCUN SIGNAL QUALIFIÉ" else "amber")
    def level_color(level):
        return "#B42318" if level == "À QUALIFIER EN PRIORITÉ" else ("#237A45" if level == "AUCUN SIGNAL QUALIFIÉ" else "#A86100")

    st.markdown(f"""
    <div class="risk-shell">
      <div class="risk-head"><div><div class="risk-title">GLOBAL RISK MONITOR</div>
      <div class="risk-sub">Market, War, CHF & New Themes · Mis à jour {stamp}</div></div>
      <div class="risk-pill" style="{pill_style}">{global_level}</div></div>
      <div class="risk-grid">
        <div class="risk-card {card_class(war_level)}"><div class="risk-label">⚔️ Géopolitique</div>
          <div class="risk-level" style="color:{level_color(war_level)}">{war_level}</div>
          <div class="risk-detail">{len(war)} article(s) classé(s) · conflits, sanctions, routes maritimes</div></div>
        <div class="risk-card {'amber' if market_level == 'VOLATIL' else 'green'}"><div class="risk-label">📈 Marchés</div>
          <div class="risk-level" style="color:{'#ffbd55' if market_level == 'VOLATIL' else '#43d17a'}">{market_level}</div>
          <div class="risk-detail">Derniers points quotidiens disponibles</div></div>
        <div class="risk-card {card_class(france_level)}"><div class="risk-label">🇫🇷 🛢️ France / approvisionnement</div>
          <div class="risk-level" style="color:{level_color(france_level)}">{france_level}</div>
          <div class="risk-detail">{len(supply)} article(s) classé(s) · pénuries, énergie, transports, produits essentiels</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_drawdown_alerts()
    st.subheader("📊 Marchés & indicateurs macro — vue rapide")
    market_icons = {
        "Or": "🥇", "Or (€/g)": "🥇", "Bitcoin": "₿", "Nasdaq Composite": "💻",
        "S&P 500": "🇺🇸", "Dow Jones": "🏛️", "EUR/CHF": "🇨🇭",
        "EUR/AED": "🇦🇪", "Pétrole": "🛢️", "Brent": "🛢️",
    }
    st.caption("Les tuiles affichent le dernier cours disponible, la variation de la dernière séance et celle des cinq dernières séances. Les couleurs signalent un mouvement ou une zone de change à surveiller, pas une prévision.")
    kpi_cols = st.columns(4)
    for idx, item in enumerate(kpi_rows):
        label, daily, weekly = item["label"], item["daily"], item["weekly"]
        assessment = None
        if daily is None:
            bg, accent, status = "#EEF1F5", "#667085", "DONNÉES INDISPONIBLES"
        elif label in ("EUR/CHF", "EUR/AED"):
            fx_symbol = "EURCHF=X" if label == "EUR/CHF" else "EURAED=X"
            assessment = fx_conversion_assessment(fx_symbol)
            if assessment:
                bg, accent, status = assessment["bg"], assessment["color"], f"{assessment['label']} · {assessment['percentile']:.0f}% annuel"
            else:
                bg, accent, status = "#EEF1F5", "#667085", "ÉVALUATION INDISPONIBLE"
        elif label in ("Nasdaq Composite", "S&P 500", "Dow Jones"):
            bg, accent = ("#FCE8E8", "#B42318") if weekly <= -4 else (("#FFF3D9", "#A86100") if weekly <= -2 else ("#E5F4E9", "#237A45"))
            status = "CORRECTION / FORTE BAISSE" if weekly <= -4 else ("BAISSE À SURVEILLER" if weekly <= -2 else ("REBOND" if weekly > 1 else "MOUVEMENT MODÉRÉ"))
        else:
            bg, accent = ("#FFF3D9", "#A86100") if abs(weekly) >= 5 or abs(daily) >= 1.5 else ("#EEF1F5", "#536273")
            status = "FORTE VOLATILITÉ" if abs(weekly) >= 5 or abs(daily) >= 1.5 else "MOUVEMENT MODÉRÉ"
        with kpi_cols[idx % 4]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {accent}80;border-radius:13px;padding:13px;margin:0 0 10px;min-height:142px">
              <div style="font-size:14px;font-weight:800;color:#344054"><span style="font-size:20px;margin-right:6px">{market_icons.get(label, "📈")}</span>{label}</div>
              <div style="font-size:25px;font-weight:900;color:#263445;margin:8px 0 3px">{item['value']}</div>
              <div style="font-size:14px;font-weight:800;color:{accent}">{'—' if daily is None else f'{daily:+.2f}%'} séance</div>
              <div style="font-size:13px;color:#596579;margin-top:3px">{'—' if weekly is None else f'{weekly:+.2f}%'} / 5 séances</div>
              <div style="font-size:11px;color:#667085;margin-top:7px">{status} · {item['date']}</div>
              {f"<div style='font-size:12px;color:#344054;margin-top:4px'>1 000 € ≈ {assessment['amount']:,.2f} {('CHF' if label == 'EUR/CHF' else 'AED')}</div>" if label in ("EUR/CHF", "EUR/AED") and assessment else ""}
            </div>
            """, unsafe_allow_html=True)
    with st.expander("Voir le tableau détaillé des cours et variations"):
        st.dataframe(pd.DataFrame(quote_rows), use_container_width=True, hide_index=True)
    st.caption("Source : Yahoo Finance · cache 5 min · données quotidiennes, pas un flux intraday garanti. Or : estimation EUR/g calculée depuis les futures COMEX USD/once convertis via EUR/USD, pas un fixing spot officiel. Change : percentile de la distribution sur 1 an, hors frais bancaires.")

    with st.expander("Portefeuille détenu — suivi des cours", expanded=False):
        held = [
            ("TotalEnergies", "TTE.PA", "EUR"), ("Exxon Mobil", "XOM", "USD"),
            ("Vanguard FTSE All-World Acc (VWCE)", "VWCE.DE", "EUR"),
            ("Frontline", "FRO", "USD"), ("FTAI Aviation", "FTAI", "USD"),
            ("Thales", "HO.PA", "EUR"), ("Lockheed Martin", "LMT", "USD"),
            ("Air Liquide", "AI.PA", "EUR"), ("Amundi Health Care (WELS)", "WELS.DE", "EUR"),
            ("World Utilities (XDWU)", "XDWU.L", "GBP")
        ]
        held_rows = []
        for name, symbol, currency in held:
            hist = fetch_history(symbol, "5d")
            if hist.empty:
                held_rows.append({"Position": name, "Ticker": symbol, "Cours": "Indisponible", "Devise": currency, "Variation séance": "—", "Date": "—"})
                continue
            last = float(hist["Cours"].iloc[-1])
            previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
            change = (last / previous - 1) * 100 if previous else 0.0
            held_rows.append({"Position": name, "Ticker": symbol, "Cours": round(last, 4), "Devise": currency,
                              "Variation séance": f"{change:+.2f}%", "Date": pd.to_datetime(hist["Date"].iloc[-1]).strftime("%d/%m/%Y")})
        st.dataframe(pd.DataFrame(held_rows), use_container_width=True, hide_index=True)
        st.caption("WELS et XDWU sont affichés comme expositions thématiques déjà détenues. Cours potentiellement différés ; comparer les performances après conversion dans une devise commune.")

    with st.expander("Nouvelles opportunités — santé, électricité et industrie", expanded=False):
        candidates = [
            ("SPDR MSCI World Health Care", "WHEA.AS", "IE00BYTRRB94", "0,30% TER · capitalisant · santé mondiale"),
            ("VanEck Electrification & Power Infrastructure", "PIKA.DE", "IE000YYVSM16", "0,55% TER · lancé le 29/05/2026 · thématique concentrée"),
            ("Schneider Electric", "SU.PA", "FR0000121972", "Réseaux, gestion électrique et data centers"),
            ("Siemens", "SIE.DE", "DE0007236101", "Automatisation, électrification et industrie")
        ]
        candidate_rows = []
        for name, symbol, isin, thesis in candidates:
            hist = fetch_history(symbol, "5d")
            if hist.empty:
                candidate_rows.append({"Support": name, "Ticker": symbol, "ISIN": isin, "Dernier cours": "Indisponible", "Variation séance": "—", "Date": "—", "Thèse / vigilance": thesis})
                continue
            last = float(hist["Cours"].iloc[-1])
            previous = float(hist["Cours"].iloc[-2]) if len(hist) > 1 else last
            change = (last / previous - 1) * 100 if previous else 0.0
            candidate_rows.append({"Support": name, "Ticker": symbol, "ISIN": isin, "Dernier cours": round(last, 4),
                                   "Variation séance": f"{change:+.2f}%", "Date": pd.to_datetime(hist["Date"].iloc[-1]).strftime("%d/%m/%Y"),
                                   "Thèse / vigilance": thesis})
        st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)
        st.markdown(
            "- **SPDR santé :** exposition mondiale développée, diversifiée entre médicaments, équipements et services ; suivre concentration, valorisation, encours et frais. Comparer avec iShares (IE00BJ5JNZ06, TER indicatif 0,18%).\n"
            "- **VanEck PIKA :** ETF récent et concentré ; suivre encours, liquidité, spreads, concentration et volatilité. La croissance de la demande électrique IA ne garantit pas un rendement boursier.\n"
            "- **Schneider / Siemens :** suivre commandes data centers, marges, cash-flow, dette, valorisation et risques d'intégration. Chevauchement important avec VanEck ; éviter une concentration involontaire.\n"
            "- **Diversification :** la santé diversifie davantage les expositions énergie/industrie ; l'électrification renforce surtout le facteur industriel déjà présent via Air Liquide et l'ETF mondial."
        )

    if not chf.empty:
        st.info(f"Suisse / CHF : {len(chf)} article(s) repéré(s). Les mouvements EUR/CHF et EUR/AED figurent dans l'onglet CHF & devises.")
    st.caption("Le nombre d'articles n'est jamais utilisé seul pour déclarer une crise. Les signaux géopolitiques et d'approvisionnement doivent être recoupés ; l'absence d'article ne signifie pas absence de risque.")

def main():
    # Do not call set_page_config here: app.py configures the parent Streamlit page.
    st.title("📰 WELS Actions — Market, War & CHF Monitor")
    st.caption("Briefing financier et veille des risques patrimoniaux · actualisation automatique toutes les 10 minutes · actualités : cache 10 min · cours : cache 5 min")
    st.caption("Dernier recalcul affiché : " + datetime.now().astimezone().strftime("%d/%m/%Y à %H:%M:%S %Z"))
    with st.spinner("Collecte des flux d'actualité…"):
        df, errors = fetch_news()
    df = enrich_news_frame(df) if not df.empty else df
    with st.sidebar:
        st.title("WELS Actions")
        page = st.radio("Navigation", [
            "Vue d'ensemble & Géopolitique", "Newsletter stratégique — IA",
            "Follow the Money — Investissements", "Marchés & portefeuille", "CHF & devises"
        ], key="wels_navigation_v4")
        # Revenir systématiquement aux widgets régionaux lorsqu'on ouvre la page,
        # sans effacer la sélection lors des reruns provoqués par un clic régional.
        previous_page = st.session_state.get("_wels_previous_navigation_page")
        if previous_page != page:
            st.session_state["wels_selected_risk_region"] = None
            st.session_state["_wels_previous_navigation_page"] = page
        if st.button("Actualiser toutes les données", key="wels_refresh_all"):
            fetch_news.clear()
            fetch_history.clear()
            fetch_article_text.clear()
            # Force a fresh AI synthesis even when the incoming RSS corpus is unchanged.
            st.session_state["wels_ai_refresh_id"] = str(time.time_ns())
            st.rerun()
        st.caption("Actualités : cache 10 min · cours : cache 5 min")
    if page == "Vue d'ensemble & Géopolitique":
        # Keep the original overview first, then show the full geopolitics/supply monitor.
        executive_dashboard(df)
        st.divider()
        geopolitics_supply_dashboard(df, errors)
    elif page == "Newsletter stratégique — IA": ai_newsletter_page(df, errors)
    elif page == "Follow the Money — Investissements": investment_flows_page()
    elif page == "Marchés & portefeuille": market_watch_page()
    elif page == "CHF & devises": fx_chf_page()

def newsletter_dashboard():
    """Point d’entrée compatible avec app.py sur Streamlit Cloud."""
    main()


if __name__ == "__main__":
    main()
