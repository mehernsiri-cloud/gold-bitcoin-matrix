# Dynamic Follow the Money — fundraising radar
from datetime import datetime
from urllib.parse import quote
import re
import html
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

try:
    import feedparser
except ImportError:
    feedparser = None

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CapitalFlowRadar/1.0)"}
FUNDING_QUERIES = {
    "IA / data centers / semi-conducteurs": '("raised" OR "closes" OR "secured") (funding OR round OR investment) (AI OR artificial intelligence OR datacenter OR semiconductor)',
    "Santé / biotech / pharmacie": '("raised" OR "closes" OR "secured") (funding OR Series OR round) (biotech OR healthcare OR pharma OR medtech)',
    "Défense / sécurité": '("raised" OR "closes" OR "secured") (funding OR round) (defense OR defence OR security OR aerospace)',
    "Énergie / climat / réseaux": '("raised" OR "closes" OR "secured") (funding OR round) (energy OR battery OR grid OR climate OR cleantech)',
    "Industrie / mobilité / robotique": '("raised" OR "closes" OR "secured") (funding OR round) (manufacturing OR robotics OR mobility OR industrial)',
    "Agroalimentaire / foodtech": '("raised" OR "closes" OR "secured") (funding OR round) (agriculture OR agritech OR foodtech OR food)',
    "Fintech / logiciels / autres": '("raised" OR "closes" OR "secured") (funding OR Series OR round) (fintech OR software OR startup)',
}
COUNTRY_RULES = [
    ("États-Unis", ["united states", " u.s.", "us-based", "san francisco", "new york", "boston", "silicon valley", "california"]),
    ("Royaume-Uni", ["united kingdom", " uk-based", "london", "cambridge, uk", "british"]),
    ("France", ["france", "paris", "lyon", "french"]),
    ("Allemagne", ["germany", "berlin", "munich", "german"]),
    ("Suisse", ["switzerland", " zurich", "geneva", "lausanne", "swiss"]),
    ("Israël", ["israel", "tel aviv", "israeli"]),
    ("Inde", ["india", "bengaluru", "bangalore", "mumbai", "delhi", "indian"]),
    ("Chine", ["china", "beijing", "shenzhen", "shanghai", "chinese"]),
    ("Singapour", ["singapore"]),
    ("Canada", ["canada", "toronto", "montreal", "vancouver"]),
    ("Pays-Bas", ["netherlands", "amsterdam", "dutch"]),
    ("Suède", ["sweden", "stockholm", "swedish"]),
    ("Espagne", ["spain", "madrid", "barcelona", "spanish"]),
    ("Italie", ["italy", "milan", "italian"]),
    ("Corée du Sud", ["south korea", "seoul", "korean"]),
    ("Japon", ["japan", "tokyo", "japanese"]),
    ("Australie", ["australia", "sydney", "melbourne"]),
    ("Émirats arabes unis", ["uae", "dubai", "abu dhabi", "emirates"]),
    ("Brésil", ["brazil", "sao paulo", "brazilian"]),
]

def _clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))).strip()

def _amount(text):
    patterns = [
        r"(?P<cur>US\$|\$|USD|EUR|€|GBP|£|CHF|INR|C\$|A\$)\s*(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>bn|billion|m|mn|million|b|k)?",
        r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>bn|billion|m|mn|million)\s*(?P<cur>USD|EUR|GBP|CHF|INR|dollars|euros|pounds)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            cur = m.group("cur").upper()
            num = float(m.group("num").replace(",", "."))
            unit = (m.groupdict().get("unit") or "").lower()
            mult = 1_000_000_000 if unit in ("bn", "billion", "b") else 1_000_000 if unit in ("m", "mn", "million") else 1_000
            symbol = {"US$":"USD", "$":"USD", "€":"EUR", "£":"GBP", "C$":"CAD", "A$":"AUD"}.get(cur, cur)
            return num * mult, symbol, m.group(0)
    return None, None, None

def _country(text):
    t = text.lower()
    for country, terms in COUNTRY_RULES:
        if any(term in t for term in terms):
            return country
    return "À vérifier"

def _company(title):
    patterns = [r"^(.{2,90}?)\s+(?:raises?|raised|secures?|secured|closes?|closed|lands?)\b",
                r"^(.{2,90}?)\s+(?:announces?|announced)\s+(?:a\s+)?(?:series|funding|investment)"]
    for p in patterns:
        m = re.search(p, title, re.I)
        if m:
            return m.group(1).strip(" —:-")
    return title[:90]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundraising_radar(days=45):
    if feedparser is None:
        return pd.DataFrame(), ["Le module feedparser manque dans requirements.txt."]
    rows, errors = [], []
    for sector, query in FUNDING_QUERIES.items():
        url = "https://news.google.com/rss/search?q=" + quote(query + f" when:{days}d") + "&hl=en-US&gl=US&ceid=US:en"
        try:
            response = requests.get(url, headers=HEADERS, timeout=(4, 12))
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            for entry in feed.entries:
                title = _clean(entry.get("title"))
                summary = _clean(entry.get("summary"))
                link = entry.get("link", "")
                blob = f"{title}. {summary}"
                if not re.search(r"\b(raised|raises|closes|closed|secured|has raised|have raised)\b", blob, re.I):
                    continue
                amount, currency, amount_text = _amount(blob)
                if not amount or not link:
                    continue
                published = entry.get("published", entry.get("updated", ""))
                try:
                    date = pd.to_datetime(published, utc=True).tz_convert(None)
                except Exception:
                    date = pd.NaT
                source = _clean(entry.get("source", {}).get("title", "")) or "Source à ouvrir"
                rows.append({
                    "Date": date, "Entreprise": _company(title), "Pays": _country(f"{title} {summary}"),
                    "Secteur": sector, "Montant": amount, "Devise": currency,
                    "Montant affiché": amount_text, "Titre": title, "Source": source,
                    "Lien": link, "Résumé": summary[:500],
                })
        except Exception as exc:
            errors.append(f"{sector} : {type(exc).__name__}")
    if not rows:
        return pd.DataFrame(), errors
    df = pd.DataFrame(rows).drop_duplicates(subset=["Lien"])
    return df.sort_values("Date", ascending=False, na_position="last"), errors

def render_financial_intelligence():
    st.markdown("""
    <style>
    .fm-hero{background:linear-gradient(120deg,#101E30,#183B52 58%,#27645F);padding:25px 28px;border-radius:16px;color:#fff;margin:4px 0 18px}
    .fm-hero h2{color:white;margin:8px 0;font-size:29px}
    .fm-hero p{color:#dce8ef;margin:0;font-size:14px}
    .fm-kpi{background:#14283B;border:1px solid #2D465D;border-radius:12px;padding:14px;color:white;min-height:105px}
    .fm-kpi small{color:#b7c8d6;text-transform:uppercase;font-weight:800;letter-spacing:1px}
    .fm-kpi strong{display:block;font-size:27px;margin:5px 0}
    </style>
    <div class="fm-hero"><div style="font-size:11px;letter-spacing:2px;color:#B7D8D1;font-weight:800">FOLLOW THE MONEY · LIVE DEAL FLOW</div>
    <h2>Les capitaux qui bougent réellement</h2>
    <p>Suivi glissant des levées de fonds rapportées : entreprise, pays, secteur, montant, date et article source. Pas de projection annuelle présentée comme un flux en temps réel.</p></div>
    """, unsafe_allow_html=True)
    c1,c2=st.columns([1,5])
    with c1:
        days=st.selectbox("Période", [7,14,30,45,90], index=3, format_func=lambda x:f"{x} derniers jours", key="fm_days")
    with c2:
        st.caption("Actualisation automatique des résultats toutes les 30 minutes · utilisez « Actualiser » pour forcer une nouvelle collecte.")
    if st.button("↻ Actualiser les levées", key="fm_refresh"):
        fetch_fundraising_radar.clear()
        st.rerun()
    with st.spinner("Recherche des levées de fonds dans les flux d’actualité…"):
        df, errors = fetch_fundraising_radar(days)
    if df.empty:
        st.warning("Aucune levée répondant aux critères stricts (montant explicite + levée réalisée/sécurisée) n’a été détectée dans les flux interrogés. Cela ne signifie pas qu’il n’y a eu aucune opération.")
        if errors:
            st.caption("Flux temporairement indisponibles : " + " · ".join(errors[:5]))
        st.markdown("[Ouvrir Google News — levées de fonds récentes](https://news.google.com/search?q=startup%20raised%20funding%20round)")
        return
    countries=sorted(df["Pays"].unique())
    sectors=sorted(df["Secteur"].unique())
    f1,f2=st.columns(2)
    with f1: selected_countries=st.multiselect("Pays",countries,default=countries,key="fm_countries")
    with f2: selected_sectors=st.multiselect("Secteurs",sectors,default=sectors,key="fm_sectors")
    view=df[df["Pays"].isin(selected_countries)&df["Secteur"].isin(selected_sectors)].copy()
    usd=view[view["Devise"]=="USD"]
    total_usd=usd["Montant"].sum()/1e9
    a,b,c,d=st.columns(4)
    metrics=[("OPÉRATIONS REPÉRÉES",str(len(view))),("PAYS IDENTIFIÉS",str(view[view["Pays"]!="À vérifier"]["Pays"].nunique())),("SECTEURS ACTIFS",str(view["Secteur"].nunique())),("MONTANTS USD",f"$ {total_usd:.2f} Md" if total_usd else "—")]
    for col,(label,value) in zip([a,b,c,d],metrics):
        with col: st.markdown(f'<div class="fm-kpi"><small>{label}</small><strong>{value}</strong><span style="font-size:11px;color:#c5d2dd">Période sélectionnée</span></div>',unsafe_allow_html=True)
    st.markdown("### Où les levées sont-elles réalisées ?")
    left,right=st.columns(2)
    with left:
        country_counts=view[view["Pays"]!="À vérifier"].groupby("Pays").size().sort_values(ascending=True).tail(15).reset_index(name="Opérations")
        if not country_counts.empty:
            fig=px.bar(country_counts,x="Opérations",y="Pays",orientation="h",text="Opérations",title="Nombre de levées par pays")
            fig.update_layout(height=390,margin=dict(l=8,r=8,t=45,b=8),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
    with right:
        sector_counts=view.groupby("Secteur").size().sort_values(ascending=True).reset_index(name="Opérations")
        fig=px.bar(sector_counts,x="Opérations",y="Secteur",orientation="h",text="Opérations",title="Nombre de levées par secteur")
        fig.update_layout(height=390,margin=dict(l=8,r=8,t=45,b=8),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Cartographie pays × secteurs")
    matrix=view[view["Pays"]!="À vérifier"].pivot_table(index="Pays",columns="Secteur",values="Entreprise",aggfunc="count",fill_value=0)
    if not matrix.empty:
        heat=px.imshow(matrix,text_auto=True,aspect="auto",color_continuous_scale="Teal",labels=dict(x="Secteur",y="Pays",color="Levées"))
        heat.update_layout(height=max(330,min(700,34*len(matrix)+150)),margin=dict(l=8,r=8,t=25,b=8))
        st.plotly_chart(heat,use_container_width=True)
    st.markdown("### Dernières levées de fonds détectées")
    st.caption("Montants dans leur devise d’origine : aucun taux de change approximatif n’est appliqué.")
    for _,row in view.head(30).iterrows():
        date=row["Date"].strftime("%d/%m/%Y") if pd.notna(row["Date"]) else "Date non précisée"
        amount=f"{row['Montant affiché']} ({row['Devise']})"
        st.markdown(f"**{row['Entreprise']}** · {row['Pays']} · {row['Secteur']}  \n<small>{date} · {amount} · {row['Source']}</small>  \n[{row['Titre']}]({row['Lien']})",unsafe_allow_html=True)
        st.divider()
    st.markdown("### Qualité et limites des données")
    st.info("Radar alimenté par Google News RSS. Le filtre exige un montant explicite et un vocabulaire de levée réalisée/sécurisée ; le pays et le secteur sont classés automatiquement à partir du titre/résumé. Chaque ligne doit être vérifiée dans son article source avant d’être considérée comme définitivement confirmée. Une absence de résultat n’est pas une preuve d’absence de levée.")
    if errors:
        st.caption("Certains flux de recherche ont échoué : " + " · ".join(errors[:7]))
