"""Real-estate listings for coastal towns in the Nabeul governorate, Tunisia.

The dashboard aggregates publicly visible classified listings where possible.
Listings are indicative; users must verify availability, ownership and legal status
with the advertiser/notary before any commitment.
"""
import re
import unicodedata
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

MAX_PRICE_TND = 500_000
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NabeulPropertyWatch/1.0)",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.7",
}
COASTAL_TOWNS = {
    "Nabeul": ["nabeul"],
    "Hammamet": ["hammamet", "yasmine hammamet", "hammamet sud", "hammamet nord"],
    "Kélibia": ["kelibia", "kélibia", "mansourah", "dar allouche", "kerkouane"],
    "Korba": ["korba"],
    "Menzel Temime": ["menzel temime", "menzel tmim"],
    "Béni Khiar": ["beni khiar", "béni khiar"],
    "Dar Chaâbane": ["dar chaabane", "dar chaâbane"],
    "El Haouaria": ["el haouaria", "haouaria"],
    "Soliman": ["soliman"],
    "Mrezga": ["mrezga"],
    "Maâmoura": ["maamoura", "maâmoura"],
    "Tazarka": ["tazarka"],
}
TAYARA_URL = "https://www.tayara.tn/listing/c/immobilier/maisons-et-villas/l/nabeul/?page=1"
SOURCE_URLS = {
    "Tayara": TAYARA_URL,
    "TounesImmo – maisons": "https://www.tounesimmo.com/maison-a-vendre-nabeul",
    "Limmobilier.tn – villas": "https://limmobilier.tn/vente/villa/nabeul",
    "Limmobilier.tn – maisons": "https://limmobilier.tn/vente/maison/nabeul",
}

def _norm(value):
    value = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))

def _price_tnd(text):
    """Extract a price in TND; MD is commonly used as thousand dinars."""
    value = _norm(text).replace("\u202f", " ").replace("\xa0", " ")
    patterns = [
        r"(\d[\d\s.,]*\d|\d)\s*(?:dt|tnd)\b",
        r"(\d+(?:[.,]\d+)?)\s*md\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue
        raw = match.group(1).replace(" ", "").replace(",", ".")
        try:
            number = float(raw)
            if "md" in match.group(0):
                number *= 1000
            else:
                number = int(number)
            if number >= 10_000:
                return int(number)
        except ValueError:
            pass
    return None

def _town_from_text(text):
    value = _norm(text)
    for town, aliases in COASTAL_TOWNS.items():
        if any(_norm(alias) in value for alias in aliases):
            return town
    return None

def _is_eligible(text, sale_context=False):
    value = _norm(text)
    sale_terms = ("a vendre", "à vendre", "vente", "for sale", "للبيع", "av ")
    property_terms = ("maison", "villa", "duplex", "maisonnette", "دار", "فيلا")
    excluded = ("a louer", "location", "loyer", "rent", "studio a louer", "appartement")
    return (
        (sale_context or any(term in value for term in sale_terms))
        and any(term in value for term in property_terms)
        and not any(term in value for term in excluded)
    )

def _candidate_container(anchor):
    node = anchor
    for _ in range(5):
        if node is None or getattr(node, "parent", None) is None:
            break
        text = node.get_text(" ", strip=True)
        if 45 <= len(text) <= 1400:
            return node
        node = node.parent
    return anchor.parent or anchor

def _extract_cards(html_text, base_url, source):
    soup = BeautifulSoup(html_text, "html.parser")
    found = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        if source == "Tayara" and "/item/maisons-et-villas/" not in href:
            continue
        if source != "Tayara" and not any(x in href for x in ("/annonce", "/bien", "/vente/", "/maison", "/villa")):
            continue
        if href in seen:
            continue
        seen.add(href)
        card = _candidate_container(anchor)
        title = anchor.get_text(" ", strip=True)
        card_text = card.get_text(" ", strip=True)
        combined = f"{title} {card_text}"
        sale_context = "/vente/" in base_url or "a-vendre" in base_url
        if not title or not _is_eligible(combined, sale_context=sale_context):
            continue
        price = _price_tnd(combined)
        town = _town_from_text(combined)
        if price is None or price > MAX_PRICE_TND or town is None:
            continue
        if len(title) < 8:
            title = next((x.get_text(" ", strip=True) for x in card.find_all(["h2", "h3", "h4"]) if len(x.get_text(" ", strip=True)) > 8), title)
        image = card.find("img")
        image_url = ""
        if image:
            image_url = image.get("src") or image.get("data-src") or ""
            image_url = urljoin(base_url, image_url)
        found.append({
            "title": title[:180],
            "town": town,
            "price": price,
            "url": href,
            "source": source,
            "image": image_url,
            "details": card_text[:500],
        })
    return found

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_listings():
    results = []
    statuses = {}
    for source, url in SOURCE_URLS.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=(5, 18))
            response.raise_for_status()
            cards = _extract_cards(response.text, url, "Tayara" if source == "Tayara" else "TounesImmo" if source.startswith("TounesImmo") else "Limmobilier.tn")
            results.extend(cards)
            statuses[source] = f"OK · {len(cards)} annonce(s) retenue(s)"
        except requests.RequestException as exc:
            statuses[source] = f"Indisponible · {type(exc).__name__}"
        except Exception as exc:
            statuses[source] = f"Erreur de lecture · {type(exc).__name__}"

    unique = {}
    for item in results:
        key = item["url"].split("?", 1)[0].rstrip("/")
        if key not in unique:
            unique[key] = item
    listings = sorted(unique.values(), key=lambda item: item["price"])
    return listings, statuses

def _search_links():
    return [
        ("Tayara – Maisons et villas à Nabeul", TAYARA_URL),
        ("TounesImmo – Maisons à vendre", SOURCE_URLS["TounesImmo – maisons"]),
        ("Limmobilier.tn – Villas à vendre", SOURCE_URLS["Limmobilier.tn – villas"]),
        ("Limmobilier.tn – Maisons à vendre", SOURCE_URLS["Limmobilier.tn – maisons"]),
        ("Cava.tn – Maisons et villas à Hammamet", "https://www.cava.tn/category/immobilier/maisons_et_villas/a/Nabeul/Hammamet"),
    ]

def _render_listing(item):
    price = f'{item["price"]:,}'.replace(",", " ")
    title = item["title"].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
        <div style="border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:13px;margin-bottom:10px;min-height:155px">
          <div style="font-size:1.08rem;font-weight:750;line-height:1.35">{title}</div>
          <div style="font-size:1.35rem;font-weight:850;margin:9px 0;color:#16804a">{price} TND</div>
          <div style="font-size:.92rem;opacity:.85">{item["town"]} · {item["source"]}</div>
          <div style="margin-top:10px"><a href="{item["url"]}" target="_blank" rel="noopener noreferrer">Consulter l’annonce ↗</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def real_estate_dashboard():
    st.title("🏡 Immobilier – Cap Bon, Tunisie")
    st.markdown(
        "Maisons et villas **à vendre** dans les villes côtières du gouvernorat de Nabeul. "
        "Budget maximal : **500 000 TND**."
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Budget maximum", "500 000 TND")
    metric_cols[1].metric("Type de bien", "Maison / Villa")
    metric_cols[2].metric("Zone", "Littoral de Nabeul")

    with st.sidebar:
        st.markdown("### Filtres immobilier")
        selected_towns = st.multiselect(
            "Villes côtières",
            list(COASTAL_TOWNS.keys()),
            default=list(COASTAL_TOWNS.keys()),
            key="nabeul_towns",
        )
        max_price = st.slider(
            "Budget maximum (TND)",
            min_value=50_000,
            max_value=MAX_PRICE_TND,
            value=MAX_PRICE_TND,
            step=10_000,
            key="nabeul_max_price",
        )
        refresh = st.button("🔄 Actualiser les annonces", use_container_width=True)
    if refresh:
        fetch_listings.clear()

    with st.spinner("Récupération des annonces immobilières publiques…"):
        listings, statuses = fetch_listings()

    filtered = [
        item for item in listings
        if item["town"] in selected_towns and item["price"] <= max_price
    ]
    st.subheader(f"{len(filtered)} annonce(s) correspondant aux critères")
    if not filtered:
        st.warning(
            "Aucune annonce n’a pu être récupérée avec les filtres actuels. "
            "Cela ne signifie pas qu’il n’existe aucun bien : certains sites bloquent "
            "l’agrégation automatique ou modifient leur structure."
        )
    else:
        for start in range(0, len(filtered), 3):
            cols = st.columns(3, gap="medium")
            for col, item in zip(cols, filtered[start:start + 3]):
                with col:
                    _render_listing(item)

    with st.expander("État des sources automatiques"):
        for source, status in statuses.items():
            st.write(f"**{source}** : {status}")
        st.caption("Les annonces sont filtrées sur le prix affiché et la localité détectée. Vérifie toujours le prix, la disponibilité, le titre foncier et les autorisations auprès du vendeur ou d’un notaire.")

    with st.expander("Autres sites à consulter directement"):
        for label, url in _search_links():
            st.markdown(f"- [{label}]({url})")
        st.caption("Les liens externes permettent de poursuivre la recherche si un site ne permet pas la lecture automatique de ses annonces.")
