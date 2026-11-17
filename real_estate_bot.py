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

MIN_PRICE_DEFAULT_TND = 300_000
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
    "Hammam Ghezèze": ["hammam ghezaz", "hammam ghezèze"],
    "Béni Khalled": ["beni khalled", "béni khalled"],
}
TAYARA_URL = "https://www.tayara.tn/listing/c/immobilier/maisons-et-villas/l/nabeul/?page=1"
SOURCE_URLS = {
    "Tayara": TAYARA_URL,
    "TounesImmo – maisons": "https://www.tounesimmo.com/maison-a-vendre-nabeul",
    "Limmobilier.tn – villas": "https://limmobilier.tn/vente/villa/nabeul",
    "Limmobilier.tn – maisons": "https://limmobilier.tn/vente/maison/nabeul",
    "Menzili": "https://www.menzili.tn/vente/nabeul",
    "Mubawab": "https://www.mubawab.tn/fr/ct/nabeul/immobilier-a-vendre",
    "Cava.tn": "https://www.cava.tn/category/immobilier/maisons_et_villas/a/Nabeul",
    "Affare.tn": "https://www.affare.tn/petites-annonces/nabeul/immobilier",
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
    property_terms = ("maison", "villa", "duplex", "maisonnette", "appartement", "terrain", "immeuble", "دار", "فيلا", "أرض")
    excluded = ("a louer", "location", "loyer", "rent", "studio a louer", "à louer")
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
    found, seen = [], set()
    sale_context = any(x in base_url.lower() for x in ("/vente", "a-vendre", "immobilier-a-vendre"))
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        if href.rstrip("/") == base_url.rstrip("/") or href in seen:
            continue
        if source == "Tayara" and "/item/maisons-et-villas/" not in href:
            continue
        if source != "Tayara" and not any(x in href.lower() for x in ("/annonce", "/bien", "/vente/", "/maison", "/villa", "/property", "/listing", "/immobilier")):
            continue
        seen.add(href)
        card = _candidate_container(anchor)
        title = anchor.get_text(" ", strip=True)
        headings = [x.get_text(" ", strip=True) for x in card.find_all(["h1", "h2", "h3", "h4"])]
        if len(title) < 8:
            title = next((x for x in headings if len(x) > 8), title)
        card_text = card.get_text(" ", strip=True)
        combined = f"{title} {card_text}"
        if not title or not _is_eligible(combined, sale_context=sale_context):
            continue
        price = _price_tnd(combined)
        town = _town_from_text(combined)
        if price is None or town is None:
            continue
        image_url = ""
        image = card.find("img")
        if image:
            for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-original-src"):
                candidate = image.get(attr)
                if candidate and not candidate.startswith("data:"):
                    image_url = candidate
                    break
            if not image_url and image.get("srcset"):
                image_url = image["srcset"].split(",")[0].strip().split(" ")[0]
        if not image_url:
            og = card.find("meta", property="og:image") or soup.find("meta", property="og:image")
            if og:
                image_url = og.get("content", "")
        image_url = urljoin(base_url, image_url) if image_url else ""
        if not image_url or image_url.startswith("data:"):
            continue
        found.append({
            "title": title[:180], "town": town, "price": price,
            "url": href, "source": source, "image": image_url,
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
            source_label = source.split(" – ")[0]
            cards = _extract_cards(response.text, url, source_label)
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
        ("Tayara – Immobilier Nabeul", SOURCE_URLS["Tayara"]),
        ("Menzili – Ventes à Nabeul", SOURCE_URLS["Menzili"]),
        ("Mubawab – Immobilier à vendre", SOURCE_URLS["Mubawab"]),
        ("Cava.tn – Maisons et villas", SOURCE_URLS["Cava.tn"]),
        ("Affare.tn – Immobilier Nabeul", SOURCE_URLS["Affare.tn"]),
        ("TounesImmo – Maisons à vendre", SOURCE_URLS["TounesImmo – maisons"]),
        ("Limmobilier.tn – Villas à vendre", SOURCE_URLS["Limmobilier.tn – villas"]),
        ("Limmobilier.tn – Maisons à vendre", SOURCE_URLS["Limmobilier.tn – maisons"]),
    ]
def _render_listing(item):
    price = f'{item["price"]:,}'.replace(",", " ")
    title = item["title"].replace("<", "&lt;").replace(">", "&gt;")
    st.image(item["image"], use_column_width=True)
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
        "Agrégation multi-sites des biens **à vendre** dans le gouvernorat de Nabeul : "
        "maisons, villas, appartements, terrains et immeubles. "
        "Le filtre démarre à **300 000 TND**, sans plafond de prix."
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Prix minimum affiché", "300 000 TND")
    metric_cols[1].metric("Types de biens", "Tous les biens à vendre")
    metric_cols[2].metric("Zone", "Gouvernorat de Nabeul")

    with st.sidebar:
        st.markdown("### Filtres immobilier")
        selected_towns = st.multiselect(
            "Villes / zones",
            list(COASTAL_TOWNS.keys()),
            default=list(COASTAL_TOWNS.keys()),
            key="nabeul_towns",
        )
        min_price = st.number_input(
            "Prix minimum (TND)",
            min_value=0, max_value=10_000_000,
            value=MIN_PRICE_DEFAULT_TND, step=25_000,
            key="nabeul_min_price",
        )
        refresh = st.button("🔄 Actualiser les annonces", use_container_width=True)
    if refresh:
        fetch_listings.clear()

    with st.spinner("Recherche multi-sources : Tayara, Menzili, Mubawab, Cava, Affare et autres portails…"):
        listings, statuses = fetch_listings()

    filtered = [
        item for item in listings
        if item["town"] in selected_towns and item["price"] >= min_price
    ]
    st.subheader(f"{len(filtered)} annonce(s) avec photo · prix ≥ {min_price:,.0f} TND".replace(",", " "))
    if not filtered:
        st.warning(
            "Aucune annonce avec photo n’a été récupérée pour ces critères. "
            "Cela ne signifie pas qu’il n’existe aucun bien : certains portails bloquent "
            "l’extraction automatique ou chargent les annonces par JavaScript."
        )
    else:
        for start in range(0, len(filtered), 3):
            cols = st.columns(3, gap="medium")
            for col, item in zip(cols, filtered[start:start + 3]):
                with col:
                    _render_listing(item)

    with st.expander("État des sources automatiques", expanded=True):
        for source, status in statuses.items():
            st.write(f"**{source}** : {status}")
        st.caption(
            "Une annonce n'est affichée que si son prix, sa localité et au moins une photo sont détectés. "
            "Les portails peuvent limiter l'accès automatique ; les liens ci-dessous permettent de consulter "
            "leurs résultats directement. Vérifie toujours la disponibilité, le titre foncier et les autorisations."
        )

    with st.expander("Tous les portails immobiliers suivis"):
        for label, url in _search_links():
            st.markdown(f"- [{label}]({url})")
