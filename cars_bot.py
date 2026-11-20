"""Used-car watchlist from La Centrale, focused on Toyota RAV4."""
import re
from datetime import datetime
from urllib.parse import urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup

BASE = "https://www.lacentrale.fr"
SEARCHES = {
    "RAV4 hybride": f"{BASE}/voiture-occasion/hybride/modele-toyota-rav%2B4.html",
    "RAV4 hybride rechargeable": f"{BASE}/voiture-occasion/hybride_rechargeable/modele-toyota-rav%2B4.html",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.7",
}
MAX_PRICE_DEFAULT = 50000
MIN_YEAR_DEFAULT = 2024
MAX_KM_DEFAULT = 50000
POSTAL_CODE = "78140"
SEARCH_RADIUS_KM = 100


def _number(pattern, text, cast=int):
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    try:
        return cast(re.sub(r"[^0-9]", "", match.group(1)))
    except (TypeError, ValueError):
        return None


def _card_for(anchor):
    node, best = anchor, anchor.parent or anchor
    for _ in range(8):
        if node is None:
            break
        text = node.get_text(" ", strip=True)
        if 35 <= len(text) <= 2400:
            best = node
            if re.search(r"\d[\d\s\u202f]*\s*€", text) and re.search(r"\d[\d\s]*\s*km", text, re.I):
                return node
        node = node.parent
    return best


def _extract(html, source_label):
    soup = BeautifulSoup(html, "html.parser")
    rows, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE, a["href"])
        if "auto-occasion-annonce-" not in href or href in seen:
            continue
        seen.add(href)
        card = _card_for(a)
        title = a.get_text(" ", strip=True)
        if len(title) < 8:
            title = next((h.get_text(" ", strip=True) for h in card.find_all(["h2", "h3", "h4"]) if len(h.get_text(" ", strip=True)) > 8), "")
        text = card.get_text(" ", strip=True)
        combined = (title + " " + text).strip()
        if not re.search(r"RAV\s*4|RAV4", combined, re.I):
            continue
        price = _number(r"([\d\s\u202f]+)\s*€", combined)
        mileage = _number(r"([\d\s\u202f]+)\s*km", combined)
        year = _number(r"\b(20[0-2]\d|19\d\d)\b", combined)
        if not price or not mileage:
            continue
        image = ""
        img = card.find("img")
        if img:
            for attr in ("src", "data-src", "data-lazy-src", "data-original", "srcset"):
                val = img.get(attr)
                if val and not val.startswith("data:"):
                    image = val.split(",")[0].strip().split(" ")[0]
                    break
        image = urljoin(BASE, image) if image else ""
        lower = combined.lower()
        drivetrain = "4x4 / AWD-i" if re.search(r"4x4|awd.?i|4 roues motrices", lower) else "Non précisé / 2 roues motrices"
        energy = "Hybride rechargeable" if ("rechargeable" in lower or "phev" in lower or "306" in lower) else "Hybride"
        rows.append({
            "title": title[:180] or combined[:150],
            "price": price, "mileage": mileage, "year": year,
            "energy": energy, "drivetrain": drivetrain,
            "url": href, "image": image, "source": source_label,
            "details": text[:450],
        })
    return rows


def _extract_reader_markdown(markdown, source_label):
    """Fallback parser for a text-rendered copy of the public search page."""
    rows, seen = [], set()
    for line in markdown.splitlines():
        links = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", line)
        for title, href in links:
            if "auto-occasion-annonce-" not in href or href in seen:
                continue
            combined = re.sub(r"\s+", " ", title + " " + line).strip()
            if not re.search(r"RAV\s*4|RAV4", combined, re.I):
                continue
            price = _number(r"([\d\s\u202f]+)\s*€", combined)
            mileage = _number(r"([\d\s\u202f]+)\s*km", combined)
            year = _number(r"\b(20[0-2]\d|19\d\d)\b", combined)
            if not price or not mileage:
                continue
            seen.add(href)
            lower = combined.lower()
            rows.append({
                "title": title[:180], "price": price, "mileage": mileage,
                "year": year,
                "energy": "Hybride rechargeable" if ("rechargeable" in lower or "phev" in lower or "306" in lower) else "Hybride",
                "drivetrain": "4x4 / AWD-i" if re.search(r"4x4|awd.?i|4 roues motrices", lower) else "Non précisé / 2 roues motrices",
                "url": href, "image": "", "source": source_label + " (lecture texte)",
                "details": combined[:450],
            })
    return rows


@st.cache_data(ttl=900, show_spinner=False)
def fetch_cars():
    cars, statuses = [], {}
    for label, url in SEARCHES.items():
        direct_error = None
        try:
            response = requests.get(url, headers=HEADERS, timeout=(6, 20))
            response.raise_for_status()
            extracted = _extract(response.text, label)
            cars.extend(extracted)
            if extracted:
                statuses[label] = f"OK · {len(extracted)} annonce(s) extraite(s) via HTML"
                continue
            statuses[label] = f"HTML reçu (HTTP {response.status_code}) mais aucune carte reconnue"
        except requests.RequestException as exc:
            direct_error = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            statuses[label] = f"Accès direct échoué (HTTP {code if code else 'réseau'})"

        # Fallback: fetch a text-rendered public page; no credentials or session cookies.
        try:
            reader_url = "https://r.jina.ai/" + url
            reader = requests.get(reader_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"}, timeout=(8, 30))
            reader.raise_for_status()
            extracted = _extract_reader_markdown(reader.text, label)
            cars.extend(extracted)
            statuses[label] = (
                f"Repli texte · {len(extracted)} annonce(s) extraite(s)"
                if extracted else
                f"Repli texte accessible (HTTP {reader.status_code}) · aucune annonce reconnue"
            )
        except requests.RequestException as fallback_exc:
            code = getattr(getattr(fallback_exc, "response", None), "status_code", None)
            previous = statuses.get(label, "Accès direct échoué")
            statuses[label] = f"{previous} · repli texte indisponible (HTTP {code if code else 'réseau'})"

    unique = {}
    for car in cars:
        unique.setdefault(car["url"], car)
    return list(unique.values()), statuses


def _render_car(car):
    from html import escape
    title, url = escape(car["title"]), escape(car["url"], quote=True)
    image = escape(car["image"], quote=True)
    price = f'{car["price"]:,}'.replace(",", " ")
    mileage = f'{car["mileage"]:,}'.replace(",", " ")
    year = car["year"] or "Année à vérifier"
    photo = (f'<img src="{image}" style="width:100%;height:185px;object-fit:cover">' if image
             else '<div style="height:185px;display:flex;align-items:center;justify-content:center;opacity:.6">Photo non disponible</div>')
    st.markdown(f"""
    <div style="height:410px;display:flex;flex-direction:column;overflow:hidden;border:1px solid rgba(128,128,128,.35);border-radius:14px;margin-bottom:16px;background:var(--secondary-background-color)">
      <div style="height:185px;min-height:185px;overflow:hidden;background:rgba(128,128,128,.1)">{photo}</div>
      <div style="padding:13px;display:flex;flex:1;flex-direction:column">
        <div style="font-weight:700;line-height:1.35;height:3.9em;overflow:hidden">{title}</div>
        <div style="font-size:1.35rem;font-weight:850;color:#16804a;margin:7px 0">{price} €</div>
        <div style="font-size:.9rem;opacity:.82">{year} · {mileage} km</div>
        <div style="font-size:.88rem;opacity:.82">{escape(car["energy"])} · {escape(car["drivetrain"])}</div>
        <div style="margin-top:auto;padding-top:10px"><a href="{url}" target="_blank" rel="noopener noreferrer" style="display:block;text-align:center;padding:9px;border:1px solid rgba(128,128,128,.45);border-radius:8px;text-decoration:none;font-weight:650">Consulter l’annonce ↗</a></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def cars_dashboard():
    st.title("🚗 Cars – La Centrale")
    st.markdown("Suivi des annonces d’occasion **Toyota RAV4**, avec priorité aux versions hybrides et AWD-i. Recherche **RAV4 hybride rechargeable**, jusqu’à **50 000 €**, à partir de **2024**, **50 000 km maximum**, code postal **78140** (rayon **100 km**).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Budget maximum", "50 000 €")
    c2.metric("Recherche", "RAV4 PHEV · 2024+")
    c3.metric("Zone", "78140 · 100 km")

    with st.sidebar:
        st.markdown("### Filtres Cars")
        max_price = st.number_input("Budget maximum (€)", min_value=10000, max_value=100000, value=MAX_PRICE_DEFAULT, step=1000, key="cars_max_price_v2")
        max_km = st.number_input("Kilométrage maximum", min_value=10000, max_value=300000, value=MAX_KM_DEFAULT, step=5000, key="cars_max_km_v2")
        min_year = st.number_input("Année minimum", min_value=2005, max_value=datetime.now().year, value=MIN_YEAR_DEFAULT, step=1, key="cars_min_year_v2")
        only_awd = st.checkbox("Prioriser / filtrer les 4x4 AWD-i", value=False, key="cars_only_awd")
        refresh = st.button("🔄 Actualiser les annonces", use_container_width=True, key="cars_refresh")
    if refresh:
        fetch_cars.clear()

    with st.spinner("Récupération des annonces Toyota RAV4 sur La Centrale…"):
        listings, statuses = fetch_cars()
    filtered = [x for x in listings if x["price"] <= max_price and x["mileage"] <= max_km and x["year"] is not None and x["year"] >= min_year and x["energy"] == "Hybride rechargeable"]
    if only_awd:
        filtered = [x for x in filtered if "4x4 / AWD-i" in x["drivetrain"]]
    # Keep RAV4 listings under budget; order by AWD-i relevance then year/mileage.
    filtered.sort(key=lambda x: ("4x4 / AWD-i" not in x["drivetrain"], -(x["year"] or 0), x["price"], x["mileage"]))
    st.subheader(f"{len(filtered)} annonce(s) correspondant aux filtres · {sum(bool(x['image']) for x in filtered)} avec photo")
    if filtered:
        for start in range(0, len(filtered), 3):
            cols = st.columns(3, gap="medium")
            for col, car in zip(cols, filtered[start:start + 3]):
                with col:
                    _render_car(car)
    else:
        st.warning("Aucune annonce n’a pu être récupérée automatiquement. Consulte les recherches La Centrale directement ; le statut ci-dessus précise si le site ou la méthode de repli a refusé l’accès.")
    with st.expander("État des recherches automatiques", expanded=True):
        for label, status in statuses.items():
            st.write(f"**{label}** : {status}")
    with st.expander("Ouvrir les recherches La Centrale"):
        for label, url in SEARCHES.items():
            st.markdown(f"- [{label}]({url})")
    st.caption("L’extraction dépend de la structure publique de La Centrale et peut être limitée. Vérifie toujours l’annonce, l’historique, l’entretien de la batterie hybride et la transmission exacte auprès du vendeur.")
