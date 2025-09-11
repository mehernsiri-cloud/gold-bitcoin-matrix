"""
roi_safe_update.py
Scrapes Dubai property listings from Bayut.com (public listings),
extracts area, type, price, computes average prices per area/type,
adds placeholder ROI, and safely saves to data/roi_data.json.
If the new scrape is empty, restores previous backup.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import time

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
ROI_BACKUP_PATH = os.path.join(DATA_DIR, "roi_data_backup.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Bayut URLs for Dubai properties
BAYUT_URLS = [
    "https://www.bayut.com/to-rent/apartments/uae/",
    "https://www.bayut.com/to-rent/villas/uae/",
    "https://www.bayut.com/for-sale/apartments/uae/",
    "https://www.bayut.com/for-sale/villas/uae/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
}

MAX_PAGES = 3  # adjust as needed

# ------------------------------
# Scraper
# ------------------------------
def fetch_listings(url, max_pages=MAX_PAGES):
    listings = []
    for page in range(1, max_pages + 1):
        full_url = f"{url}?page={page}"
        try:
            resp = requests.get(full_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"Failed to fetch {full_url}: {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            # New listing cards structure
            cards = soup.find_all("div", class_="ListingCard")
            if not cards:
                print(f"No listings found on page {page} for {url}")
                continue

            for card in cards:
                try:
                    # Property type (rent/sale + apartment/villa)
                    prop_type_tag = card.find("h2")
                    prop_type = prop_type_tag.get_text(strip=True) if prop_type_tag else "Unknown"

                    # Area / community
                    area_tag = card.find("div", class_="ListingCard__location")
                    area = area_tag.get_text(strip=True) if area_tag else "Unknown"

                    # Price
                    price_tag = card.find("span", class_="ListingPrice__price")
                    price_text = price_tag.get_text(strip=True) if price_tag else None
                    if price_text:
                        price_text = price_text.replace(",", "").replace("AED", "").replace("/month", "").replace("د.إ", "").strip()
                        try:
                            price = float(price_text)
                        except ValueError:
                            price = None
                    else:
                        price = None

                    if prop_type and area and price:
                        listings.append({"property_type": prop_type, "area": area, "price": price})
                except Exception:
                    continue

            time.sleep(1)  # polite delay between pages
        except Exception as e:
            print(f"Error fetching {full_url}: {e}")
    return listings

# ------------------------------
# ROI computation
# ------------------------------
def compute_roi(listings):
    area_prop_sums = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))
    for rec in listings:
        area = rec["area"]
        prop_type = rec["property_type"]
        price = rec["price"]
        d = area_prop_sums[area][prop_type]
        d["total"] += price
        d["count"] += 1

    roi = {}
    for area, props in area_prop_sums.items():
        roi[area] = {}
        for prop_type, info in props.items():
            if info["count"] == 0:
                continue
            avg_price = info["total"] / info["count"]
            roi_value = 6.0  # placeholder ROI %
            roi[area][prop_type] = {"avg_price": round(avg_price, 2), "roi": round(roi_value, 2)}
    return roi

# ------------------------------
# Safe save
# ------------------------------
def save_roi_safe(new_roi, path, backup_path):
    try:
        # Backup old data
        if os.path.exists(path):
            os.replace(path, backup_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_roi, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved ROI data to {path}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")
        # Restore backup if failure
        if os.path.exists(backup_path):
            os.replace(backup_path, path)
            print(f"⚠️ Restored previous ROI backup due to save error")

# ------------------------------
# Main
# ------------------------------
def main():
    all_listings = []
    for url in BAYUT_URLS:
        print(f"Fetching listings from: {url}")
        listings = fetch_listings(url)
        all_listings.extend(listings)

    if not all_listings:
        print("⚠️ No listings fetched. Restoring previous ROI backup if exists.")
        if os.path.exists(ROI_BACKUP_PATH):
            os.replace(ROI_BACKUP_PATH, ROI_JSON_PATH)
        return

    roi_dict = compute_roi(all_listings)
    save_roi_safe(roi_dict, ROI_JSON_PATH, ROI_BACKUP_PATH)

if __name__ == "__main__":
    main()
