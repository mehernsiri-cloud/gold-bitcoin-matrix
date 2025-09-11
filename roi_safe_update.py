"""
roi_safe_update.py
Scrapes static Dubai property listings (Bayut.com),
computes average prices per area/type, adds placeholder ROI,
saves safely to data/roi_data.json, restores backup if empty.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import time
import shutil

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
ROI_BACKUP_PATH = os.path.join(DATA_DIR, "roi_data_backup.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Bayut URLs that currently work for Dubai listings
BAYUT_URLS = [
    "https://www.bayut.com/to-rent/apartments/dubai/",
    "https://www.bayut.com/to-rent/villas/dubai/",
    "https://www.bayut.com/for-sale/apartments/dubai/",
    "https://www.bayut.com/for-sale/villas/dubai/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
}

MAX_PAGES = 2  # Limit pages to avoid overloading

# ------------------------------
# Functions
# ------------------------------
def fetch_listings(url, max_pages=MAX_PAGES):
    """Fetch listings HTML from Bayut (static content)."""
    listings = []
    for page in range(1, max_pages + 1):
        full_url = f"{url}?page={page}"
        try:
            resp = requests.get(full_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"Failed to fetch {full_url}: {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.find_all("li", {"class": "srp-item"})  # Current static card class
            if not cards:
                print(f"No listings found on page {page} for {url}")
                continue

            for card in cards:
                try:
                    # Property type
                    prop_type_tag = card.find("h2")
                    prop_type = prop_type_tag.get_text(strip=True) if prop_type_tag else "Unknown"

                    # Area / community
                    area_tag = card.find("div", {"class": "srp-item-location"})
                    area = area_tag.get_text(strip=True) if area_tag else "Unknown"

                    # Price
                    price_tag = card.find("div", {"class": "srp-item-price"})
                    price_text = price_tag.get_text(strip=True) if price_tag else None
                    if price_text:
                        price_text = (
                            price_text.replace(",", "")
                            .replace("AED", "")
                            .replace("/month", "")
                            .replace("/yr", "")
                            .strip()
                        )
                        price = float(price_text)
                    else:
                        price = None

                    if prop_type and area and price:
                        listings.append({"property_type": prop_type, "area": area, "price": price})
                except Exception:
                    continue
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {full_url}: {e}")
    return listings

def compute_roi(listings):
    """Compute average price per area/type with placeholder ROI."""
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
            roi_value = 6.0  # Placeholder ROI %
            roi[area][prop_type] = {"avg_price": round(avg_price, 2), "roi": round(roi_value, 2)}
    return roi

def save_roi_safely(roi_dict):
    """Save ROI safely; restore backup if empty."""
    try:
        # Backup previous ROI
        if os.path.exists(ROI_JSON_PATH):
            shutil.copyfile(ROI_JSON_PATH, ROI_BACKUP_PATH)
        # Save new ROI
        with open(ROI_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(roi_dict, f, indent=2, ensure_ascii=False)

        # Restore backup if ROI is empty
        if not roi_dict:
            print("⚠️ ROI data empty, restoring previous backup if exists.")
            if os.path.exists(ROI_BACKUP_PATH):
                shutil.copyfile(ROI_BACKUP_PATH, ROI_JSON_PATH)
        else:
            print(f"✅ Saved ROI data to {ROI_JSON_PATH}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")
        # Restore backup on any exception
        if os.path.exists(ROI_BACKUP_PATH):
            shutil.copyfile(ROI_BACKUP_PATH, ROI_JSON_PATH)

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
        print("⚠️ No listings fetched. Using empty ROI dataset.")
        save_roi_safely({})
        return

    roi_dict = compute_roi(all_listings)
    save_roi_safely(roi_dict)

if __name__ == "__main__":
    main()
