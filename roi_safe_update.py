#!/usr/bin/env python3
"""
roi_safe_update.py

Scrapes Bayut property listings (rent & sale) in UAE, computes avg price and placeholder ROI,
and safely updates data/roi_data.json. If scraping fails or no listings are found,
previous ROI backup is restored.
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
DATA_DIR = "data"
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
ROI_BACKUP_PATH = os.path.join(DATA_DIR, "roi_data_backup.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Bayut URLs (current working URLs)
URLS = {
    "apartments_rent": "https://www.bayut.com/to-rent/apartments/uae/",
    "villas_rent": "https://www.bayut.com/to-rent/villas/uae/",
    "apartments_sale": "https://www.bayut.com/for-sale/apartments/uae/",
    "villas_sale": "https://www.bayut.com/for-sale/villas/uae/"
}

# Maximum pages to scrape per URL
MAX_PAGES = 5

# Placeholder ROI value
PLACEHOLDER_ROI = 6.0

# ------------------------------
# Functions
# ------------------------------

def fetch_listings(url_base):
    """
    Fetch listings from Bayut with pagination.
    Returns a list of dicts: {'area': str, 'type': str, 'price': float}
    """
    listings = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{url_base}?page={page}"
        print(f"Fetching listings from: {url}")
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if resp.status_code != 200:
                print(f"Failed to fetch {url}: {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find listing cards
            cards = soup.find_all("div", {"data-testid": "listing-card"})
            if not cards:
                print(f"No listings found on page {page} for {url_base}")
                break

            for card in cards:
                try:
                    area_tag = card.select_one("div[data-testid='listing-location'] a")
                    area = area_tag.text.strip() if area_tag else "Unknown"

                    price_tag = card.select_one("span[data-testid='price']")
                    price_text = price_tag.text.strip() if price_tag else None
                    if not price_text:
                        continue
                    # Remove currency symbols and commas
                    price = float(
                        "".join(c for c in price_text if c.isdigit() or c == ".")
                    )

                    property_type = "apartment" if "apartment" in url_base else "villa"

                    listings.append({
                        "area": area,
                        "type": property_type,
                        "price": price
                    })
                except Exception as e:
                    print(f"Skipping a listing due to parse error: {e}")
            time.sleep(1)  # polite delay
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return listings


def compute_roi(listings):
    """
    Compute average prices per area & type and add placeholder ROI.
    Returns a nested dict: roi[area][type] = {'avg_price': ..., 'roi': ...}
    """
    area_type_sum = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))
    for l in listings:
        area_type_sum[l["area"]][l["type"]]["total"] += l["price"]
        area_type_sum[l["area"]][l["type"]]["count"] += 1

    roi = {}
    for area, types in area_type_sum.items():
        roi[area] = {}
        for prop_type, info in types.items():
            if info["count"] == 0:
                continue
            avg_price = info["total"] / info["count"]
            roi[area][prop_type] = {
                "avg_price": round(avg_price, 2),
                "roi": PLACEHOLDER_ROI
            }
    return roi


def save_roi_safe(roi):
    """
    Safely save ROI to JSON, restoring backup if empty
    """
    # Backup previous ROI
    if os.path.exists(ROI_JSON_PATH):
        try:
            os.replace(ROI_JSON_PATH, ROI_BACKUP_PATH)
        except Exception as e:
            print(f"⚠️ Failed to backup previous ROI: {e}")

    # Save new ROI
    try:
        with open(ROI_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(roi, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved ROI data to {ROI_JSON_PATH}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")
        # Restore backup
        if os.path.exists(ROI_BACKUP_PATH):
            os.replace(ROI_BACKUP_PATH, ROI_JSON_PATH)
            print("⚠️ Restored previous ROI backup due to save error")

    # Check if ROI is empty
    try:
        with open(ROI_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data and os.path.exists(ROI_BACKUP_PATH):
            os.replace(ROI_BACKUP_PATH, ROI_JSON_PATH)
            print("⚠️ ROI data empty, restored previous backup")
    except Exception:
        if os.path.exists(ROI_BACKUP_PATH):
            os.replace(ROI_BACKUP_PATH, ROI_JSON_PATH)
            print("⚠️ ROI data load failed, restored previous backup")


# ------------------------------
# Main
# ------------------------------
def main():
    all_listings = []
    for url in URLS.values():
        listings = fetch_listings(url)
        if listings:
            all_listings.extend(listings)

    if not all_listings:
        print("⚠️ No listings fetched. Using empty ROI dataset.")
        roi = {}
    else:
        roi = compute_roi(all_listings)

    save_roi_safe(roi)


if __name__ == "__main__":
    main()
