"""
update_roi_data.py
Fetch Dubai property listings from Bayut's hidden JSON feed,
compute average prices per area/type, add placeholder ROI,
save safely to data/roi_data.json with backup restore.
"""

import os
import json
import requests
import re
from collections import defaultdict
import shutil

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
ROI_BACKUP_PATH = os.path.join(DATA_DIR, "roi_data_backup.json")
os.makedirs(DATA_DIR, exist_ok=True)

BAYUT_MAIN_URLS = [
    "https://www.bayut.com/to-rent/apartments/dubai/",
    "https://www.bayut.com/to-rent/villas/dubai/",
    "https://www.bayut.com/for-sale/apartments/dubai/",
    "https://www.bayut.com/for-sale/villas/dubai/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36"
}

PLACEHOLDER_ROI = 6.0  # %

# ------------------------------
# Functions
# ------------------------------

def get_build_id(url):
    """Auto-locate BUILD_ID from Bayut page HTML."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch {url}: {resp.status_code}")
            return None
        # Search for BUILD_ID in the page
        match = re.search(r'"BUILD_ID":"(.*?)"', resp.text)
        if match:
            return match.group(1)
        else:
            print(f"⚠️ BUILD_ID not found in {url}")
            return None
    except Exception as e:
        print(f"❌ Error fetching {url} for BUILD_ID: {e}")
        return None

def fetch_listings_from_json(build_id, page=1, type_filter="rent"):
    """Fetch listings JSON using hidden feed."""
    url = f"https://www.bayut.com/_next/data/{build_id}/{type_filter}/dubai.json?page={page}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"⚠️ JSON feed failed {url}: {resp.status_code}")
            return []
        data = resp.json()
        # Traverse to listings (structure may change)
        # Typically: pageProps -> listings
        listings = []
        hits = data.get("pageProps", {}).get("listings", {}).get("hits", [])
        for item in hits:
            prop_type = item.get("property_type", "Unknown")
            area = item.get("community_name", "Unknown")
            price = item.get("price", None)
            if prop_type and area and price is not None:
                listings.append({"property_type": prop_type, "area": area, "price": float(price)})
        return listings
    except Exception as e:
        print(f"❌ Error fetching JSON listings: {e}")
        return []

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
            roi[area][prop_type] = {"avg_price": round(avg_price, 2), "roi": PLACEHOLDER_ROI}
    return roi

def save_roi_safe(roi_dict, path=ROI_JSON_PATH):
    """Save ROI safely; restore previous if empty or fails."""
    # Backup previous
    if os.path.exists(path):
        try:
            shutil.copyfile(path, ROI_BACKUP_PATH)
        except Exception as e:
            print(f"⚠️ Failed to backup previous ROI: {e}")

    # Save new ROI
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roi_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved ROI data to {path}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")
        if os.path.exists(ROI_BACKUP_PATH):
            shutil.copyfile(ROI_BACKUP_PATH, path)
            print("⚠️ Restored previous ROI backup due to save error")

    # Restore backup if ROI is empty
    if not roi_dict and os.path.exists(ROI_BACKUP_PATH):
        shutil.copyfile(ROI_BACKUP_PATH, path)
        print("⚠️ ROI data empty, restored previous backup")

# ------------------------------
# Main
# ------------------------------

def main():
    all_listings = []

    for url in BAYUT_MAIN_URLS:
        print(f"Processing {url}")
        build_id = get_build_id(url)
        if not build_id:
            continue
        # Fetch first 2 pages for each URL
        for page in range(1, 3):
            listings = fetch_listings_from_json(build_id, page=page)
            all_listings.extend(listings)

    if not all_listings:
        print("⚠️ No listings fetched. Using empty ROI dataset.")
        save_roi_safe({})
        return

    roi_dict = compute_roi(all_listings)
    save_roi_safe(roi_dict)

if __name__ == "__main__":
    main()
