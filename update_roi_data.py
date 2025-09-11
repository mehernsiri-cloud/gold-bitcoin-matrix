"""
update_roi_data.py
Fetches Dubai property listings from Bayut’s hidden JSON feed,
computes average prices per area/type, adds placeholder ROI,
and saves safely to data/roi_data.json
"""

import os
import json
import re
import requests
from collections import defaultdict

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
ROI_BACKUP_PATH = os.path.join(DATA_DIR, "roi_data_backup.json")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
}

MAX_LISTINGS_PER_TYPE = 100
PLACEHOLDER_ROI = 6.0  # %

# Property types & URLs to fetch (to-rent / for-sale)
PROPERTY_TYPES = {
    "for-sale-apartments": "https://www.bayut.com/for-sale/apartments/dubai/",
    "for-sale-villas": "https://www.bayut.com/for-sale/villas/dubai/",
    "to-rent-apartments": "https://www.bayut.com/to-rent/apartments/dubai/",
    "to-rent-villas": "https://www.bayut.com/to-rent/villas/dubai/"
}

# ------------------------------
# Functions
# ------------------------------
def get_build_id():
    """Fetch main Bayut page and extract current BUILD_ID."""
    url = "https://www.bayut.com/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        match = re.search(r'"BUILD_ID":"([a-zA-Z0-9]+)"', resp.text)
        if match:
            build_id = match.group(1)
            print(f"✅ Found BUILD_ID: {build_id}")
            return build_id
    except Exception as e:
        print(f"❌ Failed to fetch BUILD_ID: {e}")
    return None

def fetch_json_listings(build_id, category_url):
    """Fetch listings JSON using hidden Bayut feed."""
    try:
        category_path = category_url.replace("https://www.bayut.com/", "")
        json_url = f"https://www.bayut.com/_next/data/{build_id}/{category_path}.json"
        resp = requests.get(json_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Traverse to listings array
        listings = []
        # Attempt common paths
        for key in ["pageProps", "initialState", "listingSearch"]:
            temp = data
            for k in key.split("."):
                temp = temp.get(k, {})
            if isinstance(temp, dict) and "listings" in temp:
                listings = temp["listings"]
                break
        return listings[:MAX_LISTINGS_PER_TYPE]
    except Exception as e:
        print(f"❌ Failed to fetch JSON for {category_url}: {e}")
        return []

def compute_roi(listings):
    """Compute average price per area/type with placeholder ROI."""
    area_prop_sums = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))
    for rec in listings:
        try:
            area = rec.get("community", rec.get("location", "Unknown"))
            prop_type = rec.get("propertyType", "Unknown")
            price = rec.get("price")
            if price is None:
                continue
            d = area_prop_sums[area][prop_type]
            d["total"] += price
            d["count"] += 1
        except Exception:
            continue

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
    """Save ROI safely; restore backup if empty or fails."""
    # Backup previous
    if os.path.exists(path):
        try:
            os.replace(path, ROI_BACKUP_PATH)
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
            os.replace(ROI_BACKUP_PATH, path)
            print("⚠️ Restored previous ROI backup due to save error")

    # Check empty ROI
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data and os.path.exists(ROI_BACKUP_PATH):
            os.replace(ROI_BACKUP_PATH, path)
            print("⚠️ ROI data empty, restored previous backup")
    except Exception:
        if os.path.exists(ROI_BACKUP_PATH):
            os.replace(ROI_BACKUP_PATH, path)
            print("⚠️ ROI data load failed, restored previous backup")

# ------------------------------
# Main
# ------------------------------
def main():
    build_id = get_build_id()
    if not build_id:
        print("❌ Could not locate BUILD_ID. Exiting.")
        return

    all_listings = []
    for name, url in PROPERTY_TYPES.items():
        print(f"Fetching JSON listings for: {name}")
        listings = fetch_json_listings(build_id, url)
        if not listings:
            print(f"⚠️ No listings found for {name}")
        all_listings.extend(listings)

    if not all_listings:
        print("⚠️ No listings fetched. Using empty ROI dataset.")
        save_roi_safe({})
        return

    roi_dict = compute_roi(all_listings)
    save_roi_safe(roi_dict)

if __name__ == "__main__":
    main()
