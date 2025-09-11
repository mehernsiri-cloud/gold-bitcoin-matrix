"""
update_roi_data.py
Fetches Dubai property listings from Bayut.com using Playwright,
extracts area, type, price, computes average prices per area/type,
adds placeholder ROI, and saves to data/roi_data.json safely.
"""

import os
import json
import asyncio
from collections import defaultdict
import shutil
from playwright.async_api import async_playwright

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
ROI_BACKUP_PATH = os.path.join(DATA_DIR, "roi_data_backup.json")
os.makedirs(DATA_DIR, exist_ok=True)

PLACEHOLDER_ROI = 6.0  # %

# Bayut pages
BAYUT_URLS = [
    "https://www.bayut.com/for-sale/apartments/dubai/",
    "https://www.bayut.com/for-sale/villas/dubai/",
    "https://www.bayut.com/to-rent/apartments/dubai/",
    "https://www.bayut.com/to-rent/villas/dubai/"
]

# ------------------------------
# Functions
# ------------------------------

async def fetch_listings_playwright(url):
    """Fetch listings from Bayut page using Playwright (JS rendered)."""
    listings = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)  # 60 sec timeout
        await page.wait_for_selector("div[class*='ListingCard']")  # Wait until listings render

        # Extract all listings
        cards = await page.query_selector_all("div[class*='ListingCard']")
        for card in cards:
            try:
                prop_type = await card.query_selector_eval("h2", "el => el.innerText") or "Unknown"
                area = await card.query_selector_eval("div[class*='PropertyLocation']", "el => el.innerText") or "Unknown"
                price_text = await card.query_selector_eval("div[class*='Price']", "el => el.innerText") or None
                if price_text:
                    price_text = price_text.replace(",", "").replace("AED", "").replace("/month", "").replace("/yr", "").strip()
                    price = float(price_text)
                    listings.append({"property_type": prop_type.strip(), "area": area.strip(), "price": price})
            except Exception:
                continue
        await browser.close()
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
            roi[area][prop_type] = {"avg_price": round(avg_price, 2), "roi": round(PLACEHOLDER_ROI, 2)}
    return roi

def save_roi_safely(roi_dict):
    """Save ROI safely; restore backup if empty."""
    try:
        if os.path.exists(ROI_JSON_PATH):
            shutil.copyfile(ROI_JSON_PATH, ROI_BACKUP_PATH)
        with open(ROI_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(roi_dict, f, indent=2, ensure_ascii=False)

        if not roi_dict and os.path.exists(ROI_BACKUP_PATH):
            shutil.copyfile(ROI_BACKUP_PATH, ROI_JSON_PATH)
            print("⚠️ ROI data empty, restored previous backup")
        else:
            print(f"✅ Saved ROI data to {ROI_JSON_PATH}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")
        if os.path.exists(ROI_BACKUP_PATH):
            shutil.copyfile(ROI_BACKUP_PATH, ROI_JSON_PATH)

# ------------------------------
# Main
# ------------------------------

async def main():
    all_listings = []
    for url in BAYUT_URLS:
        print(f"Fetching listings from: {url}")
        try:
            listings = await fetch_listings_playwright(url)
            all_listings.extend(listings)
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")

    if not all_listings:
        print("⚠️ No listings fetched. Using empty ROI dataset.")
        save_roi_safely({})
        return

    roi_dict = compute_roi(all_listings)
    save_roi_safely(roi_dict)

if __name__ == "__main__":
    asyncio.run(main())
