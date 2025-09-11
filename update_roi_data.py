"""
Fetch ROI data from a free public JSON feed (Numbeo),
update data/roi_data.json with avg_price + placeholder ROI by area & property type.
Structure and logging preserved from original DLD version.
"""

import os
import json
import requests
from collections import defaultdict

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ROI_JSON_PATH = os.path.join(DATA_DIR, "roi_data.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------
# Public JSON source (Numbeo)
# ------------------------------
NUMBEO_URL = "https://www.numbeo.com/api/city_prices"
CITY = "Dubai"
CURRENCY = "AED"

HEADERS = {
    "Accept": "application/json"
}

PARAMS = {
    "query": CITY,
    "currency": CURRENCY
}

def fetch_json(params):
    try:
        resp = requests.get(NUMBEO_URL, params=params, headers=HEADERS, timeout=30)
        print(f"Fetching {resp.url}")
        status = resp.status_code
        print(f"Status code: {status}")
        if status != 200:
            print(f"Non-200 response: {resp.text[:200]}")
            return None
        content = resp.content
        if not content or content.strip() == b'':
            print("Empty response content.")
            return None
        try:
            data = resp.json()
            return data
        except Exception as e:
            print(f"JSON decode error: {e} — first 500 bytes: {content[:500]}")
            return None
    except Exception as e:
        print(f"HTTP request error: {e}")
        return None

def extract_records(json_data):
    # Numbeo returns "prices" list
    if isinstance(json_data, dict) and "prices" in json_data:
        return json_data["prices"]
    return []

def transform_records_to_roi(records):
    area_prop_sums = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))

    # Use a flat mapping for simplicity: city center vs outside
    for rec in records:
        name = rec.get("item_name")
        avg_price = rec.get("average_price")

        if not name or avg_price is None:
            continue

        # Map names to property types / areas
        if "City Centre" in name:
            area = "city_center"
        elif "Outside of Centre" in name:
            area = "outside_center"
        else:
            area = "other"

        if "Apartment" in name and "1 bedroom" in name:
            prop_type = "1_bedroom_apartment"
        elif "Apartment" in name and "2 bedrooms" in name:
            prop_type = "2_bedroom_apartment"
        else:
            prop_type = "other_property"

        # accumulate
        d = area_prop_sums[area][prop_type]
        try:
            amt = float(avg_price)
            if amt <= 0:
                continue
            d["total"] += amt
            d["count"] += 1
        except Exception:
            continue

    # Build ROI dict
    roi = {}
    for area, props in area_prop_sums.items():
        roi[area] = {}
        for prop_type, info in props.items():
            cnt = info["count"]
            if cnt == 0:
                continue
            avg_price = info["total"] / cnt
            # Placeholder ROI — replace later with more accurate formula
            roi_value = 6.0
            roi[area][prop_type] = {"avg_price": round(avg_price, 2), "roi": round(roi_value, 2)}

    return roi

def save_roi(roi_dict, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roi_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved ROI data to {path}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")

def main():
    all_records = []
    json_data = fetch_json(PARAMS)
    if not json_data:
        print("❌ Failed to fetch valid JSON data. Using empty ROI dataset.")
        save_roi({}, ROI_JSON_PATH)
        return

    recs = extract_records(json_data)
    if not recs:
        print("⚠️ No records found in fetched JSON. Using empty ROI dataset.")
        save_roi({}, ROI_JSON_PATH)
        return

    all_records.extend(recs)
    roi_dict = transform_records_to_roi(all_records)
    save_roi(roi_dict, ROI_JSON_PATH)

if __name__ == "__main__":
    main()
