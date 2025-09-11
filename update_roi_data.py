# update_roi_data.py
"""
Fetch DLD real estate transactions from DubaiPulse JSON API,
update data/roi_data.json with avg_price + placeholder ROI by area & property type.
Improved to handle empty responses and restricted endpoints.
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

# DLD JSON endpoint
DLD_JSON_URL = "https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open-api"

# Try with parameters
PARAMS = {
    "limit": 1000,   # or "page" / "per_page" if supported
    "page": 1
}

# Some APIs require a header to accept JSON
HEADERS = {
    "Accept": "application/json"
}

def fetch_json(params):
    try:
        resp = requests.get(DLD_JSON_URL, params=params, headers=HEADERS, timeout=30)
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
    # Try common patterns
    if isinstance(json_data, dict):
        for key in ("data", "records", "results", "items"):
            if key in json_data and isinstance(json_data[key], list):
                return json_data[key]
        # Socrata style?
        if "result" in json_data and isinstance(json_data["result"], list):
            return json_data["result"]
    if isinstance(json_data, list):
        return json_data
    return []

def transform_records_to_roi(records):
    area_prop_sums = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))

    for rec in records:
        # Adjust keys based on what you see in JSON
        prop_type = rec.get("Property Type") or rec.get("property_type") or rec.get("PropertySubType") or rec.get("Property Sub Type")
        area = rec.get("Area") or rec.get("area") or rec.get("community") or rec.get("District")
        amount_field = rec.get("Amount") or rec.get("amount") or rec.get("Transaction Amount") or rec.get("transactionAmount") or rec.get("price")

        if not prop_type or not area or not amount_field:
            continue

        # Clean amount
        try:
            amt_str = str(amount_field).replace(",", "").replace("AED", "").strip()
            amt = float(amt_str)
            if amt <= 0:
                continue
        except Exception:
            continue

        d = area_prop_sums[area][prop_type]
        d["total"] += amt
        d["count"] += 1

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
    # try first page
    json_data = fetch_json(PARAMS)
    if not json_data:
        print("❌ Failed to fetch valid JSON data.")
        return
    recs = extract_records(json_data)
    if not recs:
        print("⚠️ No records found in fetched JSON.")
        return
    all_records.extend(recs)

    # If the API supports pagination (check if json_data includes total or next-page)
    # Here we just skip paging for simplicity.

    roi_dict = transform_records_to_roi(all_records)
    save_roi(roi_dict, ROI_JSON_PATH)

if __name__ == "__main__":
    main()
