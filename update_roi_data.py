# update_roi_data.py
"""
Fetch Dubai Land Department (DLD) real estate transactions from DubaiPulse API
and update data/roi_data.json with average price & ROI estimates by area and property type.
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

# Use the DubaiPulse DLD transactions API
DLD_JSON_URL = "https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open-api"  # examine if this returns JSON
# Sometimes you need query parameters like ?$limit=1000 or similar
PARAMS = {
    "page": 1,
    "per_page": 1000
}

def fetch_transactions(page=1, per_page=1000):
    params = {"page": page, "per_page": per_page}
    try:
        resp = requests.get(DLD_JSON_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ HTTP error fetching page {page}: {e}")
        return None

    try:
        data = resp.json()
    except ValueError as e:
        print(f"❌ JSON parse error on page {page}: {e}")
        return None

    return data

def extract_records(data):
    """
    Extract list of records from the API output.
    The key may be "data", "results", or direct list depending on API schema.
    """
    # Try common keys
    if isinstance(data, dict):
        for key in ("data", "results", "records", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Otherwise maybe it's root list
        if "result" in data and isinstance(data["result"], list):
            return data["result"]
    if isinstance(data, list):
        return data
    return []

def transform_records_to_roi(records):
    """
    Aggregates average price per area + property type,
    computes placeholder ROI.
    """
    area_prop_sums = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))

    for rec in records:
        # You must adjust these keys to match the API schema:
        prop_type = rec.get("Property Type") or rec.get("property_type") or rec.get("propertySubType") or rec.get("property_sub_type")
        area = rec.get("Area") or rec.get("area") or rec.get("area")  # adjust key
        amount = rec.get("Amount") or rec.get("amount") or rec.get("transactionAmount") or rec.get("price")

        # Skip if missing
        if not prop_type or not area or not amount:
            continue

        # Clean amount
        try:
            # Remove commas, currency words if present
            amt_str = str(amount).replace(",", "").replace("AED", "").strip()
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
            # Example placeholder ROI calculation: assume 6% always, or could use rent data when available
            roi_value = 6.0  # or adjust logic here
            roi[area][prop_type] = {
                "avg_price": round(avg_price, 2),
                "roi": round(roi_value, 2)
            }
    return roi

def save_roi(roi_dict, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roi_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved updated ROI data to {path}")
    except Exception as e:
        print(f"❌ Error saving ROI data: {e}")

def main():
    # Try fetching first page
    all_records = []
    page = 1
    while True:
        print(f"Fetching page {page}...")
        data = fetch_transactions(page=page, per_page=1000)
        if data is None:
            print("❌ Stopping due to fetch error.")
            break
        recs = extract_records(data)
        if not recs:
            print(f"⚠️ No records found on page {page}. Ending.")
            break
        all_records.extend(recs)
        # If pagination info available, check if more pages
        # Some APIs return total count or has_more flag
        # For now, break if fewer than per_page => likely last page
        if len(recs) < PARAMS["per_page"]:
            break
        page += 1

    if not all_records:
        print("❌ No records collected. Keeping old ROI data (if any).")
        return

    roi_dict = transform_records_to_roi(all_records)
    save_roi(roi_dict, ROI_JSON_PATH)

if __name__ == "__main__":
    main()
