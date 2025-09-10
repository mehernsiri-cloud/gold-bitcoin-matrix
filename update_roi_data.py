import json
import requests

# ------------------------------
# Config
# ------------------------------
# Replace this URL with the actual DLD JSON Open Data endpoint for property transactions
DLD_JSON_URL = "https://www.dubaipulse.gov.ae/data/dld-registration/dld_land_registry-open?format=json"

OUTPUT_JSON_PATH = "data/roi_data.json"

# ------------------------------
# Helper functions
# ------------------------------
def fetch_dld_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def transform_to_roi(data):
    """
    Transforms raw DLD JSON data to roi_data.json structure:
    roi_data[area][property_type] = {"avg_price": float, "roi": float}
    """
    roi_data = {}

    for record in data.get("result", []):
        # Map fields; adjust keys based on actual JSON structure
        prop_type = record.get("Property Type")
        area = record.get("Area")
        amount = record.get("Amount")

        if not prop_type or not area or not amount:
            continue

        # Clean amount to float
        try:
            amount = float(str(amount).replace(",", "").replace("AED", "").strip())
        except ValueError:
            continue

        # Initialize area/property_type
        if area not in roi_data:
            roi_data[area] = {}
        if prop_type not in roi_data[area]:
            roi_data[area][prop_type] = {"total_price": 0, "count": 0}

        roi_data[area][prop_type]["total_price"] += amount
        roi_data[area][prop_type]["count"] += 1

    # Compute average price and assign placeholder ROI
    for area, types in roi_data.items():
        for prop_type, values in types.items():
            count = values.pop("count")
            avg_price = values.pop("total_price") / count if count > 0 else 0
            # Placeholder ROI: example formula (adjust as needed)
            roi = 5 + (1 if avg_price > 1_000_000 else 0)
            roi_data[area][prop_type] = {"avg_price": avg_price, "roi": roi}

    return roi_data

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ------------------------------
# Main
# ------------------------------
def main():
    try:
        print("Fetching DLD JSON data...")
        raw_data = fetch_dld_json(DLD_JSON_URL)
        print("Transforming data into ROI format...")
        roi_data = transform_to_roi(raw_data)
        save_json(roi_data, OUTPUT_JSON_PATH)
        print(f"✅ ROI data updated successfully and saved to {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
