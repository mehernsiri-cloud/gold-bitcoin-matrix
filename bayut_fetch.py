import requests
import re
import csv

BASE_URL = "https://www.bayut.com"
LISTINGS_PATH = "/to-rent/apartments/dubai/"

def get_build_id():
    """Fetch the Bayut homepage and extract Next.js BUILD_ID"""
    url = BASE_URL + LISTINGS_PATH
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise Exception(f"Failed to load homepage: {r.status_code}")

    # Look for the buildId in the HTML
    match = re.search(r'"buildId":"(.*?)"', r.text)
    if not match:
        raise Exception("Could not find buildId")
    return match.group(1)

def fetch_listings(build_id, page=1):
    """Fetch listings JSON using hidden API"""
    url = f"{BASE_URL}/_next/data/{build_id}/en{LISTINGS_PATH}.json?page={page}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise Exception(f"Failed to fetch listings: {r.status_code}")
    return r.json()

def parse_listings(data):
    """Extract useful fields from JSON"""
    properties = []
    try:
        props = data["pageProps"]["props"]["hits"]  # Bayut JSON structure
    except KeyError:
        props = []

    for p in props:
        properties.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "price": p.get("price"),
            "area": p.get("area"),
            "location": " / ".join([loc["name"] for loc in p.get("location", [])]),
            "url": BASE_URL + p.get("slug", "")
        })
    return properties

def save_to_csv(properties, filename="bayut_listings.csv"):
    keys = properties[0].keys() if properties else []
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(properties)

if __name__ == "__main__":
    build_id = get_build_id()
    print(f"[INFO] Found buildId: {build_id}")

    data = fetch_listings(build_id, page=1)
    listings = parse_listings(data)

    if listings:
        save_to_csv(listings)
        print(f"[INFO] Saved {len(listings)} listings to bayut_listings.csv")
    else:
        print("[WARN] No listings found")
