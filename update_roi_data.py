import csv
import json
import requests

# Define the URL for the CSV data
CSV_URL = 'https://www.dubaipulse.gov.ae/data/dld-registration/dld_land_registry-open?organisation=dld&page=6&service=dld-registration'

# Define the path for the output JSON file
OUTPUT_JSON_PATH = 'roi_data.json'

def fetch_csv_data(url):
    """Fetch CSV data from the provided URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_csv_to_dict(csv_data):
    """Parse CSV data into a list of dictionaries."""
    reader = csv.DictReader(csv_data.splitlines())
    return [row for row in reader]

def transform_data_to_roi_format(parsed_data):
    """Transform parsed data into the desired ROI format."""
    roi_data = []
    for row in parsed_data:
        roi_entry = {
            'transaction_id': row['Transaction Number'],
            'transaction_date': row['Transaction Date'],
            'transaction_type': row['Transaction Type'],
            'property_type': row['Property Type'],
            'amount': row['Amount'],
            'property_size': row['Property Size (sq.m)'],
            'location': {
                'area': row['Area'],
                'zone': row['Zone']
            }
        }
        roi_data.append(roi_entry)
    return roi_data

def save_to_json(data, path):
    """Save the transformed data to a JSON file."""
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def main():
    """Main function to orchestrate the data fetching, processing, and saving."""
    try:
        # Fetch and parse the CSV data
        csv_data = fetch_csv_data(CSV_URL)
        parsed_data = parse_csv_to_dict(csv_data)

        # Transform the data into the desired format
        roi_data = transform_data_to_roi_format(parsed_data)

        # Save the transformed data to a JSON file
        save_to_json(roi_data, OUTPUT_JSON_PATH)

        print(f"ROI data successfully updated and saved to {OUTPUT_JSON_PATH}")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
