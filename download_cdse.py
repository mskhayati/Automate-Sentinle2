import json
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pystac_client
import requests

# 1. Load environment variables from .env
load_dotenv()
CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing CDSE credentials in .env file.")

# 2. Load locations config
with open("locations.json", "r", encoding="utf-8") as f:
    LOCATIONS = json.load(f)

BASE_OUTPUT_DIR = "./output_cdse"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# 3. Authenticate with CDSE Keycloak
def get_access_token(client_id, client_secret):
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    resp = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


token = get_access_token(CLIENT_ID, CLIENT_SECRET)
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# 4. Define Visual Evalscripts
EVALSCRIPTS = {
    "True_Color": """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["B02", "B03", "B04"] }],
        output: { id: "default", bands: 3, sampleType: "AUTO" }
      };
    }
    function evaluatePixel(sample) {
      return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """,
    "Highlight_Optimized": """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["B02", "B03", "B04"] }],
        output: { id: "default", bands: 3, sampleType: "AUTO" }
      };
    }
    function evaluatePixel(sample) {
      return [
        Math.cbrt(0.6 * sample.B04),
        Math.cbrt(0.6 * sample.B03),
        Math.cbrt(0.6 * sample.B02)
      ];
    }
    """,
}

# 5. Process each location
catalog = pystac_client.Client.open(
    "https://catalogue.dataspace.copernicus.eu/stac"
)
now_utc = datetime.now(timezone.utc)
lookback_utc = now_utc - timedelta(days=15)
time_window = (
    f"{lookback_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
    f"{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
)

print(
    f"System Date: {now_utc.strftime('%Y-%m-%d')} | Checking 10-day window..."
)

for loc_key, loc_info in LOCATIONS.items():
    loc_name = loc_info["name"]
    bbox = loc_info["bbox"]
    loc_dir = os.path.join(BASE_OUTPUT_DIR, loc_key)
    os.makedirs(loc_dir, exist_ok=True)

    print(f"\n--- Checking: {loc_name} ({loc_info['country']}) ---")

    # Search scenes
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_window
    )
    items = list(search.items())
    items.sort(key=lambda x: x.datetime, reverse=True)

    # Deduplicate to extract the last 2 unique acquisition dates
    unique_dates = []
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        if date_str not in unique_dates:
            unique_dates.append(date_str)
        if len(unique_dates) == 2:
            break

    if not unique_dates:
        print(f"No recent scenes found for {loc_name}.")
        continue

    print(f"Target passes to verify: {unique_dates}")

    # Process both dates
    for capture_date in unique_dates:
        for script_name, script in EVALSCRIPTS.items():
            filename = f"{loc_key}_{script_name}_{capture_date}.png"
            file_path = os.path.join(loc_dir, filename)

            # Idempotency check: Skip if already downloaded
            if os.path.exists(file_path):
                print(f"  [SKIP] {filename} already exists.")
                continue

            # Request image from Process API
            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": {
                            "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                        },
                    },
                    "data": [{
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{capture_date}T00:00:00Z",
                                "to": f"{capture_date}T23:59:59Z",
                            }
                        },
                    }],
                },
                "output": {
                    "width": 1280,
                    "height": 1024,
                    "responses": [{
                        "identifier": "default",
                        "format": {"type": "image/png"},
                    }],
                },
                "evalscript": script,
            }

            try:
                res = requests.post(
                    "https://sh.dataspace.copernicus.eu/api/v1/process",
                    headers=headers,
                    json=payload,
                )
                res.raise_for_status()
                with open(file_path, "wb") as out_f:
                    out_f.write(res.content)
                print(f"  [DOWNLOADED] {filename}")
            except Exception as e:
                print(f"  [ERROR] Failed downloading {filename}: {e}")