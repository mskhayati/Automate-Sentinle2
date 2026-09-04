import json
import os
from datetime import datetime, timedelta, timezone
import cv2
import numpy as np
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

# 1. Load locations config
with open("locations.json", "r", encoding="utf-8") as f:
    LOCATIONS = json.load(f)

BASE_OUTPUT_DIR = "./output_aws"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 2. Connect to AWS STAC
catalog = pystac_client.Client.open(
    "https://earth-search.aws.element84.com/v1"
)
now_utc = datetime.now(timezone.utc)
lookback_utc = now_utc - timedelta(days=15)
time_window = (
    f"{lookback_utc.strftime('%Y-%m-%d')}/{now_utc.strftime('%Y-%m-%d')}"
)

print(
    f"System Date: {now_utc.strftime('%Y-%m-%d')} | Checking 15-day window..."
)

# 3. Process each location
for loc_key, loc_info in LOCATIONS.items():
    loc_name = loc_info["name"]
    bbox_wgs84 = loc_info["bbox"]
    loc_dir = os.path.join(BASE_OUTPUT_DIR, loc_key)
    os.makedirs(loc_dir, exist_ok=True)

    print(f"\n--- Checking: {loc_name} ({loc_info['country']}) ---")

    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox_wgs84, datetime=time_window
    )
    items = list(search.items())
    items.sort(key=lambda x: x.datetime, reverse=True)

    # Deduplicate to extract the last 2 unique acquisition scenes
    unique_scenes = []
    seen_dates = set()
    for item in items:
        d = item.datetime.strftime("%Y-%m-%d")
        if d not in seen_dates:
            seen_dates.add(d)
            unique_scenes.append((d, item))
        if len(unique_scenes) == 2:
            break

    if not unique_scenes:
        print(f"No scenes found on AWS for {loc_name}.")
        continue

    print(f"Target passes to verify: {[d for d, _ in unique_scenes]}")

    for capture_date, item in unique_scenes:
        tc_file = os.path.join(
            loc_dir, f"{loc_key}_True_Color_{capture_date}.png"
        )
        opt_file = os.path.join(
            loc_dir, f"{loc_key}_Highlight_Optimized_{capture_date}.png"
        )

        # Check if both files already exist
        if os.path.exists(tc_file) and os.path.exists(opt_file):
            print(f"  [SKIP] Files for {capture_date} already exist.")
            continue

        try:
            # 1. Download True Color if missing
            if not os.path.exists(tc_file) and "visual" in item.assets:
                with rasterio.open(item.assets["visual"].href) as src:
                    minx, miny, maxx, maxy = transform_bounds(
                        "EPSG:4326", src.crs, *bbox_wgs84
                    )
                    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                    tci = src.read([1, 2, 3], window=win)
                    cv2.imwrite(
                        tc_file,
                        cv2.cvtColor(tci.transpose(1, 2, 0), cv2.COLOR_RGB2BGR),
                    )
                print(f"  [DOWNLOADED] {os.path.basename(tc_file)}")

            # 2. Download and Compute Highlight Optimized if missing
            if not os.path.exists(opt_file):
                rgb_data = {}
                for b in ["red", "green", "blue"]:
                    with rasterio.open(item.assets[b].href) as src:
                        minx, miny, maxx, maxy = transform_bounds(
                            "EPSG:4326", src.crs, *bbox_wgs84
                        )
                        win = from_bounds(
                            minx, miny, maxx, maxy, transform=src.transform
                        )
                        rgb_data[b] = (
                            src.read(1, window=win).astype(np.float32) / 10000.0
                        )

                r_opt = np.cbrt(0.6 * np.clip(rgb_data["red"], 0, None))
                g_opt = np.cbrt(0.6 * np.clip(rgb_data["green"], 0, None))
                b_opt = np.cbrt(0.6 * np.clip(rgb_data["blue"], 0, None))

                stacked = np.stack([r_opt, g_opt, b_opt], axis=-1)
                img_uint8 = np.clip(stacked * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(opt_file, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                print(f"  [DOWNLOADED] {os.path.basename(opt_file)}")

        except Exception as e:
            print(f"  [ERROR] Processing {capture_date}: {e}")