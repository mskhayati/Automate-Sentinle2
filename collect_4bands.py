from datetime import datetime, timedelta, timezone
import numpy as np
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


bbox_wgs84 = [36.73, 31.78, 36.84, 31.87]

# 2. Dynamic Date Range: Last 15 days from OS clock
now_utc = datetime.now(timezone.utc)
lookback_utc = now_utc - timedelta(days=15)
time_window = (
    f"{lookback_utc.strftime('%Y-%m-%d')}/{now_utc.strftime('%Y-%m-%d')}"
)

print(
    f"Current Date (UTC): {now_utc.strftime('%Y-%m-%d')} | Searching"
    f" window: {time_window}"
)

# 3. Connect to AWS STAC Catalog
catalog = pystac_client.Client.open(
    "https://earth-search.aws.element84.com/v1"
)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox_wgs84,
    datetime=time_window,
)

items = list(search.items())
if not items:
    raise RuntimeError("No cloud-free scenes found within the date range.")

# Sort to get the latest available scene
items.sort(key=lambda x: x.datetime, reverse=True)
latest_item = items[0]
capture_date = latest_item.datetime.strftime("%Y-%m-%d")
print(
    f"Streaming from Scene: {latest_item.id} (Acquired: {capture_date})"
)

# 4. Extract the 4 bands: Red (B04), Green (B03), Blue (B02), NIR (B08)
band_names = ["red", "green", "blue", "nir"]
band_arrays = []
out_profile = None

for band in band_names:
    band_url = latest_item.assets[band].href

    with rasterio.open(band_url) as src:
        # Reproject WGS84 degree coordinates to scene's native UTM projection
        minx, miny, maxx, maxy = transform_bounds(
            "EPSG:4326", src.crs, *bbox_wgs84
        )
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        # Read only the bounding box pixels
        data = src.read(1, window=window)
        band_arrays.append(data)

        # Capture geospatial profile on the first band
        if out_profile is None:
            win_transform = rasterio.windows.transform(window, src.transform)
            out_profile = src.profile.copy()
            out_profile.update({
                "count": 4,
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": win_transform,
                "dtype": "uint16",
            })

    print(f"  [STREAMED] {band.upper()} band ({data.shape[1]}x{data.shape[0]} px)")

# 5. Stack into a 4-band array [Shape: (4, H, W)]
four_band_stack = np.stack(band_arrays, axis=0)

# 6. Save as a standard multi-band GeoTIFF
output_filename = f"Muwaffaq_Salti_Air Base_4bands_{capture_date}.tif"
with rasterio.open(output_filename, "w", **out_profile) as dst:
    dst.write(four_band_stack)

print(f"\nSuccessfully saved: {output_filename}")
print(f"Stack Band Order -> 1: Red (B04), 2: Green (B03), 3: Blue (B02), 4: NIR (B08)")