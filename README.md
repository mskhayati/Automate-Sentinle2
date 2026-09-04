# Sentinel-2 Strategic Site Monitoring & AI Super-Resolution (SEN2SR)

An end-to-end Python pipeline to monitor strategic sites and military installations using Sentinel-2 multispectral satellite imagery. The repository includes automated tools for multi-temporal visual pass comparison (True Color and Highlight-Optimized Natural Color), direct Cloud-Optimized GeoTIFF (COG) band harvesting via AWS Open Data, and deep-learning-based 4× spatial super-resolution using `tacofoundation/SEN2SR` (enhancing 10 m Ground Sample Distance down to 2.5 m).

---

## Repository Layout

```text
.
├── .env.example          # Sample environment credentials template
├── .gitignore            # Git exclusion rules
├── locations.json        # GeoJSON bounding boxes for target installations
├── download_cdse.py      # Method 1: Copernicus Data Space Process API downloader
├── download_aws.py       # Method 2: AWS Public COGs window-streaming downloader
├── collect_4bands.py     # Crop and extract 4-band 10m rasters (B04, B03, B02, B08)
├── run_sen2sr.py         # Patch-based 4x AI Super-Resolution pipeline (10m -> 2.5m)
└── README.md
```
---

## Installation & Setup:
1. Clone the Repository:
```text
git clone https://github.com/mskhayati/Automate-Sentinle2.git
cd Automate-Sentinle2
```
2. Set Up a Virtual Environment (Python 3.10+)
# Windows
```text
python -m venv venv
venv\Scripts\activate
```
# Linux / macOS
```text
python3 -m venv venv
source venv/bin/activate
```
3. Install PyTorch with CUDA SupportIf you are running on an NVIDIA GPU (recommended for super-resolution):Bash# Example for CUDA 12.x
```text
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
4. Install Project Requirements:
```text
pip install rasterio numpy opencv-python pystac-client requests python-dotenv tqdm mlstac sen2sr safetensors
```
Configurations:
1. Copernicus Data Space Credentials (.env):
To use Method 1 (download_cdse.py), generate OAuth client credentials on the Copernicus Data Space Ecosystem Dashboard:
```text
cp .env.example .env
```
Open .env and fill in your values:
```text
CDSE_CLIENT_ID=your_oauth_client_id_here
CDSE_CLIENT_SECRET=your_oauth_client_secret_here
```
2. Updating Locations (locations.json)You can add custom regions of interest in standard WGS84 degree bounds:
```text
JSON{
  "camp_titin": {
    "name": "Camp Titin",
    "country": "Jordan",
    "bbox": [35.01, 29.32, 35.09, 29.39]
  }
}
```
# Usage Guide
1. Automated Change-Detection Visuals (Method 1: CDSE) Identifies the last two distinct satellite acquisitions for every site in locations.json and downloads both True Color and Highlight-Optimized visual renders:
```Bash
python download_cdse.py
```
Outputs are saved to ./output_cdse/<location_key>/

2. Free Visual Streaming (Method 2: AWS Public COGs)Downloads the exact same targets via AWS STAC COGs without requiring API credentials or authentication:
```Bash
python download_aws.py
```
Outputs are saved to ./output_aws/<location_key>/.

3. Collect Raw 4-Band Sentinel-2 GeoTIFFDownloads the native surface reflectance bands (Red, Green, Blue, NIR) cropped for a target area and stacks them into a 16-bit GeoTIFF:
```Bash
python collect_4bands.py
```
Outputs a 4-band GeoTIFF: AlUdeid_4bands_<date>.tif.

4. Run AI Super-Resolution (10 m $\rightarrow$ 2.5 m)Runs the SEN2SR model across the stacked raster using sliding-window tiling and Hann blending:
```Bash
python run_sen2sr.py
```
Outputs generated in ./sen2sr_output/
