import glob
import math
import os
import cv2
import mlstac
import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
from tqdm import tqdm

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# 2. Download and Load Pretrained Model
model_dir = "models/SEN2SRLite_RGBN"
os.makedirs(model_dir, exist_ok=True)

print("Fetching SEN2SR RGBN model...")
mlstac.download(
    file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
    output_dir=model_dir,
)
model = mlstac.load(model_dir).compiled_model(device=device)
model.eval()
print("Model loaded successfully.")

# 3. Locate Input GeoTIFF
input_files = glob.glob("Muwaffaq_Salti_Air Base_4bands_*.tif")
if not input_files:
    raise FileNotFoundError("Could not find 'AlUdeid_4bands_*.tif'. Run collect_4bands.py first!")

input_path = input_files[0]
print(f"Processing input: {input_path}")

output_dir = "./sen2sr_output"
os.makedirs(output_dir, exist_ok=True)

with rasterio.open(input_path) as src:
    profile = src.profile.copy()
    orig_transform = src.transform
    raw_data = src.read().astype(np.float32)  # Shape: (4, H, W)

_, orig_h, orig_w = raw_data.shape
print(f"Original size: {orig_w}x{orig_h} px @ 10m Ground Sample Distance")

# 4. Normalize Data (0-10000 DN -> 0.0-1.0)
norm_data = np.nan_to_num(raw_data / 10000.0, nan=0.0, posinf=0.0, neginf=0.0)

# 5. Fixed-Size Sliding Window Parameters
LR_PATCH = 128       # Model requires strictly 128x128 input
HR_PATCH = 512       # 4x scale -> 512x512 output
LR_STRIDE = 96       # 32px overlap for seamless blending
HR_STRIDE = LR_STRIDE * 4

# Calculate required padding so all patches are exactly 128x128
num_y = max(1, math.ceil((orig_h - LR_PATCH) / LR_STRIDE) + 1)
num_x = max(1, math.ceil((orig_w - LR_PATCH) / LR_STRIDE) + 1)

pad_h = (num_y - 1) * LR_STRIDE + LR_PATCH - orig_h
pad_w = (num_x - 1) * LR_STRIDE + LR_PATCH - orig_w

# Pad using reflection to avoid edge distortion
padded_input = np.pad(norm_data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

# Create 2D Hann blending window
w1d = np.hanning(HR_PATCH)
blend_window = np.outer(w1d, w1d).astype(np.float32)
blend_window = np.maximum(blend_window, 1e-4)  # Prevent zero-weights at boundaries

# Buffers for High-Resolution Reconstruction
hr_padded_h = padded_input.shape[1] * 4
hr_padded_w = padded_input.shape[2] * 4
hr_output = np.zeros((4, hr_padded_h, hr_padded_w), dtype=np.float32)
weight_map = np.zeros((hr_padded_h, hr_padded_w), dtype=np.float32)

# 6. Run Strict 128x128 Tiled Inference
total_patches = num_y * num_x
print(f"Running super-resolution across {total_patches} patches ({num_y}x{num_x})...")

with torch.no_grad():
    with tqdm(total=total_patches, desc="Enhancing Patches") as pbar:
        for iy in range(num_y):
            y_lr = iy * LR_STRIDE
            y_hr = y_lr * 4

            for ix in range(num_x):
                x_lr = ix * LR_STRIDE
                x_hr = x_lr * 4

                # Extract strictly 128x128 patch
                patch_lr = padded_input[:, y_lr : y_lr + LR_PATCH, x_lr : x_lr + LR_PATCH]
                tensor_lr = torch.from_numpy(patch_lr).unsqueeze(0).to(device)

                # Predict 512x512 super-resolved patch
                patch_hr = model(tensor_lr).squeeze(0).cpu().numpy()

                # Accumulate with blending window
                for c in range(4):
                    hr_output[c, y_hr : y_hr + HR_PATCH, x_hr : x_hr + HR_PATCH] += (
                        patch_hr[c] * blend_window
                    )
                weight_map[y_hr : y_hr + HR_PATCH, x_hr : x_hr + HR_PATCH] += blend_window

                pbar.update(1)

# 7. Normalize Overlapping Blends and Crop to Exact Dimensions
hr_output /= np.expand_dims(weight_map, axis=0)

# Crop padding out: final size is exactly (orig_h * 4, orig_w * 4)
final_h = orig_h * 4
final_w = orig_w * 4
final_sr = hr_output[:, :final_h, :final_w]

# 8. Save Enhanced 2.5m GeoTIFF
sr_bands_uint16 = np.clip(final_sr * 10000.0, 0, 10000).astype(np.uint16)
scale_factor = 4
new_transform = orig_transform * Affine.scale(1 / scale_factor, 1 / scale_factor)

profile.update({
    "height": final_h,
    "width": final_w,
    "transform": new_transform,
    "dtype": "uint16",
})

output_geotiff = os.path.join(output_dir, "AlUdeid_SEN2SR_2.5m_4bands.tif")
with rasterio.open(output_geotiff, "w", **profile) as dst:
    dst.write(sr_bands_uint16)

print(f"\n[SUCCESS] Saved GeoTIFF: {output_geotiff}")
print(f"Dimensions: {final_w}x{final_h} px (2.5m Ground Sample Distance)")

# 9. Export Highlight-Optimized Visual PNG
# Channels: Red=0, Green=1, Blue=2
rgb = final_sr[0:3, :, :].transpose(1, 2, 0)
# Tone curve: cbrt(0.6 * reflectance)
rgb_opt = np.cbrt(0.6 * np.clip(rgb, 0, None))
rgb_uint8 = np.clip(rgb_opt * 255.0, 0, 255).astype(np.uint8)
bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

output_png = os.path.join(output_dir, "Muwaffaq_Salti_Air Base_4bands_2026-09-03_Optimized.png")
cv2.imwrite(output_png, bgr_uint8)
print(f"[SUCCESS] Saved Visual Preview: {output_png}")