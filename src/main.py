# python -m src.main
import os
import cv2
import numpy as np

from src.preprocessing import preprocess_image
from src.color_conversion import compute_ndwi
from src.segmentation import segment_combined
from src.postprocessing import refine_mask
from src.classification import classify_water_bodies
from src.visualization import show_results, save_results


# -------------------------
# Configuration
# -------------------------
RAW_DIR = "data/raw_images"
RESULTS_DIR = "data/classified"
os.makedirs(RESULTS_DIR, exist_ok=True)


def process_all_images():
    """Full adaptive water body detection & classification pipeline."""
    image_files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("[WARNING] No images found in data/raw_images/")
        return

    for filename in image_files:
        print(f"\n[INFO] Processing: {filename}")

        input_path = os.path.join(RAW_DIR, filename)
        output_prefix = os.path.join(RESULTS_DIR, os.path.splitext(filename)[0])

        # --- Step 1: Preprocess ---
        image = preprocess_image(input_path)

        # --- Step 2: Color Conversion ---
        ndwi = compute_ndwi(image)

        # Convert NDWI to uint8 [0–255]
        ndwi_uint8 = cv2.normalize(ndwi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- Step 3: Additional Blue Ratio Map ---
        b, g, r = cv2.split(image)
        blue_ratio = np.clip((b.astype(np.float32) / (r.astype(np.float32) + g + 1e-5)) * 127.5, 0, 255)
        blue_ratio_uint8 = blue_ratio.astype(np.uint8)

        # --- Step 4: Segmentation (Adaptive Otsu on NDWI + Blue Ratio) ---
        combined_mask = segment_combined(image, ndwi_uint8, blue_ratio_uint8)

        # --- Step 5: Postprocessing (refine mask) ---
        final_mask = refine_mask(combined_mask)

        # --- Step 6: Classification ---
        classified = classify_water_bodies(final_mask, image)

        # --- Step 7: Visualization & Save ---
        save_results(image, ndwi_uint8, final_mask, classified, output_prefix)
        show_results(image, ndwi_uint8, final_mask, classified)

        print(f"[SAVED] Results saved for {filename}")


if __name__ == "__main__":
    print("[INFO] Starting Adaptive Water Body Classification Pipeline...")
    process_all_images()
    print("\n[INFO] Processing Complete ✅")
