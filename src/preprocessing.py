"""
Preprocessing module for water body detection.
- Reads images from data/raw_images/
- Resizes them for uniformity
- Applies denoising, contrast enhancement, and sharpening
- Saves processed outputs to data/processed_images/
"""

import cv2
import os
import numpy as np

RAW_DIR = "data/raw_images"
PROCESSED_DIR = "data/processed_images"

# Ensure output folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)


def resize_image(image, width=512):
    """Resize image maintaining aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_h = int(width * aspect_ratio)
    resized = cv2.resize(image, (width, new_h))
    return resized


def enhance_contrast(image):
    """Apply CLAHE on the L-channel (for better water-land distinction)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def denoise_image(image):
    """Apply bilateral filtering to reduce noise while keeping edges."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def sharpen_image(image):
    """Slightly sharpen the image to enhance water boundaries."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_image(image_path):
    """Full preprocessing pipeline for one image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Could not read {image_path}")
        return None

    image = resize_image(image)
    image = enhance_contrast(image)
    image = denoise_image(image)
    image = sharpen_image(image)
    return image


def run_preprocessing():
    """Run preprocessing for all images in raw_images."""
    print("[INFO] Starting preprocessing...")
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(RAW_DIR, filename)
            processed = preprocess_image(path)
            if processed is not None:
                save_path = os.path.join(PROCESSED_DIR, filename)
                cv2.imwrite(save_path, processed)
                print(f"[OK] Processed and saved: {save_path}")

    print("[INFO] Preprocessing complete.")


def count_images():
    count=0
    """Run preprocessing for all images in raw_images."""
    print("[INFO] Starting preprocessing...")
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            count=count+1
    
    print("Total images found:",count)

if __name__ == "__main__":
    run_preprocessing()
    # count_images()

