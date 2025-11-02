# src/postprocessing.py

import cv2
import numpy as np

def fill_holes(binary_mask):
    """Fill holes inside binary mask using floodFill from background."""
    # Ensure mask is single-channel uint8 with 0 & 255
    mask = binary_mask.copy()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    h, w = mask.shape[:2]
    # Invert mask to floodfill background
    inv = cv2.bitwise_not(mask)
    # Pad to avoid border issues
    padded = np.pad(inv, ((1,1),(1,1)), mode='constant', constant_values=0)
    padded = padded.copy().astype(np.uint8)
    h2, w2 = padded.shape[:2]
    mask_ff = np.zeros((h2+2, w2+2), np.uint8)  # for floodFill
    cv2.floodFill(padded, mask_ff, (0,0), 255)
    padded = cv2.bitwise_not(padded)
    # remove padding
    filled = padded[1:-1,1:-1]
    # Combine filled holes with original foreground
    out = cv2.bitwise_or(mask, filled)
    return out

def refine_mask(mask, kernel_size=5, min_area_ratio=0.0005):
    """
    Full cleanup:
    - morphological opening/closing
    - fill holes
    - remove small components based on image area ratio (scale invariant)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    filled = fill_holes(closed)

    # Remove small components
    h,w = filled.shape[:2]
    image_area = h * w
    min_area = max(200, int(image_area * min_area_ratio))
    cleaned = np.zeros_like(filled)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
    return cleaned
