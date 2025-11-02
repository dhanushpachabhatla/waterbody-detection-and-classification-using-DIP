#Convert from RGB → other spaces

import cv2
import numpy as np

def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSV color space.
    Useful for color-based thresholding (e.g., hue-based water detection).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

def compute_ndwi(image: np.ndarray) -> np.ndarray:
    """
    Compute NDWI (Normalized Difference Water Index).
    NDWI = (Green - NIR) / (Green + NIR)
    Since standard RGB images lack NIR, we approximate NIR using the Red channel.
    This works decently for water/non-water separation.
    """
    # Split channels
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Approximate NDWI
    ndwi = (g - r) / (g + r + 1e-5)
    
    # Normalize to 0–255
    ndwi_normalized = cv2.normalize(ndwi, None, 0, 255, cv2.NORM_MINMAX)
    ndwi_normalized = ndwi_normalized.astype(np.uint8)
    return ndwi_normalized


def compute_blue_ratio(image: np.ndarray) -> np.ndarray:
    """
    Blue ratio = B / (R + G + B)
    Normalizes blue intensity so we can threshold in a relative way.
    Returns values in 0-255 uint8.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    denom = (r + g + b) + 1e-8
    br = b / denom
    br_norm = cv2.normalize((br * 255).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    return br_norm