import cv2
import numpy as np

def otsu_threshold(image_uint8):
    """Return binary mask using Otsu thresholding on a uint8 image."""
    blur = cv2.GaussianBlur(image_uint8, (5,5), 0)
    th_val, th_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th_mask, th_val

def segment_water_ndwi_adaptive(ndwi_image: np.ndarray):
    """
    NDWI is expected as uint8 (0-255). Use Otsu to compute threshold.
    Returns binary mask (0/255).
    """
    mask, th = otsu_threshold(ndwi_image)
    # NDWI positive values are water — sometimes Otsu picks a threshold that's too low,
    # so we can optionally raise threshold slightly:
    # mask = cv2.threshold(ndwi_image, min(th+10, 255), 255, cv2.THRESH_BINARY)[1]
    return mask

def segment_water_blue_ratio_adaptive(blue_ratio_uint8: np.ndarray):
    """Segment using Otsu on blue ratio map."""
    mask, th = otsu_threshold(blue_ratio_uint8)
    return mask

def segment_combined(image, ndwi_uint8, blue_ratio_uint8):
    """
    Combine NDWI and blue-ratio masks using OR, but also allow NDWI strong pixels only.
    """
    m1 = segment_water_ndwi_adaptive(ndwi_uint8)
    m2 = segment_water_blue_ratio_adaptive(blue_ratio_uint8)

    # Combine and do a little morphological opening to drop tiny specks early.
    combined = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined
