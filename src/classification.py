import cv2
import numpy as np

def classify_water_bodies(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Classify detected water regions into:
    - River
    - Pond
    - Lake
    - Sea
    using geometric heuristics (aspect ratio, circularity, area ratio).

    Returns: color-coded labeled overlay image.
    """
    output_overlay = original_image.copy()
    h, w = original_image.shape[:2]
    img_area = h * w

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore very small blobs
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / bh if bh != 0 else 0
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
        area_ratio = area / img_area  # normalize by total image size

        # --- Heuristic rules (scale-invariant) ---
        if aspect_ratio > 3 and area_ratio > 0.0008:
            label = "River"
            color = (255, 100, 100)  # Light red
        elif circularity > 0.72 and area_ratio < 0.003:
            label = "Pond"
            color = (0, 255, 255)  # Yellow
        elif circularity > 0.6 and area_ratio >= 0.003:
            label = "Lake"
            color = (0, 255, 0)  # Green
        else:
            label = "Sea"
            color = (255, 0, 0)  # Blue

        # Draw contour and label
        cv2.drawContours(output_overlay, [cnt], -1, color, 2)
        cv2.putText(output_overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output_overlay
