import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_results(original, ndwi, mask, classified):
    """
    Display the intermediate and final results using matplotlib.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ndwi, cmap="gray")
    axes[1].set_title("NDWI Map")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Water Mask")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(classified, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Classified (Heuristic)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def save_results(original, ndwi, mask, classified, output_path_prefix: str):
    """
    Save all key results (NDWI, mask, classified overlay) to disk.
    """
    cv2.imwrite(f"{output_path_prefix}_ndwi.jpg", ndwi)
    cv2.imwrite(f"{output_path_prefix}_mask.jpg", mask)
    cv2.imwrite(f"{output_path_prefix}_classified.jpg", classified)
