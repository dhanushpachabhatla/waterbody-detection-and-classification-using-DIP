# 🌊 Satellite Waterbody Segmentation and Classification

This project performs **automated detection and classification of water bodies** from satellite imagery using traditional image processing techniques.  
It processes RGB satellite images, computes **NDWI (Normalized Difference Water Index)** and **blue channel ratios**, performs **adaptive Otsu thresholding-based segmentation**, and applies **postprocessing** to produce clean, accurate binary masks of detected water regions.

---

## 🧠 Project Overview

The pipeline follows these stages:

1. **Preprocessing** – Reads and resizes input satellite images.
2. **Color Conversion** – Converts RGB → HSV and computes NDWI.
3. **Segmentation** – Detects water regions using NDWI and blue ratio via Otsu thresholding.
4. **Postprocessing** – Fills holes, smooths edges, and removes small noisy regions.
5. **Classification** – Categorizes detected water regions based on color/size features.
6. **Visualization** – Displays and saves final masks and overlays for review.

---

## 🗂️ Project Structure

<img width="430" height="449" alt="image" src="https://github.com/user-attachments/assets/d42e9310-4361-45e9-8fe0-c5d6f8e3f986" />


---

## ⚙️ Setup Instructions

Follow these steps to run the project locally:

### 1️. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages listed in requirements.txt:
pip install -r requirements.txt
```

### 3. Prepare Input Data
can download data from kaggle datasets- `https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies/data`
```bash
data/raw_images/
```

### 4. Run the Pipeline
```bash
#Execute the main script to process all images:
python -m src.main
```

### 5. Output Results
```bash
data/classified/
```
Each image will have:
_mask.png → Cleaned binary mask
_classified.png → Final visualization with detected water regions
_ndwi.png → NDWI map visualization
