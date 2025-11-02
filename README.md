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



### first draft results - 
its working nice but still the classification is going wrong sometimes as it is based on heuristics....need to work on it.

<img width="1919" height="656" alt="Screenshot 2025-11-02 121041" src="https://github.com/user-attachments/assets/f87e57e0-cea5-439c-b02c-3aaf4194d534" />
<img width="1866" height="738" alt="Screenshot 2025-11-02 120901" src="https://github.com/user-attachments/assets/09d06b19-9453-48d6-b897-cec18a1adbb8" />
<img width="1875" height="726" alt="Screenshot 2025-11-02 121017" src="https://github.com/user-attachments/assets/d0788f37-31f3-4d5f-8a39-0db28b477269" />


