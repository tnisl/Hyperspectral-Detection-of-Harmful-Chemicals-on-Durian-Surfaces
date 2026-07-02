# Hyperspectral Detection of Harmful Chemicals on Durian Surfaces

This project develops a rapid, non-destructive screening solution to detect harmful chemical residues on durian surfaces using Hyperspectral Imaging (HSI) combined with machine learning techniques.

## 📌 System Overview

The system is systematically designed into two main operational phases based on Computational Thinking (CAT):
1. **System Preparation Phase:** Processes the labeled dataset, applies a **Genetic Algorithm (GA)** for optimal spectral band selection, and fits a **Multi-label K-Nearest Neighbors (KNN)** classifier.
2. **Durian Inspection Phase:** Executed for each incoming sample in real-time. It extracts and refines the spectrum, restricts the features to the selected bands, and predicts the presence of target chemicals.

The system simultaneously detects five classes of harmful chemicals: *Captan, Captafol, Methyl Parathion, Carbofuran, and Lindane*.

## 🛠️ Pipeline & Algorithm Specifications

* **Spectral Extraction:** Isolates the fruit region using intensity thresholding and averages the masked pixels. It concatenates the mean spectra from the VNIR system (400–1000 nm, 279 bands) and SWIR system (1000–2500 nm, 233 bands) into a unified **512-dimensional vector**.
* **Normalization:** Applies per-band **Z-score Standardization** estimated from the training set.
* **Smoothing:** Utilizes a **Savitzky-Golay filter** to flatten high-frequency sensor noise while preserving crucial absorption peaks.
* **Band Selection:** A Genetic Algorithm optimizes a custom fitness function combining the $F1_{macro}$ score with a penalty proportional to the number of active bands.
* **Prediction:** A Multi-label KNN classifier operates under a binary-relevance scheme to predict independent chemical states concurrently.

## 📁 File Structure & Descriptions

*(Please update this list based on your actual repository layout)*

* `main.py` / `app.py`: The main entry point to run the demonstration or Streamlit dashboard.
* `src/preprocessing.py`: Contains functions for spectral extraction, Z-score normalization, and Savitzky-Golay filtering.
* `src/genetic_algorithm.py`: Implements the Genetic Algorithm for optimal spectral band selection.
* `src/model.py`: Defines the Multi-label KNN classifier and evaluation metrics.
* `data/`: Directory to store raw hyperspectral images (.hdr/.raw) and dataset annotations.
* `requirements.txt`: Lists all Python dependencies needed to execute the project.

## 🚀 Installation & How to Run

### 1. Setup Environment
Clone the repository and install the required dependencies:
```bash
# Clone the repository
git clone [https://github.com/tnisl/Hyperspectral-Detection-of-Harmful-Chemicals-on-Durian-Surfaces.git](https://github.com/tnisl/Hyperspectral-Detection-of-Harmful-Chemicals-on-Durian-Surfaces.git)
cd Hyperspectral-Detection-of-Harmful-Chemicals-on-Durian-Surfaces
```
### 2. Execution
For the Gradio app, run this file.
```bash
python app_center_samples_clean.py
```

[How to Use the Demo (Video Guide)](https://youtu.be/i8BxLMI_YGQ)
