---
title: Coral Reef Bleaching Predictor
emoji: 🪸
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🪸 Coral Reef Bleaching Risk Predictor

A machine learning web application that predicts coral reef bleaching risk using four trained classification models. Built as part of an ML assignment at SLIIT.

## Overview

Coral bleaching occurs when ocean temperatures rise above normal levels, causing corals to expel their symbiotic algae and turn white. This app takes real-world oceanographic measurements as input and predicts whether a reef site is at risk of bleaching.

**Dataset:** Global Coral Bleaching Database 1980–2020 ([BCO-DMO](https://doi.org/10.26008/1912/bco-dmo.773466.2))

---

## Models

| Model | Test Accuracy | F1 Score | ROC-AUC | Scaling |
|---|---|---|---|---|
| Logistic Regression | 67.39% | 0.6835 | 0.7338 | Yes |
| Random Forest | 90.45% | 0.9080 | 0.9704 | No |
| XGBoost | 90.21% | 0.9049 | 0.9683 | No |
| SVM (RBF Kernel) | 87.89% | 0.8817 | 0.9285 | Yes |

All four models vote on the final prediction. The consensus is determined by the number of models predicting bleaching:

- **4/4 votes** → CRITICAL
- **3/4 votes** → HIGH RISK
- **2/4 votes** → MODERATE
- **1/4 votes** → LOW RISK
- **0/4 votes** → SAFE

---

## Input Features

| Feature | Description |
|---|---|
| `ClimSST` | Climatological Sea Surface Temperature (°C) |
| `Temperature_Mean` | Mean SST (°C) |
| `Temperature_Minimum` | Minimum SST (°C) |
| `Temperature_Maximum` | Maximum SST (°C) |
| `SSTA` | Sea Surface Temperature Anomaly (°C) |
| `SSTA_DHW` | SST Anomaly Degree Heating Weeks |
| `TSA` | Thermal Stress Anomaly (°C) |
| `TSA_DHW` | Thermal Stress Anomaly DHW |
| `TSA_DHW_Frequency` | Frequency of TSA DHW events |
| `Windspeed` | Wind speed (m/s) |
| `SSTA_Frequency` | Frequency of positive SSTA |
| `SSTA_Frequency_Standard_Deviation` | Std dev of SSTA frequency |
| `Turbidity_ct` | Turbidity count |
| `Turbidity` | Water turbidity |
| `Cyclone_Frequency` | Frequency of cyclone events |
| `Distance` | Distance to nearest land (km) |
| `Depth` | Reef depth (m) |
| `Latitude_Degrees` | Latitude |
| `Longitude_Degrees` | Longitude |
| `Date_Year` | Year of observation |

**Key bleaching risk indicators:**
- `TSA_DHW > 8` → severe bleaching expected
- `Temperature_Mean > 30°C` + `SSTA > 1.5` → high risk
- Low turbidity + high DHW → maximum risk

---

## App Features

### Single Prediction
Adjust ocean condition sliders to get an instant prediction from all 4 models with a probability bar chart and consensus gauge.

### Sample Reef Sites
Load pre-configured data from 10 real-world reef locations including the Great Barrier Reef, Maldives, Red Sea, and Caribbean.

### Batch Prediction
Run all 10 sample sites through all 4 models simultaneously and view results as a probability heatmap.

### Model Info
View loaded model performance metrics (accuracy, F1, ROC-AUC) and feature counts.

---

## Project Structure

```
├── app.py                    # Gradio web application
├── requirements.txt          # Python dependencies
├── models/
│   ├── lr/                   # Logistic Regression artifacts
│   │   ├── lr_model.pkl
│   │   ├── lr_scaler.pkl
│   │   ├── lr_features.pkl
│   │   └── lr_metadata.json
│   ├── rf/                   # Random Forest artifacts
│   ├── xgb/                  # XGBoost artifacts
│   └── svm/                  # SVM artifacts
└── notebooks/                # Training notebooks
```

---

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

---

## Tech Stack

- **ML:** scikit-learn, XGBoost
- **UI:** Gradio
- **Data:** pandas, numpy
- **Visualization:** matplotlib
