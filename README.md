# Surabaya Property Valuation & Segmentation System

## Project Overview
Surabaya Property Valuation & Segmentation System is a data-driven software solution designed to address market inefficiencies in the Surabaya real estate sector. This project leverages advanced Data Mining techniques to provide objective property valuation and automated market segmentation. The system aims to assist real estate agents, investors, and stakeholders in making informed, data-backed decisions.

The core functionality is built upon two machine learning paradigms:
1.  **Supervised Learning (Market Valuation):** Utilizes a Random Forest Regressor to estimate fair market prices based on 12 distinct property attributes (physical dimensions, location, and legality).
2.  **Unsupervised Learning (Asset Profiling):** Utilizes K-Means Clustering to automatically categorize properties into distinct market segments (Entry-Level, Mid-Tier, Luxury) based on physical characteristics.

## System Architecture
The project is structured into three main components:
* **Dataset:** Raw data storage.
* **Research Lab:** Scripts for statistical analysis, model evaluation, and reporting (RMSE & Silhouette Score calculation).
* **Deployment:** A production-ready Flask web application serving the models via a graphical user interface.

## Technology Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, K-Means, Pipeline, ColumnTransformer)
* **Model Persistence:** Joblib
* **Backend Framework:** Flask
* **Frontend Interface:** HTML5, Tailwind CSS

## Project Directory Structure
Ensure your project folder follows this specific hierarchy for the scripts to function correctly:

/root
│
├── README.md                           # Project documentation
├── requirements.txt                    # List of Python dependencies
│
├── DATASET/
│   └── Combined_Datalist_v1.1.csv      # Primary dataset (Source)
│
├── Analisis/                       # Analytics & Evaluation Module
│   ├── analisis.py               # Script for statistical analysis & CSV export
│   ├── Output_Supervised_Prediksi.csv  # Generated report: Actual vs Predicted Price
│   └── Output_Unsupervised_Cluster.csv # Generated report: Clustered Data
│
└── DEPLOYMENT/                      # Production Module (Web App)
    ├── train_pipeline.py               # Script to train and save models
    ├── app.py                          # Main application server
    ├── models/                         # Folder for trained .joblib models (Auto-generated)
    └── templates/
        └── index.html                  # Web interface template

```

## Installation & Setup Guide

### 1. Prerequisites

Ensure Python is installed on your system. It is recommended to use a virtual environment.

### 2. Install Dependencies

Open your terminal or command prompt in the root directory and execute the following command:

```bash
pip install -r requirements.txt

```

*(Note: Create a requirements.txt file with contents: flask, pandas, scikit-learn, joblib, numpy)*

---

## Usage Instructions

This project requires a two-step process: Model Training and Server Execution.

### Step 1: Initialize and Train Models

Before running the web application, you must generate the model files and metadata.

1. Navigate to the Deployment directory:
```bash
cd "DEPLOYMENT"

```


2. Run the training pipeline:
```bash
python train_pipeline.py

```


*Output: This will create a `models/` directory containing `model_supervised.joblib`, `model_unsupervised.joblib`, and `meta_options.joblib`.*

### Step 2: Launch the Web Application

Once the models are trained, you can start the Flask server.

1. Ensure you are still in the `"DEPLOYMENT"` directory.
2. Run the application script:
```bash
python app.py

```


3. The application will start locally. Open your web browser and access:
`http://127.0.0.1:5000`

---

## Research & Evaluation (Optional)

To generate statistical reports for academic purposes (Chapter 4 & 5 of the thesis):

1. Navigate to the Research Lab directory:
```bash
cd "../ANALISIS"

```


2. Run the analysis script:
```bash
python analisis.py

```


3. The script will output the **RMSE** and **Silhouette Score** in the terminal and generate detailed CSV reports in the folder.

## Disclaimer

This software is developed for educational and research purposes as part of the Master of Computer Science curriculum (Group 7). The valuation estimates provided by the system are based on historical data patterns and should be used as a reference tool, not as a certified appraisal.

---

© 2025 Group 7 - Data Mining Project. All Rights Reserved.

```

### Tips Tambahan untuk Anda:
Jangan lupa membuat file `requirements.txt` di folder root (sejajar dengan README ini) agar instruksi instalasi di atas valid. Isi `requirements.txt`-nya cukup:

```text
flask
pandas
numpy
scikit-learn
joblib

```