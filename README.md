# 🛒 Amazon ML Challenge 2025 - Smart Product Pricing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning solution for predicting product prices based on catalog content and product images.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)

---

## 🎯 Overview

This project develops an ML solution to predict optimal product prices for e-commerce products based on:
- **Text Data**: Product titles, descriptions, bullet points
- **Quantity Information**: Item pack quantity (IPQ), units
- **Image Data**: Product images (optional enhancement)

### Challenge Details

| Metric | Value |
|--------|-------|
| Training Samples | 75,000 |
| Test Samples | 75,000 |
| Evaluation Metric | SMAPE |
| Model Constraint | ≤8B parameters, MIT/Apache 2.0 |

---

## 📁 Project Structure

```
Amazon-ML-Challenge/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── Documentation_template.md  # Submission documentation template
│
├── dataset/                   # Data files
│   ├── train.csv             # Training data (75K samples)
│   ├── test.csv              # Test data (75K samples)
│   ├── sample_test.csv       # Sample test file
│   └── sample_test_out.csv   # Sample output format
│
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── data_preprocessing.py # Text cleaning & preprocessing
│   ├── feature_extraction.py # TF-IDF & feature engineering
│   ├── model.py              # ML models (RF, XGB, LightGBM)
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Prediction pipeline
│   └── utils.py              # Image download utilities
│
├── notebooks/                 # Jupyter notebooks
│   └── complete_analysis.ipynb # Full EDA & training notebook
│
├── models/                    # Saved models
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│   ├── scaler.pkl            # Feature scaler
│   └── price_model.pkl       # Trained model
│
├── outputs/                   # Prediction outputs
│   └── test_out.csv          # Final predictions
│
└── images/                    # Downloaded product images
```

---

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Using Main Script

```bash
# Quick training on sample data
python main.py train --quick

# Generate predictions on sample test
python main.py predict --sample
```

### Option 2: Using Individual Scripts

```bash
# Train model
python src/train.py --quick

# Generate predictions
python src/predict.py --sample
```

### Option 3: Using Jupyter Notebook

Open `notebooks/complete_analysis.ipynb` for interactive analysis.

---

## 📖 Usage

### Training

```bash
# Quick training (sample data)
python main.py train --quick

# Train specific model
python main.py train --model rf          # Random Forest
python main.py train --model xgb         # XGBoost
python main.py train --model lgbm        # LightGBM
python main.py train --model ensemble    # Ensemble (default)

# Compare all models
python main.py train --compare

# Full training on train.csv
python main.py train --train-path dataset/train.csv --model ensemble
```

### Prediction

```bash
# Predict on sample test
python main.py predict --sample

# Predict on full test set
python main.py predict --test-path dataset/test.csv --output-path outputs/test_out.csv
```

---

## 🏗️ Model Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   catalog_content ──► TextPreprocessor ──► Clean Text               │
│         │                    │                                       │
│         │                    ├──► Item Name                         │
│         │                    ├──► Description                       │
│         │                    ├──► Bullet Points                     │
│         │                    ├──► Quantity (Value, Unit)            │
│         │                    └──► Categorical Features              │
│         │                                                            │
│         ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   FEATURE ENGINEERING                        │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │  TF-IDF Vectorizer (5000 features, n-gram: 1-2)             │   │
│   │  Numeric Features (quantity, pack_size, certifications)      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    ENSEMBLE MODEL                            │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │  • Random Forest (200 trees, max_depth=20)                  │   │
│   │  • XGBoost (300 trees, learning_rate=0.1)                   │   │
│   │  • LightGBM (300 trees, num_leaves=31)                      │   │
│   │  • Ridge Regression (baseline)                               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│      Predicted Price ($)                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Text Features Extracted

| Feature | Description |
|---------|-------------|
| `item_name` | Product title |
| `description` | Product description |
| `quantity_value` | Numeric quantity |
| `quantity_unit` | Unit of measure (oz, count, etc.) |
| `pack_size` | Pack/case size |
| `has_organic` | Organic certification |
| `has_gluten_free` | Gluten-free certification |
| `is_gift` | Gift product indicator |
| `is_bulk` | Bulk purchase indicator |

---

## 📊 Evaluation

### SMAPE (Symmetric Mean Absolute Percentage Error)

```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

- Range: 0% (perfect) to 200% (worst)
- Lower is better

### Example
```
Actual = $100, Predicted = $120
SMAPE = |100-120| / ((100+120)/2) × 100% = 18.18%
```

---

## ⚠️ Important Rules

1. **No External Price Lookup** - Prices must NOT be scraped from the internet
2. **Use Only Provided Data** - Train exclusively on provided 75K samples
3. **Output Format** - Must match sample_test_out.csv exactly
4. **All Predictions Required** - Every test sample must have a prediction

---

## 📝 Submission Checklist

- [ ] `test_out.csv` with all 75,000 predictions
- [ ] Documentation (Documentation_template.md)
- [ ] Code repository/drive link

---

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `src/config.py` | All configuration parameters |
| `src/data_preprocessing.py` | Text cleaning & feature extraction |
| `src/feature_extraction.py` | TF-IDF & numeric features |
| `src/model.py` | ML models with SMAPE evaluation |
| `src/train.py` | Complete training pipeline |
| `src/predict.py` | Prediction generation |

---

**Good luck with the challenge! 🚀**


### Tips for Success:

- Consider both textual features (catalog_content) and visual features (product images)
- Explore feature engineering techniques for text and image data
- Consider ensemble methods combining different model types
- Pay attention to outliers and data preprocessing
