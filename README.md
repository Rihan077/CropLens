---
title: CropLens
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# 🌾 CropLens — Agricultural Risk Intelligence System

> District-level crop risk assessment for Maharashtra using Machine Learning, Explainable AI, and Live Weather Data.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/rihan077/CropLens)
[![GitHub](https://img.shields.io/badge/GitHub-CropLens-green)](https://github.com/Rihan077/CropLens)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black)](https://flask.palletsprojects.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-orange)](https://lightgbm.readthedocs.io)

---

## 🔗 Live Demo

**[Try CropLens Live](https://huggingface.co/spaces/rihan077/CropLens)**

---

## 📌 Problem Statement

In Maharashtra, farmers, cooperative bank officers, and insurance companies make critical decisions about crop loans, planting choices, and premium pricing mostly based on experience and guesswork.

CropLens solves this by providing:
- Data-driven crop risk assessment at district level
- Explainable AI reasoning for every prediction
- Live weather integration for current season assessment
- Actionable recommendations for farmers, banks, and insurers

---

## 🎯 Real World Users

| User | Problem | CropLens Solution |
|------|---------|-------------------|
| Bank Officer | No data to assess crop loan risk | Risk score + SHAP explanation per prediction |
| Farmer | No guidance on crop selection | District-wise risk comparison across crops |
| Insurance Company | Blind premium pricing | Data-driven risk scores for all 36 districts |

---

## 🏗️ System Architecture

Raw Data Sources → Data Pipeline → Feature Engineering → ML Model → SHAP Explainability → Flask REST API → Frontend Dashboard → Docker → Hugging Face Spaces

---

## 📊 Dataset

| Dataset | Source | Records |
|---------|--------|---------|
| District-wise Crop Production | Kaggle / data.gov.in | 345,407 filtered to 8,288 Maharashtra |
| Rainfall Data 1901-2017 | IMD / Kaggle | 4,187 subdivision records |
| Soil Type | ICAR / Manual | 36 districts |
| MSP Prices | CACP India | 19 years x 11 crops |

Maharashtra Focus:
- 36 Districts
- 11 Major Crops
- Years: 1997-98 to 2019-20

---

## ⚙️ Feature Engineering

| Feature | Description | Why Important |
|---------|-------------|---------------|
| yield_lag_1/2/3 | Previous 1/2/3 year yield | Past yield predicts future |
| yield_3yr_avg | Rolling 3-year average | Smooths single-year anomalies |
| yield_trend | 5-year yield slope | Is yield improving or declining |
| yield_volatility | 3-year std deviation | Unpredictable districts = higher risk |
| rainfall_deviation_pct | % deviation from 30yr normal | Key drought indicator |
| drought_streak | Consecutive drought years | Compound risk factor |
| compatibility_score | Soil-crop suitability 0-1 | Custom domain knowledge feature |
| msp_growth_rate | Year-over-year MSP change | Economic stress indicator |

---

## 🤖 Model Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 43.73% | 0.4194 |
| Random Forest | 61.10% | 0.5909 |
| XGBoost | 59.95% | 0.5752 |
| LightGBM Final | 61.58% | 0.5994 |

Final Model: LightGBM tuned with Optuna 50 trials

Cross Validation 5-Fold:
- Mean Macro F1: 0.6022
- Std Dev: 0.0080 very stable

Per Class Performance:
- High Risk F1: 0.69
- Low Risk F1: 0.66
- Medium Risk F1: 0.42

---

## 🔍 SHAP Explainability

Every prediction includes a human-readable explanation showing which features increased or reduced the risk level with their actual values and impact direction.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/predict | Get risk prediction + SHAP |
| GET | /api/districts | List all 36 districts |
| GET | /api/crops | List all 11 crops |
| GET | /api/seasons | List available seasons |
| GET | /api/history/district/crop | Historical yield data |
| GET | /api/map-data | Risk data for all districts |
| GET | /api/weather/district | Live rainfall via Open-Meteo |
| POST | /api/generate-pdf | Download PDF risk report |
| GET | /api/health | API health check |

---

## 🗺️ Features

### 1. Risk Prediction Dashboard
- Select district, crop, season
- Live rainfall auto-fetched from Open-Meteo API
- Risk level High / Medium / Low with confidence score
- SHAP explanation with reasons

### 2. Maharashtra Risk Map
- All 36 districts color-coded by risk
- District-wise rainfall comparison chart
- Sortable risk summary table

### 3. PDF Report Download
- Professional PDF with full risk assessment
- Input details, risk level, statistics
- SHAP analysis table
- Recommendation for farmers and banks

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | LightGBM, XGBoost, Scikit-learn |
| Explainability | SHAP |
| Hyperparameter Tuning | Optuna |
| Backend | Flask, Flask-CORS |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| PDF Generation | ReportLab |
| Weather API | Open-Meteo free no key needed |
| Deployment | Docker, Hugging Face Spaces |
| Version Control | Git, GitHub |

---

## 🚀 How To Run Locally

Prerequisites: Python 3.11, Git

Clone repository:
git clone https://github.com/Rihan077/CropLens.git
cd CropLens

Create virtual environment:
python -m venv venv
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run Flask app:
cd api
python app.py

Open browser at http://127.0.0.1:7860

---

## 📁 Project Structure

- api/app.py — Flask backend and all routes
- data/processed/ — Cleaned datasets
- data/external/ — Soil and MSP reference files
- frontend/templates/ — HTML pages
- frontend/static/ — CSS and JS files
- models/ — Trained model and encoders
- notebooks/ — EDA and training notebooks
- Dockerfile — Container configuration
- requirements.txt — Python dependencies

---

## 🔮 Future Scope

- Satellite imagery NDVI from Copernicus Sentinel
- Weather forecast integration 7-day prediction
- Multi-state support beyond Maharashtra
- User authentication for bank officers
- Mobile responsive PWA

---

## 👨‍💻 Author

**Rihan Bagwan**

B.Tech CSE AI and Data Science — Sanjay Ghodawat University Kolhapur

IIT Roorkee IntelliPath AI and Data Science Programme

- LinkedIn: linkedin.com/in/rihan-bagwan-726478277
- GitHub: github.com/Rihan077
- Email: rihanbagwan46@gmail.com

---

## 📄 License

MIT License — feel free to use and build upon this project.

---

Built with love for Maharashtra's farming community