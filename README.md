# 🧠 Credit Risk Analyst

This project analyzes the **German Credit Data** from the UCI Machine Learning Repository to predict whether a client represents a **good** or **bad** credit risk.  
It applies **machine learning models** and **feature engineering techniques** to build a transparent and interpretable scoring model.

---

## 📊 Project Overview

Credit risk modeling is a key task in the banking sector — it helps institutions estimate the probability of loan default and make data-driven lending decisions.  
This project explores the German Credit dataset, performs preprocessing and feature selection, and compares several classification models.

---

## 🧱 Repository Structure
```
credit-risk-analyst/
│
├── data/
│ ├── german.data
│ ├── german.data-numeric
│ ├── german.doc
│ └── Index
│
├── src/
│ ├── feature_importance_rf_model.py # Feature importance extraction using Random Forest
│ ├── GridSearchCV.py # Model optimization functions
│ ├── k_cross_validation.py # Cross-validation helper
│ ├── logistic_regression.py # Logistic regression model pipeline
│ ├── ohe_and_scaling.py # Encoding and scaling transformations
│ └── test.py # Testing and validation scripts
│
└── index.ipynb # Main notebook with analysis and visualizations
```
