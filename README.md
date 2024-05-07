# Financial Predictive Modeling Project

## Introduction

This project aimed to explore various machine learning models for predicting financial outcomes using a merged dataset comprising trade data for India and daily stock market data for ONGC. The objective was to evaluate the effectiveness of different models in predicting target classes based on the provided features.

## Data Description

The dataset includes two main components:
- **India Trade Data (`india_df`)**: Data related to trade activities in India.
- **ONGC Daily Stock Market Data (`ongc_daily_df`)**: Daily stock market data specifically related to ONGC.

Features include trade value, year, date, stock market indicators (close, high, low, open, volume), and binary target classes indicating certain financial events or outcomes.

## Data Preparation
Data frames were cleaned, preprocessed, and merged based on common attributes. Missing values were handled, and outliers were addressed if present.

## Exploratory Data Analysis (EDA)
Summary statistics and visualizations were generated to understand the distribution and relationships within the data. Insights were derived regarding the behavior of financial indicators and their potential impact on target classes.

## Modeling Approach
Several machine learning models were employed, including:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

Each model was trained and evaluated using appropriate performance metrics.

## Model Evaluation
- **Linear Regression**:
  - Mean Squared Error: 2.164
  - Coefficient of Determination (R-squared): 0.999

- **Logistic Regression (Classification)**:
  - Accuracy: 0.67
  - Precision, Recall, and F1-score: Balanced performance, but limited precision for class 0.

- **Decision Tree**:
  - Accuracy: 0.90
  - Precision, Recall, and F1-score: High performance with balanced precision and recall for both classes.

- **Random Forest**:
  - Accuracy: 0.91
  - Precision, Recall, and F1-score: High performance with improved precision for class 0 compared to Logistic Regression.

- **XGBoost**:
  - Accuracy: 0.84
  - Precision, Recall, and F1-score: Balanced performance, slightly lower than Random Forest.

- **Support Vector Machine (SVM)**:
  - Accuracy: 0.67
  - Precision, Recall, and F1-score: Imbalanced performance, with high recall but low precision for class 0.

- **SVM with Pipeline (Undersampling and Oversampling)**:
  - Accuracy: 0.51
  - Precision, Recall, and F1-score: Improved precision for class 0 but lower overall accuracy.

## Conclusion
The analysis demonstrates the effectiveness of ensemble models such as Random Forest and Decision Tree in predicting financial outcomes. Logistic Regression and SVM show mixed performance, highlighting the importance of choosing appropriate models for specific tasks. Further fine-tuning and feature engineering could potentially enhance the performance of certain models.

## Possible Applications
The predictive models developed in this project have various potential applications in the financial domain and beyond, including:
- Financial Forecasting
- Risk Management
- Algorithmic Trading
- Market Sentiment Analysis
- Credit Risk Assessment
- Fraud Detection
- Supply Chain Optimization
- Healthcare Analytics
- Customer Churn Prediction
