# 🏢 Pune Real Estate Price Estimator


A modern, interactive Deep Learning web application that estimates real estate prices in Pune, Maharashtra. The app uses a neural network built with TensorFlow/Keras and is deployed via Streamlit on Render.

## 🚀 Live Application
Check out the live web app here: **[Pune House DL Model](https://pune-house-dl-model-3.onrender.com/)**

## 📖 Project Overview
Predicting property prices requires analyzing various interconnected features. This project takes inputs such as Total Square Footage, BHK, Bathrooms, Balconies, Site Location, and Area Type, and processes them through a trained Deep Learning model to estimate the property's market value. 

### Key Features
* **181-Dimensional Feature Encoding:** Automatically handles one-hot encoding for specific Pune localities (e.g., Baner, Kothrud, Shivaji Nagar) and property Area Types.
* **Dual Scaling:** Utilizes Scikit-Learn `StandardScaler` for both the input features (X) and the target variable (y) to ensure high neural network accuracy.
* **Deep Learning Architecture:** Employs a Keras Sequential model utilizing Dense layers, Batch Normalization, and Dropout for robust regression.
* **Sleek UI:** A responsive, user-friendly frontend built purely in Python using Streamlit.

## 📂 Project Structure
Ensure all these files are present in the root directory:

```text
├── app.py                      # Main Streamlit application script
├── requirements.txt            # Python dependencies (optimized for cloud deployment)
├── house_price_model.keras     # Trained TensorFlow/Keras neural network
├── scaler_X.pkl                # Fitted StandardScaler for input features
├── scaler_y.pkl                # Fitted StandardScaler for the target price
├── model_columns.pkl           # Saved list of 181 column names for exact matching
└── README.md                   # Project documentation
