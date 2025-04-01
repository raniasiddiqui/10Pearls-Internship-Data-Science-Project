# 10Pearls-Internship-Data-Science-Project
## 🌍 AQI Predictor
## 🚀 Overview:
AQI Predictor is an end-to-end, serverless system that forecasts the Air Quality Index (AQI) for the next 3 days in any city. The project leverages real-time weather and pollution data, machine learning models, and automated pipelines to provide accurate predictions and insights.

## 🔥 Features:
- Real-time Data Fetching: Collects raw weather and pollution data from API.

- Feature Engineering: Computes time-based and derived features (e.g., AQI change rate).

- ML Model Training: Trains and evaluates models using Scikit-learn and deep learning frameworks like TensorFlow/PyTorch.

- Automated Pipelines: Uses CI/CD tools (Apache Airflow, GitHub Actions) to automate data collection and model training.

- Interactive Dashboard: Displays real-time and forecasted AQI using Streamlit and a backend built with Flask/FastAPI.


## 📌 Project Workflow
1️⃣ Feature Pipeline:

- Fetches real-time weather & pollutant data.

- Computes relevant features and stores them in a Feature Store (Hopsworks/Vertex AI).

- Backfills historical data for training.

2️⃣ Training Pipeline:

- Fetches historical data, trains models, and evaluates performance using RMSE, MAE, R².

- Stores the best-performing model in the Model Registry.

3️⃣ Deployment & Automation:

- CI/CD pipelines trigger hourly feature updates and daily model training.

A web app loads the trained model, generates predictions, and presents insights via an interactive dashboard.


## 🛠 Tech Stack
- Data Processing: Python, Pandas, NumPy

- Machine Learning: Scikit-learn, TensorFlow, PyTorch

- Automation: Apache Airflow, GitHub Actions

- Web App: Flask, FastAPI, Streamlit, Gradio

- Deployment: Docker, CI/CD
