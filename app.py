from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests_cache
import openmeteo_requests
import joblib
import confluent_kafka
import retry_requests
from fetch_data import fetch_openmeteo_data
from compute_targets import compute_pm25_aqi, add_lag_features, compute_features_targets
from train_features import train_evaluate_models
from store_features import store_in_feature_store, store_model_in_registry
from predict_futures import predict_future

def main():
    print("Fetching data...")
    df = fetch_openmeteo_data(start_date="2020-01-01")

    print("Raw Open-Meteo data:")
    print(df.head())  # show first few rows

    print("Computing features...")
    features_df, targets, full_df = compute_features_targets(df)

    print("Adding hour/day/month to features for Feature Store compatibility...")
    features_df['hour'] = full_df['hour']
    features_df['day'] = full_df['day']
    features_df['month'] = full_df['month']

    print("Training models...")
    results = train_evaluate_models(features_df, targets)

    print("Model performance:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, val in metrics.items():
            if metric != 'model':
                print(f"  {metric}: {val:.2f}")

    print("Storing features in Hopsworks...")
    store_in_feature_store(features_df, targets)

    print("Registering best model...")
    project = hopsworks.login(api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS')
    best_model, best_model_name = store_model_in_registry(results, project)
    print(f"Best model registered: {best_model_name}")

    print("Predicting future AQI for next 3 days...")
    future_df = fetch_openmeteo_data(start_date=(datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"),
                                     end_date=(datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d"))
    future_features, _, _ = compute_features_targets(future_df)
    future_preds = predict_future(best_model, future_features)

    print("\nForecasted AQI for next 3 days:")
    for ts, pred in zip(future_df['date'], future_preds):
        print(f"{ts.strftime('%Y-%m-%d %H:%M')} -> AQI (PM2.5): {pred:.2f}")

if __name__ == "__main__":
    main()

