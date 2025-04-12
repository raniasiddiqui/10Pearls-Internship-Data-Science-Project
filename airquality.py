import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from datetime import datetime, timedelta

import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import confluent_kafka

def fetch_openmeteo_data(start_date="2015-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 24.8608,
        "longitude": 67.0104,
        "hourly": ["pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "start_date": start_date,
        "end_date": end_date
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm2_5": hourly.Variables(0).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(1).ValuesAsNumpy(),
        "carbon_dioxide": hourly.Variables(2).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(4).ValuesAsNumpy(),
        "ozone": hourly.Variables(5).ValuesAsNumpy(),
    }

    return pd.DataFrame(data)

def compute_pm25_aqi(pm25):
    # Breakpoints based on US EPA
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    aqi = []
    for val in pm25:
        for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
            if Clow <= val <= Chigh:
                aqi_val = ((Ihigh - Ilow) / (Chigh - Clow)) * (val - Clow) + Ilow
                aqi.append(aqi_val)
                break
        else:
            # Instead of appending None, append a default value like 0
            aqi.append(0)  # Or another suitable default value
    return np.array(aqi)

def add_lag_features(df, column, lags=[1, 3, 6]):
    for lag in lags:
        df[f'{column}_lag{lag}'] = df[column].shift(lag)
    return df


def compute_features_targets(df):
    df = df.copy()
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # --- Changed ---
    # Using 'ffill' and 'bfill' consistently to fill missing values for ALL features.
    for col in ['pm2_5', 'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        
    df['aqi'] = compute_pm25_aqi(df['pm2_5'])

    # Add lag features for PM2.5 and ozone (for example)
    df = add_lag_features(df, 'pm2_5')
    df = add_lag_features(df, 'ozone')

    df = df.dropna()  

    features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide',
                'sulphur_dioxide', 'ozone',
                'pm2_5_lag1', 'pm2_5_lag3', 'pm2_5_lag6',
                'ozone_lag1', 'ozone_lag3', 'ozone_lag6']
    # Add missing features to features list (Assuming they are available in df after previous operations):
    missing_features = ['hour','day','month']
    features.extend(missing_features)


    X = df[features]
    y = df['aqi']
    return X, y.values, df # Return the full dataframe

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def train_evaluate_models(X, y):
    print("Splitting data (train/test)...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Normalize where needed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['RandomForest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'R2': r2_score(y_test, rf_pred),
        'model': rf
    }
    print("RandomForest feature importance:")
    print(dict(zip(X.columns, rf.feature_importances_)))

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    results['Ridge'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
        'MAE': mean_absolute_error(y_test, ridge_pred),
        'R2': r2_score(y_test, ridge_pred),
        'model': ridge
    }

    # Extra Trees
    et = ExtraTreesRegressor(n_estimators=150, random_state=42)
    et.fit(X_train, y_train)
    et_pred = et.predict(X_test)
    results['ExtraTrees'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, et_pred)),
        'MAE': mean_absolute_error(y_test, et_pred),
        'R2': r2_score(y_test, et_pred),
        'model': et
    }

    # # Gradient Boosting
    # gb = GradientBoostingRegressor(n_estimators=150, random_state=42)
    # gb.fit(X_train, y_train)
    # gb_pred = gb.predict(X_test)
    # results['GradientBoosting'] = {
    #     'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
    #     'MAE': mean_absolute_error(y_test, gb_pred),
    #     'R2': r2_score(y_test, gb_pred),
    #     'model': gb
    # }


    # # SVR
    # svr = SVR(C=1.0, epsilon=0.2)
    # svr.fit(X_train_scaled, y_train)
    # svr_pred = svr.predict(X_test_scaled)
    # results['SVR'] = {
    #     'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
    #     'MAE': mean_absolute_error(y_test, svr_pred),
    #     'R2': r2_score(y_test, svr_pred),
    #     'model': svr
    # }

    # # MLP
    # mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    # mlp.fit(X_train_scaled, y_train)
    # mlp_pred = mlp.predict(X_test_scaled)
    # results['MLP'] = {
    #     'RMSE': np.sqrt(mean_squared_error(y_test, mlp_pred)),
    #     'MAE': mean_absolute_error(y_test, mlp_pred),
    #     'R2': r2_score(y_test, mlp_pred),
    #     'model': mlp
    # }

    return results

def store_in_feature_store(features_df, target_list):
    import hopsworks
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS')
    fs = project.get_feature_store()

    print("Creating or getting feature group (version 2)...")
    fg = fs.get_or_create_feature_group(
        name="air_quality_features",
        version=2,  # NEW VERSION to accommodate updated schema
        primary_key=['hour', 'day', 'month'],
        description="Air quality features and targets v2"
    )

    print("Preparing data for insertion...")
    data = features_df.copy()
    data['target_aqi'] = target_list.astype('float32')  # Ensure compatible dtype

    print("Inserting data into feature group...")
    fg.insert(data)
    print("✅ Feature data stored successfully!")
    return fg


def store_model_in_registry(results, project):
    mr = project.get_model_registry()

    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_model_name]['model']
    best_metrics = {k: v for k, v in results[best_model_name].items() if k != 'model'}

    joblib.dump(best_model, 'best_model.pkl')
    model = mr.python.create_model(
        name="air_quality_predictor",
        metrics=best_metrics,
        description=f"Best model: {best_model_name}"
    )
    model.save('best_model.pkl')
    return best_model, best_model_name

def predict_future(best_model, latest_features_df):
    return best_model.predict(latest_features_df.values)

def main():
    print("Fetching data...")
    df = fetch_openmeteo_data(start_date="2015-01-01")

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

    