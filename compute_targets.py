import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
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