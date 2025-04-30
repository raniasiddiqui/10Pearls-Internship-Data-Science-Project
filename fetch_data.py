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

def fetch_openmeteo_data(start_date="2020-01-01", end_date=None):
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

