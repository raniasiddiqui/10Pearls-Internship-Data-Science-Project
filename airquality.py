# import streamlit as st
# import hopsworks
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import joblib
# from datetime import datetime, timedelta
# import os

# # Function to connect to Hopsworks and load features
# @st.cache_resource
# def load_features_from_hopsworks():
#     try:
#         # Login to Hopsworks
#         project = hopsworks.login(
#             api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS'
#         )
#         fs = project.get_feature_store()

#         # Load feature group
#         fg = fs.get_feature_group(name="air_quality_features", version=2)
#         df = fg.read()

#         return df
#     except Exception as e:
#         st.error(f"Error loading features from Hopsworks: {e}")
#         return None

# # Function to load the trained model
# @st.cache_resource
# def load_model_from_hopsworks():
#     try:
#         project = hopsworks.login(
#             api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS'
#         )
#         mr = project.get_model_registry()
#         model = mr.get_model(name="air_quality_predictor", version=1)  # Adjust version if needed
#         model_dir = model.download()
#         model_path = f"{model_dir}/best_model.pkl"
#         loaded_model = joblib.load(model_path)
#         return loaded_model
#     except Exception as e:
#         st.error(f"Error loading model from Hopsworks: {e}")
#         return None

# # Function to prepare features for prediction
# def prepare_features(df):
#     features = [
#         'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
#         'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide',
#         'sulphur_dioxide', 'ozone',
#         'pm2_5_lag1', 'pm2_5_lag3', 'pm2_5_lag6',
#         'ozone_lag1', 'ozone_lag3', 'ozone_lag6',
#         'hour', 'day', 'month'
#     ]
#     # Ensure all required features are present
#     missing_features = [f for f in features if f not in df.columns]
#     if missing_features:
#         st.error(f"Missing features: {missing_features}")
#         return None
    
#     return df[features]

# # Function to generate future timestamps for prediction
# def generate_future_timestamps(start_date, days=3):
#     start = pd.to_datetime(start_date)
#     timestamps = pd.date_range(start=start, periods=24*days, freq='H')
#     return timestamps

# # Streamlit App
# st.title("Air Quality Prediction Dashboard")
# st.markdown("""
# This dashboard displays air quality index (AQI) predictions based on PM2.5 levels for Karachi (24.8608°N, 67.0104°E).
# The predictions are generated using a machine learning model trained on historical air quality data.
# """)

# # Sidebar for user input
# st.sidebar.header("Prediction Settings")
# days_to_predict = st.sidebar.slider("Days to Predict", 1, 7, 3)
# start_date = st.sidebar.date_input("Start Date", datetime.utcnow().date())

# # Load data and model
# with st.spinner("Loading features and model..."):
#     features_df = load_features_from_hopsworks()
#     model = load_model_from_hopsworks()

# if features_df is not None and model is not None:
#     # Prepare features for prediction
#     X = prepare_features(features_df)
    
#     if X is not None:
#         # Make predictions
#         predictions = model.predict(X)
        
#         # Create a DataFrame for visualization
#         pred_df = pd.DataFrame({
#             'timestamp': features_df['date'] if 'date' in features_df.columns else pd.date_range(start='2015-01-01', periods=len(predictions), freq='H'),
#             'AQI': predictions
#         })

#         # Filter data for display (last 30 days for historical, future predictions)
#         historical_df = pred_df[pred_df['timestamp'] <= datetime.utcnow()]
#         historical_df = historical_df.tail(24 * 30)  # Last 30 days

#         # Generate future predictions (mocked here using latest features)
#         future_timestamps = generate_future_timestamps(start_date, days_to_predict)
#         future_features = X.tail(24 * days_to_predict)  # Simplified: reuse recent features
#         future_predictions = model.predict(future_features)
        
#         future_df = pd.DataFrame({
#             'timestamp': future_timestamps,
#             'AQI': future_predictions
#         })

#         # Plot historical data
#         st.subheader("Historical AQI (Last 30 Days)")
#         fig1 = px.line(historical_df, x='timestamp', y='AQI', title="Historical AQI based on PM2.5")
#         fig1.update_layout(xaxis_title="Date", yaxis_title="AQI", hovermode="x unified")
#         st.plotly_chart(fig1, use_container_width=True)

#         # Plot future predictions
#         st.subheader(f"Forecasted AQI (Next {days_to_predict} Days)")
#         fig2 = px.line(future_df, x='timestamp', y='AQI', title=f"AQI Forecast from {start_date}")
#         fig2.update_layout(xaxis_title="Date", yaxis_title="AQI", hovermode="x unified")
#         fig2.add_trace(go.Scatter(
#             x=future_df['timestamp'],
#             y=future_df['AQI'],
#             mode='markers',
#             marker=dict(size=8, color='red'),
#             name='Predicted Points'
#         ))
#         st.plotly_chart(fig2, use_container_width=True)

#         # Display prediction summary
#         st.subheader("Prediction Summary")
#         avg_aqi = future_df['AQI'].mean()
#         max_aqi = future_df['AQI'].max()
#         min_aqi = future_df['AQI'].min()
#         st.write(f"- Average AQI: {avg_aqi:.2f}")
#         st.write(f"- Maximum AQI: {max_aqi:.2f}")
#         st.write(f"- Minimum AQI: {min_aqi:.2f}")

#         # AQI interpretation
#         st.subheader("AQI Interpretation")
#         aqi_ranges = {
#             (0, 50): "Good",
#             (51, 100): "Moderate",
#             (101, 150): "Unhealthy for Sensitive Groups",
#             (151, 200): "Unhealthy",
#             (201, 300): "Very Unhealthy",
#             (301, 500): "Hazardous"
#         }
#         for (low, high), category in aqi_ranges.items():
#             if low <= avg_aqi <= high:
#                 st.write(f"The average forecasted AQI falls in the **{category}** category.")
#                 break
# else:
#     st.error("Unable to load data or model. Please check Hopsworks connection and try again.")

# # Footer
# st.markdown("---")
# st.markdown("Built with Streamlit and Hopsworks by [Your Name]. Data sourced from Open-Meteo Air Quality API.")

import streamlit as st
import hopsworks
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import os

# Function to connect to Hopsworks and load features
@st.cache_resource
def load_features_from_hopsworks():
    try:
        project = hopsworks.login(
            api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS'
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="air_quality_features", version=2)
        df = fg.read()
        return df
    except Exception as e:
        st.error(f"Error loading features from Hopsworks: {e}")
        return None

# Function to load the trained model
@st.cache_resource
def load_model_from_hopsworks():
    try:
        project = hopsworks.login(
            api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS'
        )
        mr = project.get_model_registry()
        model = mr.get_model(name="air_quality_predictor", version=2)  # Adjust version if needed
        model_dir = model.download()
        model_path = f"{model_dir}/best_model.pkl"
        loaded_model = joblib.load(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model from Hopsworks: {e}")
        return None

# Function to prepare features for prediction
def prepare_features(df, model):
    # Define all possible features (as per original code)
    possible_features = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide',
        'sulphur_dioxide', 'ozone',
        'pm2_5_lag1', 'pm2_5_lag3', 'pm2_5_lag6',
        'ozone_lag1', 'ozone_lag3', 'ozone_lag6',
        'hour', 'day', 'month'
    ]
    
    # Get the number of features the model expects
    expected_n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 8  # Fallback to 8 if unknown
    
    # Filter available features in df that match possible_features
    available_features = [f for f in possible_features if f in df.columns]
    
    if len(available_features) < expected_n_features:
        st.error(f"Not enough features available. Model expects {expected_n_features}, but only {len(available_features)} found: {available_features}")
        return None
    
    # Select the first 'expected_n_features' to match the model's input
    selected_features = available_features[:expected_n_features]
    st.write(f"Using features for prediction: {selected_features}")
    
    # Return features as a NumPy array to avoid feature name issues
    return df[selected_features].values

# Function to generate future timestamps for prediction
def generate_future_timestamps(start_date, days=3):
    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=24*days, freq='H')
    return timestamps

# Streamlit App
st.title("Air Quality Prediction Dashboard")
st.markdown("""
This dashboard displays air quality index (AQI) predictions based on PM2.5 levels for Karachi (24.8608°N, 67.0104°E).
The predictions are generated using a machine learning model trained on historical air quality data.
""")

# Sidebar for user input
st.sidebar.header("Prediction Settings")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 7, 3)
start_date = st.sidebar.date_input("Start Date", datetime.utcnow().date())

# Load data and model
with st.spinner("Loading features and model..."):
    features_df = load_features_from_hopsworks()
    model = load_model_from_hopsworks()

if features_df is not None and model is not None:
    # Prepare features for prediction
    X = prepare_features(features_df, model)
    
    if X is not None:
        # Make predictions
        try:
            predictions = model.predict(X)
        except ValueError as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
        
        # Create a DataFrame for visualization
        pred_df = pd.DataFrame({
            'timestamp': features_df['date'] if 'date' in features_df.columns else pd.date_range(start='2015-01-01', periods=len(predictions), freq='H'),
            'AQI': predictions
        })

        # Filter data for display (last 30 days for historical, future predictions)
        historical_df = pred_df[pred_df['timestamp'] <= datetime.utcnow()]
        historical_df = historical_df.tail(24 * 30)  # Last 30 days

        # Generate future predictions (mocked using latest features)
        future_timestamps = generate_future_timestamps(start_date, days_to_predict)
        future_features = X[-24 * days_to_predict:]  # Reuse recent features
        try:
            future_predictions = model.predict(future_features)
        except ValueError as e:
            st.error(f"Future prediction failed: {e}")
            st.stop()
        
        future_df = pd.DataFrame({
            'timestamp': future_timestamps,
            'AQI': future_predictions
        })

        # Plot historical data
        st.subheader("Historical AQI (Last 30 Days)")
        fig1 = px.line(historical_df, x='timestamp', y='AQI', title="Historical AQI based on PM2.5")
        fig1.update_layout(xaxis_title="Date", yaxis_title="AQI", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        # Plot future predictions
        st.subheader(f"Forecasted AQI (Next {days_to_predict} Days)")
        fig2 = px.line(future_df, x='timestamp', y='AQI', title=f"AQI Forecast from {start_date}")
        fig2.update_layout(xaxis_title="Date", yaxis_title="AQI", hovermode="x unified")
        fig2.add_trace(go.Scatter(
            x=future_df['timestamp'],
            y=future_df['AQI'],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Predicted Points'
        ))
        st.plotly_chart(fig2, use_container_width=True)

        # Display prediction summary
        st.subheader("Prediction Summary")
        avg_aqi = future_df['AQI'].mean()
        max_aqi = future_df['AQI'].max()
        min_aqi = future_df['AQI'].min()
        st.write(f"- Average AQI: {avg_aqi:.2f}")
        st.write(f"- Maximum AQI: {max_aqi:.2f}")
        st.write(f"- Minimum AQI: {min_aqi:.2f}")

        # AQI interpretation
        st.subheader("AQI Interpretation")
        aqi_ranges = {
            (0, 50): "Good",
            (51, 100): "Moderate",
            (101, 150): "Unhealthy for Sensitive Groups",
            (151, 200): "Unhealthy",
            (201, 300): "Very Unhealthy",
            (301, 500): "Hazardous"
        }
        for (low, high), category in aqi_ranges.items():
            if low <= avg_aqi <= high:
                st.write(f"The average forecasted AQI falls in the **{category}** category.")
                break
else:
    st.error("Unable to load data or model. Please check Hopsworks connection and try again.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Hopsworks. Data sourced from Open-Meteo Air Quality API.")