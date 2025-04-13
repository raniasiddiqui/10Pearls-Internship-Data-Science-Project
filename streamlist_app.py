
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hopsworks
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import functions from your airquality module
from airquality import fetch_openmeteo_data, compute_features_targets

st.set_page_config(page_title="Air Quality Predictor", layout="wide")

st.title("🌍 Air Quality Forecasting Dashboard (PM2.5 AQI)")
st.markdown("This app shows predicted Air Quality Index (PM2.5) for the next 3 days using machine learning models.")

# Load Hopsworks model
@st.cache_resource
def load_model_from_registry():
    project = hopsworks.login(api_key_value='7DjzBJJSuZ0kQOVZ.FucwuhwkAaImOt3IqR6GJv6BOL5q6LzJEhWWNsvITAD0s7uVVeWnh5XINaotS0AS')
    mr = project.get_model_registry()
    model = mr.get_model("air_quality_predictor", version=2)  # Adjust version as needed
    model_dir = model.download()
    model = joblib.load(model_dir + "/best_model.pkl")
    return model

# Define the exact 8 features needed by the model
# These should be the exact feature names used during training
MODEL_FEATURES = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'
]
# Note: If these aren't the correct 8 features, you'll need to determine the 
# exact features used when the model was trained

# Fetch future data and make predictions
@st.cache_data
def get_predictions():
    try:
        # Fetch data for next 3 days
        future_df = fetch_openmeteo_data(
            start_date=(datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=(datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")
        )
        
        # Compute all features as before
        future_features, _, _ = compute_features_targets(future_df)
        
        # Only select the 8 features needed by the model
        available_features = set(future_features.columns)
        missing_features = [f for f in MODEL_FEATURES if f not in available_features]
        
        if missing_features:
            return None, None, f"Missing required features: {missing_features}"
        
        # Select only the required features in the correct order
        prediction_features = future_features[MODEL_FEATURES]
        
        # Load model and make predictions
        model = load_model_from_registry()
        preds = model.predict(prediction_features)
        
        return future_df['date'], preds, None
    except Exception as e:
        return None, None, str(e)

# Main prediction workflow
with st.spinner("Fetching predictions..."):
    dates, aqi_preds, error = get_predictions()
    
    if error:
        st.error(f"Error making predictions: {error}")
        
        st.info("The model expects exactly 8 features. Check if the MODEL_FEATURES list contains the correct features used during training.")
        
        # Display troubleshooting section
        st.subheader("Troubleshooting Steps")
        st.markdown("""
        1. **Check the training code**: Look at the features used when the model was trained.
        2. **Update MODEL_FEATURES**: Modify the 8 features in the MODEL_FEATURES list in the code.
        3. **Verify feature names**: Make sure the spelling and format match exactly with the training features.
        """)
    else:
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'Date': dates,
            'Predicted AQI (PM2.5)': aqi_preds
        })
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        
        # Line chart
        st.subheader("📈 AQI Forecast Over Time")
        st.line_chart(df_plot.set_index('Date'))
        
        # Metrics summary
        st.subheader("📊 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Minimum AQI", f"{df_plot['Predicted AQI (PM2.5)'].min():.2f}")
        with col2:
            st.metric("Maximum AQI", f"{df_plot['Predicted AQI (PM2.5)'].max():.2f}")
        with col3:
            st.metric("Average AQI", f"{df_plot['Predicted AQI (PM2.5)'].mean():.2f}")
        
        # AQI Level Categorization
        def categorize_aqi(aqi):
            if aqi <= 50:
                return "Good"
            elif aqi <= 100:
                return "Moderate"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                return "Unhealthy"
            elif aqi <= 300:
                return "Very Unhealthy"
            else:
                return "Hazardous"
        
        df_plot['AQI Category'] = df_plot['Predicted AQI (PM2.5)'].apply(categorize_aqi)
        
        # Color mapping for categories
        category_colors = {
            "Good": "#00e400",
            "Moderate": "#ffff00", 
            "Unhealthy for Sensitive Groups": "#ff7e00",
            "Unhealthy": "#ff0000",
            "Very Unhealthy": "#99004c",
            "Hazardous": "#7e0023"
        }
        
        # Display AQI category counts
        st.subheader("🏆 AQI Category Distribution")
        category_counts = df_plot['AQI Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Hours']
        
        # Create a more visually appealing chart using pyplot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            category_counts['Category'], 
            category_counts['Hours'],
            color=[category_colors.get(cat, "#1f77b4") for cat in category_counts['Category']]
        )
        ax.set_xlabel('AQI Category')
        ax.set_ylabel('Number of Hours')
        ax.set_title('Distribution of Air Quality Categories')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Optional: Display table with color coding
        with st.expander("🔍 See hourly prediction data"):
            # Add color coding based on AQI category
            def color_aqi(val):
                category = categorize_aqi(val)
                return f'background-color: {category_colors[category]}; color: {"black" if category in ["Good", "Moderate"] else "white"}'
            
            styled_df = df_plot.style.applymap(
                color_aqi, 
                subset=['Predicted AQI (PM2.5)']
            )
            st.dataframe(styled_df, use_container_width=True)

# Add debug section for feature inspection
with st.expander("Debug: Inspect Available Features"):
    st.write("This section helps identify which features are available for prediction.")
    
    try:
        # Fetch a small sample to inspect
        sample_df = fetch_openmeteo_data(
            start_date=datetime.utcnow().strftime("%Y-%m-%d"),
            end_date=(datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        )
        
        # Process it to get features
        sample_features, _, _ = compute_features_targets(sample_df)
        
        # Display available features
        st.write("Available features:", sample_features.columns.tolist())
        st.write("Number of features available:", len(sample_features.columns))
        st.write("First few rows of feature data:")
        st.dataframe(sample_features.head())
        
        # Highlight MODEL_FEATURES in the available features
        available_set = set(sample_features.columns)
        for feature in MODEL_FEATURES:
            if feature in available_set:
                st.write(f"✅ Model feature '{feature}' is available")
            else:
                st.write(f"❌ Model feature '{feature}' is NOT available")
    except Exception as e:
        st.write(f"Error inspecting features: {e}")