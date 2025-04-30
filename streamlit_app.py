import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Import the necessary functions from your original code
from airquality import (
    fetch_openmeteo_data,
    compute_features_targets,
    train_evaluate_models,
    store_in_feature_store,
    store_model_in_registry,
    predict_future,
    compute_pm25_aqi
)

# Set page configuration
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4D7CFE;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E90FF;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Air Quality Prediction Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Controls</h2>", unsafe_allow_html=True)
    
    # Date range for historical data
    st.markdown("### Historical Data")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.date_input("End Date", datetime.now())
    
    # Future prediction range
    st.markdown("### Prediction")
    prediction_days = st.slider("Prediction Days", 1, 7, 3)
    
    # Actions
    st.markdown("### Actions")
    load_data_button = st.button("Fetch Data")
    train_model_button = st.button("Train Model")
    predict_button = st.button("Generate Predictions")
    
    # Information
    st.markdown("### About")
    st.info("""
        This dashboard visualizes air quality data for Karachi, Pakistan.
        The system fetches data from Open-Meteo API, trains machine learning models,
        and predicts future air quality index (AQI) values.
    """)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'future_df' not in st.session_state:
    st.session_state.future_df = None
if 'future_preds' not in st.session_state:
    st.session_state.future_preds = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None
if 'targets' not in st.session_state:
    st.session_state.targets = None
if 'full_df' not in st.session_state:
    st.session_state.full_df = None
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

# Function to categorize AQI values
def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

# Function to display a loading spinner
def with_loading_spinner(func):
    def wrapper(*args, **kwargs):
        st.session_state.is_loading = True
        result = func(*args, **kwargs)
        st.session_state.is_loading = False
        return result
    return wrapper

# Fetch data
@with_loading_spinner
def load_data():
    with st.spinner("Fetching air quality data..."):
        df = fetch_openmeteo_data(start_date=start_date.strftime("%Y-%m-%d"), 
                                  end_date=end_date.strftime("%Y-%m-%d"))
        st.session_state.df = df
        
        # Compute features and targets
        features_df, targets, full_df = compute_features_targets(df)
        st.session_state.features_df = features_df
        st.session_state.targets = targets
        st.session_state.full_df = full_df
        
        return df, features_df, targets, full_df

# Train model
@with_loading_spinner
def train_model():
    with st.spinner("Training models..."):
        results = train_evaluate_models(st.session_state.features_df, st.session_state.targets)
        st.session_state.results = results
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        best_model = results[best_model_name]['model']
        st.session_state.best_model = best_model
        st.session_state.best_model_name = best_model_name
        
        return results, best_model, best_model_name

# Generate predictions
@with_loading_spinner
def generate_predictions():
    with st.spinner("Generating predictions..."):
        future_df = fetch_openmeteo_data(
            start_date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=(datetime.now() + timedelta(days=prediction_days)).strftime("%Y-%m-%d")
        )
        
        future_features, _, _ = compute_features_targets(future_df)
        future_preds = predict_future(st.session_state.best_model, future_features)
        
        st.session_state.future_df = future_df
        st.session_state.future_preds = future_preds
        
        return future_df, future_preds

# Button handlers
if load_data_button:
    df, features_df, targets, full_df = load_data()
    st.success("Data loaded successfully!")

if train_model_button:
    if st.session_state.df is None:
        st.error("Please load data first!")
    else:
        results, best_model, best_model_name = train_model()
        st.success(f"Model training complete! Best model: {best_model_name}")

if predict_button:
    if st.session_state.best_model is None:
        st.error("Please train model first!")
    else:
        future_df, future_preds = generate_predictions()
        st.success(f"Predictions generated for {prediction_days} days!")

# Main dashboard
if st.session_state.is_loading:
    st.spinner("Processing...")

# Display historical data if available
if st.session_state.df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Historical Air Quality</h2>", unsafe_allow_html=True)
        
        # Calculate AQI
        if 'aqi' not in st.session_state.df.columns:
            st.session_state.df['aqi'] = compute_pm25_aqi(st.session_state.df['pm2_5'])
        
        # Create daily average dataframe
        daily_df = st.session_state.df.copy()
        daily_df['date'] = daily_df['date'].dt.date
        daily_avg = daily_df.groupby('date').agg({
            'pm2_5': 'mean',
            'aqi': 'mean',
            'carbon_monoxide': 'mean',
            'nitrogen_dioxide': 'mean',
            'sulphur_dioxide': 'mean',
            'ozone': 'mean'
        }).reset_index()
        
        # Line chart for AQI
        fig = px.line(daily_avg, x='date', y='aqi', 
                      title='Daily Average Air Quality Index (AQI)',
                      labels={'date': 'Date', 'aqi': 'AQI'},
                      color_discrete_sequence=['#4D7CFE'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Add AQI range markers
        for daily_aqi in daily_avg['aqi']:
            cat, color = categorize_aqi(daily_aqi)
        
        # Add AQI category legend
        legend_data = [
            ("Good (0-50)", "#00e400"),
            ("Moderate (51-100)", "#ffff00"),
            ("Unhealthy for Sensitive Groups (101-150)", "#ff7e00"),
            ("Unhealthy (151-200)", "#ff0000"),
            ("Very Unhealthy (201-300)", "#8f3f97"),
            ("Hazardous (301+)", "#7e0023")
        ]
        
        # Create a legend chart
        legend_fig = go.Figure()
        for i, (label, color) in enumerate(legend_data):
            legend_fig.add_trace(go.Scatter(
                x=[0], y=[i],
                mode="markers",
                marker=dict(size=15, color=color),
                name=label,
                showlegend=True
            ))
        legend_fig.update_layout(
            title="AQI Categories",
            showlegend=True,
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(legend_fig, use_container_width=True)

    with col2:
        st.markdown("<h2 class='sub-header'>Pollutant Trends</h2>", unsafe_allow_html=True)
        
        # Select pollutant
        pollutant = st.selectbox("Select Pollutant", 
                                ['pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'])
        
        # Line chart for selected pollutant
        fig = px.line(daily_avg, x='date', y=pollutant,
                      title=f'Daily Average {pollutant.replace("_", " ").title()} Concentration',
                      labels={'date': 'Date', pollutant: 'Concentration'},
                      color_discrete_sequence=['#1E90FF'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("<h3>Pollutant Correlations</h3>", unsafe_allow_html=True)
        corr_columns = ['pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
        corr_df = st.session_state.df[corr_columns].corr()
        
        fig = px.imshow(corr_df, 
                        labels=dict(x="Pollutant", y="Pollutant", color="Correlation"),
                        x=corr_columns,
                        y=corr_columns,
                        color_continuous_scale="RdBu_r")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Display model results if available
    if st.session_state.results is not None:
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create performance metrics table
            results_df = pd.DataFrame({
                'Model': [],
                'RMSE': [],
                'MAE': [],
                'R²': []
            })
            
            for model, metrics in st.session_state.results.items():
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Model': [model],
                    'RMSE': [metrics['RMSE']],
                    'MAE': [metrics['MAE']],
                    'R²': [metrics['R2']]
                })], ignore_index=True)
            
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['R²']).highlight_min(axis=0, subset=['RMSE', 'MAE']), 
                        use_container_width=True)
            
            # Add best model indicator
            st.success(f"Best Model: {st.session_state.best_model_name} (R² = {st.session_state.results[st.session_state.best_model_name]['R2']:.4f})")
            
        with col2:
            # Create performance comparison chart
            fig = go.Figure()
            
            for model in results_df['Model']:
                fig.add_trace(go.Bar(
                    x=['RMSE', 'MAE'],
                    y=[results_df.loc[results_df['Model'] == model, 'RMSE'].values[0],
                       results_df.loc[results_df['Model'] == model, 'MAE'].values[0]],
                    name=model
                ))
            
            fig.update_layout(
                title="Error Metrics Comparison",
                xaxis_title="Metric",
                yaxis_title="Value",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create R² comparison chart
            fig = px.bar(results_df, x='Model', y='R²',
                         title="R² Score Comparison", 
                         color='Model',
                         labels={'R²': 'R² Score', 'Model': 'Model'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance if available
            if st.session_state.best_model_name == 'RandomForest' or st.session_state.best_model_name == 'ExtraTrees':
                st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
                
                # Get feature importances
                importances = st.session_state.best_model.feature_importances_
                features = st.session_state.features_df.columns
                
                # Create sorted dataframe
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Create feature importance chart
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                            orientation='h',
                            title=f"Feature Importance ({st.session_state.best_model_name})",
                            color='Importance',
                            labels={'Importance': 'Importance', 'Feature': 'Feature'})
                
                st.plotly_chart(fig, use_container_width=True)

    # Display predictions if available
    if st.session_state.future_preds is not None:
        st.markdown("<h2 class='sub-header'>AQI Forecasts</h2>", unsafe_allow_html=True)
        
        # Create forecast dataframe
        # FIX: Check and align the lengths of future_df['date'] and future_preds
        future_dates = st.session_state.future_df['date']
        future_predictions = st.session_state.future_preds
        
        # Make sure both arrays have the same length
        min_len = min(len(future_dates), len(future_predictions))
        future_dates = future_dates[:min_len]
        future_predictions = future_predictions[:min_len]
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'AQI': future_predictions
        })
        
        # Add AQI category
        forecast_df['Category'], forecast_df['Color'] = zip(*forecast_df['AQI'].apply(lambda x: categorize_aqi(x)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Line chart for forecasted AQI
            fig = px.line(forecast_df, x='Date', y='AQI',
                          title=f'Forecasted AQI for Next {prediction_days} Days',
                          labels={'Date': 'Date', 'AQI': 'AQI'},
                          color_discrete_sequence=['#4D7CFE'])
            
            # Add range markers
            fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="#00e400", opacity=0.2)
            fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="#ffff00", opacity=0.2)
            fig.add_hrect(y0=100, y1=150, line_width=0, fillcolor="#ff7e00", opacity=0.2)
            fig.add_hrect(y0=150, y1=200, line_width=0, fillcolor="#ff0000", opacity=0.2)
            fig.add_hrect(y0=200, y1=300, line_width=0, fillcolor="#8f3f97", opacity=0.2)
            fig.add_hrect(y0=300, y1=500, line_width=0, fillcolor="#7e0023", opacity=0.2)
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Display predictions table
            st.markdown("<h3>Hourly Predictions</h3>", unsafe_allow_html=True)
            
            # Format table
            display_df = forecast_df.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['AQI'] = display_df['AQI'].round(1)
            
            # Create styled dataframe
            st.dataframe(
                display_df[['Date', 'AQI', 'Category']].style.apply(
                    lambda x: ['background-color: ' + display_df['Color'].iloc[i] + '; color: black' 
                              if col == 'Category' else '' for i, col in enumerate(x)], 
                    axis=0
                ),
                height=400,
                use_container_width=True
            )
            
            # AQI distribution
            st.markdown("<h3>AQI Category Distribution</h3>", unsafe_allow_html=True)
            
            # Count categories
            category_counts = forecast_df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Create category colors dict
            category_colors = {cat: color for cat, color in zip(forecast_df['Category'], forecast_df['Color'])}
            
            # Create pie chart
            fig = px.pie(category_counts, values='Count', names='Category',
                         title='Forecast AQI Categories',
                         color='Category',
                         color_discrete_map=category_colors)
            
            st.plotly_chart(fig, use_container_width=True)

    # Display metrics cards
    if st.session_state.df is not None:
        st.markdown("<h2 class='sub-header'>Current Air Quality Metrics</h2>", unsafe_allow_html=True)
        
        # Get latest values
        latest_df = st.session_state.df.iloc[-1]
        
        # Create metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            cat, color = categorize_aqi(latest_df['aqi'])
            st.markdown(f"<div style='text-align: center;'><span class='metric-label'>Current AQI</span><br>"+
                        f"<span class='metric-value' style='color:{color};'>{latest_df['aqi']:.1f}</span><br>"+
                        f"<span style='color:{color};'>{cat}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center;'><span class='metric-label'>PM2.5</span><br>"+
                        f"<span class='metric-value'>{latest_df['pm2_5']:.1f}</span><br>"+
                        f"<span>μg/m³</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center;'><span class='metric-label'>Ozone</span><br>"+
                        f"<span class='metric-value'>{latest_df['ozone']:.1f}</span><br>"+
                        f"<span>μg/m³</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center;'><span class='metric-label'>NO₂</span><br>"+
                        f"<span class='metric-value'>{latest_df['nitrogen_dioxide']:.1f}</span><br>"+
                        f"<span>μg/m³</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Display additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>24-Hour Trend</h3>", unsafe_allow_html=True)
            
            # Get last 24 hours data
            last_24h = st.session_state.df.iloc[-24:].copy()
            
            # Create mini trend chart
            fig = px.line(last_24h, x='date', y='aqi',
                         labels={'date': '', 'aqi': ''},
                         color_discrete_sequence=['#4D7CFE'])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                             showlegend=False,
                             height=150)
            fig.update_xaxes(showticklabels=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend info
            trend = last_24h['aqi'].iloc[-1] - last_24h['aqi'].iloc[0]
            trend_color = "#00e400" if trend < 0 else "#ff0000"
            
            st.markdown(f"<div style='text-align: center;'>"+
                        f"<span style='color:{trend_color};'>{'↓' if trend < 0 else '↑'} {abs(trend):.1f} points</span>"+
                        f"</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Pollution Breakdown</h3>", unsafe_allow_html=True)
            
            # Create pollutant breakdown
            pollutants = ['pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
            pollutant_labels = ['PM2.5', 'CO', 'NO₂', 'SO₂', 'O₃']
            
            # Create radar chart
            values = []
            for p in pollutants:
                # Normalize values (0-1 scale)
                if p == 'pm2_5':
                    values.append(min(1, latest_df[p] / 100))
                elif p == 'carbon_monoxide':
                    values.append(min(1, latest_df[p] / 15000))
                elif p == 'nitrogen_dioxide':
                    values.append(min(1, latest_df[p] / 200))
                elif p == 'sulphur_dioxide':
                    values.append(min(1, latest_df[p] / 350))
                elif p == 'ozone':
                    values.append(min(1, latest_df[p] / 180))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=pollutant_labels,
                fill='toself',
                name='Pollutants'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Health Recommendations</h3>", unsafe_allow_html=True)
            
            # Get health recommendations based on AQI
            aqi_value = latest_df['aqi']
            
            if aqi_value <= 50:
                recommendation = "Air quality is good. Perfect for outdoor activities."
            elif aqi_value <= 100:
                recommendation = "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion."
            elif aqi_value <= 150:
                recommendation = "Members of sensitive groups may experience health effects. Limit outdoor activities."
            elif aqi_value <= 200:
                recommendation = "Everyone may begin to experience health effects. Avoid prolonged outdoor activities."
            elif aqi_value <= 300:
                recommendation = "Health alert: everyone may experience more serious health effects. Avoid outdoor activities."
            else:
                recommendation = "Health warning of emergency conditions. Stay indoors and keep activity levels low."
            
            st.markdown(f"<div style='text-align: center; padding: 10px;'>"+
                        f"<p>{recommendation}</p>"+
                        f"</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# If no data is loaded yet, show welcome message
if st.session_state.df is None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center;'>Welcome to the Air Quality Dashboard</h2>
    <p style='text-align: center;'>
        This dashboard allows you to visualize and predict air quality data for Karachi, Pakistan.
        To get started, use the controls in the sidebar to fetch data and train models.
    </p>
    <ol>
        <li>Select a date range for historical data</li>
        <li>Click "Fetch Data" to load air quality measurements</li>
        <li>Click "Train Model" to train machine learning models</li>
        <li>Click "Generate Predictions" to forecast future air quality</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add demo image
    st.image("https://www.epa.gov/sites/default/files/styles/large/public/2021-03/aqi_graphic.png", 
             caption="Air Quality Index (AQI) Scale", use_column_width=True)

# Footer
st.markdown("<div class='footer'>Developed with ❤️ for cleaner air</div>", unsafe_allow_html=True)