import streamlit as st
import pandas as pd
import os
import glob
import json
import numpy as np
import plotly.express as px
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Electricity Demand Prediction", layout="wide")

# Title of the app
st.title("Electricity Demand Prediction Dashboard")

# Sidebar for task navigation
st.sidebar.title("Navigation")
task = st.sidebar.selectbox("Select Task", [
    "Task 1: Data Loading and Integration",
    "Task 2: Data Preprocessing",
    "Task 3: Exploratory Data Analysis (EDA)",
    "Task 4: Outlier Detection and Handling",
    "Task 5: Regression Modeling"
])

# Define raw data folder path
raw_folder = os.path.join(os.getcwd(), "raw")

# --- Helper Functions ---

## Data Loading Function
@st.cache_data
def load_data(files, data_type):
    """Load and concatenate data from CSV or JSON files with consistent timezone handling."""
    dfs = []
    for file in files:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8')
            elif file.endswith('.json'):
                with open(file, encoding='utf-8') as f:
                    json_data = json.load(f)
                if 'response' in json_data and 'data' in json_data['response']:
                    df = pd.json_normalize(json_data['response']['data'], sep='_')
                else:
                    continue
            else:
                continue

            # Convert key columns to appropriate types with timezone handling
            if data_type == 'electricity':
                if 'period' in df.columns:
                    df['period'] = pd.to_datetime(df['period'], errors='coerce', utc=True)
                if 'value' in df.columns:
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
            elif data_type == 'weather':
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
                if 'temperature_2m' in df.columns:
                    df['temperature_2m'] = pd.to_numeric(df['temperature_2m'], errors='coerce')

            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")

    if dfs:
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df
        except ValueError as e:
            st.error(f"Error concatenating {data_type} DataFrames: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

## Preprocessing Functions
@st.cache_data
def preprocess_electricity_data(elect_data):
    """Preprocess electricity data by aggregating, reindexing, and interpolating."""
    if elect_data.empty:
        return pd.DataFrame()

    # Select relevant columns
    elect_data = elect_data[['period', 'value']].copy()

    # Aggregate by period (sum values across sub-regions)
    elect_data = elect_data.groupby('period')['value'].sum().reset_index()

    # Create full hourly date range
    min_period = elect_data['period'].min()
    max_period = elect_data['period'].max()
    full_range = pd.date_range(start=min_period, end=max_period, freq='H')

    # Reindex to ensure continuous time series
    elect_data = elect_data.set_index('period').reindex(full_range).reset_index()
    elect_data = elect_data.rename(columns={'index': 'period'})

    # Interpolate missing values
    elect_data['value'] = elect_data['value'].interpolate(method='linear')
    # Handle edge cases with forward/backward fill
    elect_data['value'] = elect_data['value'].fillna(method='ffill').fillna(method='bfill')

    return elect_data

@st.cache_data
@st.cache_data
def preprocess_weather_data(weather_data, full_range):
    """Preprocess weather data by handling duplicates, reindexing, and interpolating."""
    if weather_data.empty or full_range is None:
        return pd.DataFrame()

    # Select relevant columns
    weather_data = weather_data[['date', 'temperature_2m']].copy()

    # Drop rows with invalid (NaT) dates
    weather_data = weather_data.dropna(subset=['date'])

    # Handle duplicate dates by taking the mean temperature
    weather_data = weather_data.groupby('date')['temperature_2m'].mean().reset_index()

    # Set 'date' as index
    weather_data = weather_data.set_index('date')

    # Reindex to match the full_range from electricity data
    weather_data = weather_data.reindex(full_range)

    # Interpolate missing temperature values
    weather_data['temperature_2m'] = weather_data['temperature_2m'].interpolate(method='linear')
    weather_data['temperature_2m'] = weather_data['temperature_2m'].fillna(method='ffill').fillna(method='bfill')

    # Reset index and rename to 'period'
    weather_data = weather_data.reset_index()
    weather_data = weather_data.rename(columns={'index': 'period'})

    return weather_data

@st.cache_data
def merge_datasets(elect_data, weather_data):
    """Merge electricity and weather data on period."""
    if elect_data.empty or weather_data.empty:
        return pd.DataFrame()

    merged_data = pd.merge(elect_data, weather_data, on='period', how='inner')
    return merged_data

@st.cache_data
def feature_engineering(merged_data):
    """Add temporal features to the merged dataset."""
    if merged_data.empty:
        return pd.DataFrame()

    merged_data = merged_data.copy()
    merged_data['hour'] = merged_data['period'].dt.hour
    merged_data['day_of_week'] = merged_data['period'].dt.dayofweek
    merged_data['month'] = merged_data['period'].dt.month
    merged_data['is_weekend'] = merged_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    return merged_data

# --- Load Data Once for All Tasks ---
elec_data_files = glob.glob(os.path.join(raw_folder, 'electricity_raw_data', '**', '*'), recursive=True)
weather_data_files = glob.glob(os.path.join(raw_folder, 'weather_raw_data', '**', '*'), recursive=True)
elect_data = load_data(elec_data_files, 'electricity')
weather_data = load_data(weather_data_files, 'weather')

# --- Task 1: Data Loading and Integration ---
if task == "Task 1: Data Loading and Integration":
    st.header("Task 1: Data Loading and Integration")
    st.markdown("""
    This task loads electricity and weather data from CSV and JSON files in the `raw` directory.
    Electricity data includes demand values over time, while weather data includes temperature.
    """)

    if not os.path.exists(raw_folder):
        st.error(f"Raw data folder not found at: {raw_folder}. Please ensure the 'raw' directory exists.")
    else:
        st.markdown(f"**Found {len(elec_data_files)} electricity data files and {len(weather_data_files)} weather data files.**")

        # Electricity Data
        st.subheader("Electricity Data")
        if not elect_data.empty:
            st.write(f"**Total Records:** {elect_data.shape[0]}")
            st.write(f"**Columns:** {', '.join(elect_data.columns)}")
            st.write("**Data Types:**")
            st.dataframe(elect_data.dtypes, use_container_width=True)
            st.write("**Missing Values:**")
            st.dataframe(elect_data.isnull().sum(), use_container_width=True)
            st.write("**Sample Data:**")
            st.dataframe(elect_data.head(), use_container_width=True)
        else:
            st.warning("No electricity data loaded.")

        # Weather Data
        st.subheader("Weather Data")
        if not weather_data.empty:
            st.write(f"**Total Records:** {weather_data.shape[0]}")
            st.write(f"**Columns:** {', '.join(weather_data.columns)}")
            st.write("**Data Types:**")
            st.dataframe(weather_data.dtypes, use_container_width=True)
            st.write("**Missing Values:**")
            st.dataframe(weather_data.isnull().sum(), use_container_width=True)
            st.write("**Sample Data:**")
            st.dataframe(weather_data.head(), use_container_width=True)
        else:
            st.warning("No weather data loaded.")

# --- Task 2: Data Preprocessing ---
elif task == "Task 2: Data Preprocessing":
    st.header("Task 2: Data Preprocessing")
    st.markdown("""
    This task preprocesses the data by:
    - Aggregating electricity demand by timestamp.
    - Reindexing to a continuous hourly timeline.
    - Interpolating missing values.
    - Merging electricity and weather data.
    - Engineering temporal features.
    """)

    # Preprocess data
    elect_data_preprocessed = preprocess_electricity_data(elect_data)
    full_range = elect_data_preprocessed['period'] if not elect_data_preprocessed.empty else None
    weather_data_preprocessed = preprocess_weather_data(weather_data, full_range)
    merged_data = merge_datasets(elect_data_preprocessed, weather_data_preprocessed)
    merged_data = feature_engineering(merged_data)

    # Display results
    st.subheader("Preprocessed Merged Dataset")
    if not merged_data.empty:
        st.write(f"**Total Records:** {merged_data.shape[0]}")
        st.write(f"**Columns:** {', '.join(merged_data.columns)}")
        st.write("**Missing Values After Preprocessing:**")
        st.write(merged_data.isnull().sum())
        st.write("**Sample Data:**")
        st.dataframe(merged_data.head())
        st.success("Data successfully preprocessed and merged.")
        st.write('**All Data**')
        st.dataframe(merged_data, use_container_width=True, selection_mode="multi-row")
    else:
        st.error("Preprocessing failed. Check data availability and format.")

# --- Task 3: Exploratory Data Analysis (EDA) ---
elif task == "Task 3: Exploratory Data Analysis (EDA)":
    st.header("Task 3: Exploratory Data Analysis (EDA)")
    st.markdown("""
    This task explores the preprocessed data through:
    - Statistical summaries.
    - Time series plots.
    - Univariate distributions.
    - Correlation analysis.
    - Time series decomposition and stationarity tests.
    """)

    # Preprocess data (reuse from Task 2)
    elect_data_preprocessed = preprocess_electricity_data(elect_data)
    full_range = elect_data_preprocessed['period'] if not elect_data_preprocessed.empty else None
    weather_data_preprocessed = preprocess_weather_data(weather_data, full_range)
    merged_data = merge_datasets(elect_data_preprocessed, weather_data_preprocessed)
    merged_data = feature_engineering(merged_data)

    if merged_data.empty:
        st.error("No data available for EDA. Please ensure Task 2 completes successfully.")
    else:
        # Statistical Summary
        st.subheader("Statistical Summary")
        st.dataframe(merged_data.describe(), use_container_width=True)

        # Time Series Plot
        st.subheader("Time Series Analysis")
        fig = px.line(merged_data, x='period', y='value', title='Electricity Demand Over Time')
        st.plotly_chart(fig)

        # Univariate Analysis
        st.subheader("Univariate Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(merged_data, x='value', title='Distribution of Electricity Demand')
            st.plotly_chart(fig1)
            fig2 = px.box(merged_data, y='value', title='Boxplot of Electricity Demand')
            st.plotly_chart(fig2)
        with col2:
            fig3 = px.histogram(merged_data, x='temperature_2m', title='Distribution of Temperature')
            st.plotly_chart(fig3)
            fig4 = px.box(merged_data, y='temperature_2m', title='Boxplot of Temperature')
            st.plotly_chart(fig4)

        # Correlation Analysis
        st.subheader("Correlation Analysis")
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        corr = merged_data[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Matrix')
        st.plotly_chart(fig)

        # Advanced Time Series Analysis
        st.subheader("Advanced Time Series Analysis")
        try:
            decomposition = seasonal_decompose(merged_data.set_index('period')['value'], model='additive', period=24)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
            decomposition.observed.plot(ax=ax1); ax1.set_title('Observed')
            decomposition.trend.plot(ax=ax2); ax2.set_title('Trend')
            decomposition.seasonal.plot(ax=ax3); ax3.set_title('Seasonal')
            decomposition.resid.plot(ax=ax4); ax4.set_title('Residual')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")

        result = adfuller(merged_data['value'].dropna())
        st.info("**Augmented Dickey-Fuller Test for Stationarity:**")
        st.write(f"- Test Statistic: {result[0]}")
        st.write(f"- p-value: {result[1]}")
        st.info("**Critical Values:**")
        for key, value in result[4].items():
            st.write(f"  - {key}: {value}")
        st.markdown("### **Interpretation:**")
        st.success("The p-value is < 0.05, indicating the series is stationary.")

# --- Task 4: Outlier Detection and Handling ---
elif task == "Task 4: Outlier Detection and Handling":
    st.header("Task 4: Outlier Detection and Handling")
    st.markdown("""
    This task identifies outliers in electricity demand using IQR and Z-score methods and provides an option to cap them.
    """)

    # Preprocess data (reuse from Task 2)
    elect_data_preprocessed = preprocess_electricity_data(elect_data)
    full_range = elect_data_preprocessed['period'] if not elect_data_preprocessed.empty else None
    weather_data_preprocessed = preprocess_weather_data(weather_data, full_range)
    merged_data = merge_datasets(elect_data_preprocessed, weather_data_preprocessed)
    merged_data = feature_engineering(merged_data)

    if merged_data.empty:
        st.error("No data available for outlier detection. Please ensure Task 2 completes successfully.")
    else:
        # Outlier Detection
        st.subheader("Outlier Detection Results")
        Q1 = merged_data['value'].quantile(0.25)
        Q3 = merged_data['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = merged_data[(merged_data['value'] < lower_bound) | (merged_data['value'] > upper_bound)]
        st.write(f"**IQR Method:** {len(outliers_iqr)} outliers detected.")

        merged_data['z_score'] = zscore(merged_data['value'])
        outliers_z = merged_data[abs(merged_data['z_score']) > 3]
        st.write(f"**Z-score Method (>3):** {len(outliers_z)} outliers detected.")

        # Visualization
        fig = px.line(merged_data, x='period', y='value', title='Electricity Demand with Outliers')
        fig.add_scatter(x=outliers_iqr['period'], y=outliers_iqr['value'], mode='markers', name='IQR Outliers', marker=dict(color='red'))
        fig.add_scatter(x=outliers_z['period'], y=outliers_z['value'], mode='markers', name='Z-score Outliers', marker=dict(color='blue'))
        st.plotly_chart(fig)

        # Outlier Handling
        st.subheader("Handle Outliers")
        if st.button("Cap Outliers at IQR Bounds"):
            merged_data['value'] = merged_data['value'].clip(lower=lower_bound, upper=upper_bound)
            st.success(f"Outliers capped between {lower_bound:.2f} and {upper_bound:.2f}.")
            fig_updated = px.line(merged_data, x='period', y='value', title='Electricity Demand After Capping')
            st.plotly_chart(fig_updated)

# --- Task 5: Regression Modeling ---
elif task == "Task 5: Regression Modeling":
    st.header("Task 5: Regression Modeling")
    st.markdown("""
    This task builds a linear regression model to predict electricity demand using temporal features and temperature.
    The data is split sequentially (80% train, 20% test), and the model is evaluated with MSE, RMSE, and R² metrics.
    """)

    # Preprocess data (reuse from Task 2)
    elect_data_preprocessed = preprocess_electricity_data(elect_data)
    full_range = elect_data_preprocessed['period'] if not elect_data_preprocessed.empty else None
    weather_data_preprocessed = preprocess_weather_data(weather_data, full_range)
    merged_data = merge_datasets(elect_data_preprocessed, weather_data_preprocessed)
    merged_data = feature_engineering(merged_data)

    if merged_data.empty:
        st.error("No data available for modeling. Please ensure Task 2 completes successfully.")
    else:
        # Prepare features and target
        features = ['hour', 'day_of_week', 'month', 'temperature_2m']
        target = 'value'
        X = merged_data[features].dropna()
        y = merged_data.loc[X.index, target]

        # Sequential split for time series
        merged_data = merged_data.sort_values('period')
        split_index = int(0.8 * len(merged_data))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Actual vs Predicted Plot
        st.subheader("Actual vs Predicted Values")
        fig = px.line(x=merged_data['period'].iloc[split_index:], y=y_test, title='Actual vs Predicted Demand')
        fig.add_scatter(x=merged_data['period'].iloc[split_index:], y=y_pred, mode='lines', name='Predicted')
        st.plotly_chart(fig)

        # Residual Plot
        st.subheader("Residual Analysis")
        residuals = y_test - y_pred
        fig_res = px.scatter(x=y_pred, y=residuals, title='Residuals vs Predicted', labels={'x': 'Predicted', 'y': 'Residuals'})
        st.plotly_chart(fig_res)