# Electricity Demand Prediction Dashboard
![](/ss/img_1.png)
![Electricity Demand Prediction](https://img.shields.io/badge/Streamlit-App-orange) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

A Streamlit application for analyzing and predicting electricity demand using historical data and weather information. This dashboard provides insights into electricity consumption patterns and forecasts future demand using machine learning techniques. It integrates electricity demand data with weather data to identify correlations and improve prediction accuracy.

## Key Features
- **Data Loading and Integration**: Load and integrate data from multiple file formats (CSV, JSON).
- **Data Preprocessing**: Handle missing values, duplicates, and interpolate gaps in time series data.
- **Exploratory Data Analysis (EDA)**: Visualize data through time series plots, histograms, boxplots, and correlation matrices.
- **Advanced Time Series Analysis**: Perform decomposition and stationarity tests (Augmented Dickey-Fuller test).
- **Outlier Detection and Handling**: Identify outliers using IQR and Z-score methods with options to cap extreme values.
- **Regression Modeling**: Build and evaluate a linear regression model to predict electricity demand with performance metrics (MSE, RMSE, R²).

## Technologies Used
- **Python 3.8+**
- **Streamlit** for the interactive dashboard
- **Pandas** and **NumPy** for data manipulation
- **Plotly** and **Matplotlib** for visualizations
- **SciPy** and **StatsModels** for statistical analysis
- **Scikit-learn** for machine learning

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Anas-Altaf/Weather_Electricity_Data_Analysis.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd Weather_Electricity_Data_Analysis
   ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Directory Structure
- **`raw/`**: Contains subfolders for raw data.
    - **`electricity_raw_data/`**: Electricity demand data files (CSV, JSON).
    - **`weather_raw_data/`**: Weather data files (CSV, JSON).
- **`app.py`**: Main Streamlit application file.
- **`requirements.txt`**: List of required Python packages.

**Note**: This project requires electricity demand and weather data in CSV or JSON format. Place the raw data files in the `raw/electricity_raw_data/` and `raw/weather_raw_data/` directories, respectively.

## Usage
After running the app, use the sidebar to navigate between different tasks:
- **Task 1: Data Loading and Integration**  
  View loaded data and initial statistics for electricity and weather datasets.
- **Task 2: Data Preprocessing**  
  Observe the preprocessed and merged dataset, including handling of missing values and feature engineering.
- **Task 3: Exploratory Data Analysis (EDA)**  
  Explore data through various plots (time series, histograms, boxplots) and advanced analyses (decomposition, stationarity tests).
- **Task 4: Outlier Detection and Handling**  
  Detect outliers in electricity demand and optionally cap them to improve data quality.
- **Task 5: Regression Modeling**  
  Train a linear regression model to predict electricity demand and evaluate its performance with metrics and visualizations.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.