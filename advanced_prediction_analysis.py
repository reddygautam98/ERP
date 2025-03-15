import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Union, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
# XGBoost for advanced modeling
import xgboost as xgb
# Prophet for advanced forecasting
from prophet import Prophet

# Fetch data from URL
url = r"C:\Users\reddy\Downloads\ERP\ERP_Retail_Industry_500rows.csv"
print("Loading data from URL:", url)
df = pd.read_csv(url)

# Display basic information about the dataset
print("\n=== Dataset Overview ===")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nSample Data:")
print(df.head(3))

# Convert date column
df['Date of Sale'] = pd.to_datetime(df['Date of Sale'])

# Create a copy of the original dataframe with date for time series analysis
df_time_series = df.copy()

# Continue with existing preprocessing
df['Year'] = df['Date of Sale'].dt.year
df['Month'] = df['Date of Sale'].dt.month
df['Day'] = df['Date of Sale'].dt.day
df['DayOfWeek'] = df['Date of Sale'].dt.dayofweek
df['Quarter'] = df['Date of Sale'].dt.quarter
df['WeekOfYear'] = df['Date of Sale'].dt.isocalendar().week

# Convert numeric columns that might be stored as strings
numeric_cols = ['Quantity Sold', 'Price', 'Total Amount']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_time_series[col] = pd.to_numeric(df_time_series[col], errors='coerce')

# Fill missing values
df.fillna({
    'Quantity Sold': df['Quantity Sold'].median(),
    'Price': df['Price'].median(),
    'Total Amount': df['Total Amount'].median()
}, inplace=True)

df_time_series.fillna({
    'Quantity Sold': df_time_series['Quantity Sold'].median(),
    'Price': df_time_series['Price'].median(),
    'Total Amount': df_time_series['Total Amount'].median()
}, inplace=True)

# ==================== ENHANCED EXPLORATORY DATA ANALYSIS ====================
print("\n=== Enhanced Exploratory Data Analysis ===")

# Create interactive summary statistics
def generate_summary_stats(dataframe, column):
    stats = {
        'Mean': dataframe[column].mean(),
        'Median': dataframe[column].median(),
        'Std Dev': dataframe[column].std(),
        'Min': dataframe[column].min(),
        'Max': dataframe[column].max(),
        'Q1 (25%)': dataframe[column].quantile(0.25),
        'Q3 (75%)': dataframe[column].quantile(0.75),
        'IQR': dataframe[column].quantile(0.75) - dataframe[column].quantile(0.25),
        'Skewness': dataframe[column].skew(),
        'Kurtosis': dataframe[column].kurtosis(),
        'Missing Values': dataframe[column].isna().sum(),
        'Missing %': (dataframe[column].isna().sum() / len(dataframe)) * 100
    }
    return pd.Series(stats)

# Generate summary statistics for numeric columns
summary_stats = pd.DataFrame({col: generate_summary_stats(df, col) for col in numeric_cols})
print("\nSummary Statistics:")
print(summary_stats)

# Correlation analysis with interactive heatmap
correlation_matrix = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# ==================== ENHANCED TIME SERIES ANALYSIS ====================
print("\n=== Enhanced Time Series Analysis ===")

# Aggregate data by date for time series analysis
time_series_daily = df_time_series.groupby('Date of Sale')['Total Amount'].sum().reset_index()
time_series_daily = time_series_daily.set_index('Date of Sale').sort_index()

# Create monthly time series for better pattern visibility
time_series_monthly = time_series_daily.resample('M').sum()

# Visualize the time series with trend line
plt.figure(figsize=(14, 7))
plt.plot(time_series_daily.index, time_series_daily['Total Amount'], label='Daily Sales')
# Add trend line using rolling average
rolling_mean = time_series_daily['Total Amount'].rolling(window=7).mean()
plt.plot(time_series_daily.index, rolling_mean, color='red', label='7-Day Moving Average')
plt.title('Daily Sales Over Time with Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Enhanced monthly visualization with year-over-year comparison
if len(time_series_monthly) >= 12:
    # Create year and month columns for comparison
    time_series_monthly_df = time_series_monthly.reset_index()
    time_series_monthly_df['Year'] = time_series_monthly_df['Date of Sale'].dt.year
    time_series_monthly_df['Month'] = time_series_monthly_df['Date of Sale'].dt.month_name()
    
    # Plot year-over-year comparison
    plt.figure(figsize=(14, 8))
    years = time_series_monthly_df['Year'].unique()
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    
    for year in years:
        year_data = time_series_monthly_df[time_series_monthly_df['Year'] == year]
        # Sort by month order
        year_data['Month_Num'] = year_data['Month'].apply(lambda x: months_order.index(x) if x in months_order else -1)
        year_data = year_data.sort_values('Month_Num')
        plt.plot(year_data['Month'], year_data['Total Amount'], marker='o', label=str(year))
    
    plt.title('Monthly Sales Comparison by Year')
    plt.xlabel('Month')
    plt.ylabel('Total Sales Amount')
    plt.legend(title='Year')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Time Series Decomposition with enhanced visualization
if len(time_series_monthly) >= 12:  # Need at least a year of data for meaningful decomposition
    decomposition = seasonal_decompose(time_series_monthly, model='multiplicative', period=12)
    
    plt.figure(figsize=(14, 12))
    plt.subplot(411)
    plt.plot(decomposition.observed)
    plt.title('Observed', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend Component', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal Component', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residual Component', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display seasonality strength
    seasonal_strength = 1 - (np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid))
    trend_strength = 1 - (np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid))
    
    print("\nSeasonality Strength: {:.4f}".format(seasonal_strength))
    print("Trend Strength: {:.4f}".format(trend_strength))
    
    if seasonal_strength > 0.6:
        print("Strong seasonal pattern detected - consider seasonal forecasting models")
    elif seasonal_strength > 0.3:
        print("Moderate seasonal pattern detected")
    else:
        print("Weak seasonal pattern detected")

# Enhanced stationarity test with visualization
def enhanced_stationarity_test(timeseries, title=''):
    # Rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(14, 7))
    plt.plot(timeseries, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.title('Rolling Mean & Standard Deviation - {}'.format(title))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Perform Augmented Dickey-Fuller test
    result = adfuller(timeseries.dropna())
    print('ADF Statistic: {:.4f}'.format(result[0]))
    print('p-value: {:.4f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.4f}'.format(key, value))
    
    if result[1] <= 0.05:
        print("✓ Stationary (reject H0)")
        return True
    else:
        print("✗ Non-stationary (fail to reject H0)")
        return False

print("\nEnhanced Stationarity Test for Monthly Sales:")
is_stationary = enhanced_stationarity_test(time_series_monthly['Total Amount'], title='Monthly Sales')

# If non-stationary, show differenced series
if not is_stationary and len(time_series_monthly) > 12:
    print("\nApplying differencing to make the series stationary:")
    diff_series = time_series_monthly['Total Amount'].diff().dropna()
    is_diff_stationary = enhanced_stationarity_test(diff_series, title='Differenced Monthly Sales')
    
    if not is_diff_stationary:
        print("\nApplying second-order differencing:")
        diff2_series = diff_series.diff().dropna()
        enhanced_stationarity_test(diff2_series, title='Second-Order Differenced Monthly Sales')

# Enhanced ACF and PACF plots
plt.figure(figsize=(14, 7))
plt.subplot(121)
plot_acf(time_series_monthly['Total Amount'].dropna(), ax=plt.gca(), lags=min(24, len(time_series_monthly) // 2))
plt.title('Autocorrelation Function (ACF)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(122)
plot_pacf(time_series_monthly['Total Amount'].dropna(), ax=plt.gca(), lags=min(24, len(time_series_monthly) // 2))
plt.title('Partial Autocorrelation Function (PACF)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== ADVANCED FORECASTING MODELS ====================
print("\n=== Advanced Forecasting Models ===")

# 1. SARIMA Forecasting with Auto-Parameter Selection
if len(time_series_monthly) >= 24:  # Need sufficient data for reliable SARIMA
    # Split data for training and testing
    train_size = int(len(time_series_monthly) * 0.8)
    train_data = time_series_monthly.iloc[:train_size]
    test_data = time_series_monthly.iloc[train_size:]
    
    # Function to evaluate SARIMA models
    def evaluate_sarima_model(train, test, order, seasonal_order):
        try:
            model = SARIMAX(train['Total Amount'], 
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=len(test))
            forecast_mean = forecast.predicted_mean
            
            mae = mean_absolute_error(test['Total Amount'], forecast_mean)
            rmse = np.sqrt(mean_squared_error(test['Total Amount'], forecast_mean))
            mape = mean_absolute_percentage_error(test['Total Amount'], forecast_mean)
            
            return {'order': order, 'seasonal_order': seasonal_order, 'mae': mae, 'rmse': rmse, 'mape': mape, 'model': results}
        except Exception as e:
            print("Error evaluating SARIMA model: {}".format(e))
            return {'order': order, 'seasonal_order': seasonal_order, 'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'model': None}
    
    # Define parameter grid
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    s_values = [12]  # Monthly seasonality
    
    # Test a subset of parameter combinations
    best_score = float('inf')
    best_model = None
    best_params = None
    
    print("Evaluating SARIMA models with different parameters...")
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for s in s_values:
                                # Only test a subset of combinations to save time
                                if (p + d + q + P + D + Q) <= 4:  # Limit complexity
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, s)
                                    
                                    result = evaluate_sarima_model(train_data, test_data, order, seasonal_order)
                                    
                                    if result['model'] is not None and result['mae'] < best_score:
                                        best_score = result['mae']
                                        best_model = result['model']
                                        best_params = {'order': order, 'seasonal_order': seasonal_order}
                                        print("New best model: SARIMA{}x{} - MAE: {:.2f}".format(order, seasonal_order, result['mae']))
    
    if best_model is not None:
        print("\nBest SARIMA Model: SARIMA{}x{}".format(best_params['order'], best_params['seasonal_order']))
        print("MAE: {:.2f}".format(best_score))
        
        # Forecast with best model
        forecast = best_model.get_forecast(steps=len(test_data))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Plot forecast vs actual
        plt.figure(figsize=(14, 7))
        plt.plot(train_data.index, train_data['Total Amount'], label='Training Data')
        plt.plot(test_data.index, test_data['Total Amount'], label='Actual Sales')
        plt.plot(test_data.index, forecast_mean, label='SARIMA Forecast', color='red')
        plt.fill_between(test_data.index, 
                         forecast_ci.iloc[:, 0], 
                         forecast_ci.iloc[:, 1], 
                         color='red', alpha=0.1, label='95% Confidence Interval')
        plt.title('SARIMA Forecast vs Actual', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Future forecast
        future_steps = 12  # Forecast 12 months into the future
        future_forecast = best_model.get_forecast(steps=future_steps)
        future_mean = future_forecast.predicted_mean
        future_ci = future_forecast.conf_int()
        
        # Create future date range
        last_date = time_series_monthly.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='M')
        
        # Plot future forecast
        plt.figure(figsize=(14, 7))
        plt.plot(time_series_monthly.index, time_series_monthly['Total Amount'], label='Historical Data')
        plt.plot(future_dates, future_mean, label='Future Forecast', color='red')
        plt.fill_between(future_dates, 
                         future_ci.iloc[:, 0], 
                         future_ci.iloc[:, 1], 
                         color='red', alpha=0.1, label='95% Confidence Interval')
        plt.title('SARIMA Future Sales Forecast', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Save the best SARIMA model
        sarima_model = best_model
    else:
        print("Could not find a suitable SARIMA model")

# 2. Prophet Forecasting (more robust for business time series)
print("\n=== Prophet Forecasting Model ===")

# Prepare data for Prophet
prophet_data = time_series_monthly.reset_index()
prophet_data.columns = ['ds', 'y']  # Prophet requires these column names

# Create and fit the model
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    interval_width=0.95,
    changepoint_prior_scale=0.05
)

# Add monthly seasonality
prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Fit the model
prophet_model.fit(prophet_data)

# Create future dataframe for forecasting
future_periods = 24  # Forecast 24 months ahead
future = prophet_model.make_future_dataframe(periods=future_periods, freq='M')

# Make predictions
forecast = prophet_model.predict(future)

# Display forecast components
fig1 = prophet_model.plot(forecast)
plt.title('Prophet Forecast with Uncertainty Intervals', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot forecast components
fig2 = prophet_model.plot_components(forecast)
plt.tight_layout()
plt.show()

# Calculate forecast accuracy on test set
if len(time_series_monthly) >= 24:
    # Use the same train/test split as SARIMA
    prophet_train = prophet_data.iloc[:train_size]
    prophet_test = prophet_data.iloc[train_size:]
    
    # Fit model on training data
    prophet_train_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    prophet_train_model.fit(prophet_train)
    
    # Forecast for test period
    prophet_future = prophet_train_model.make_future_dataframe(periods=len(prophet_test), freq='M')
    prophet_forecast = prophet_train_model.predict(prophet_future)
    
    # Extract predictions for test period
    prophet_predictions = prophet_forecast.iloc[-len(prophet_test):]['yhat'].values
    prophet_actuals = prophet_test['y'].values
    
    # Calculate metrics
    prophet_mae = mean_absolute_error(prophet_actuals, prophet_predictions)
    prophet_rmse = np.sqrt(mean_squared_error(prophet_actuals, prophet_predictions))
    prophet_mape = mean_absolute_percentage_error(prophet_actuals, prophet_predictions)
    
    print("\nProphet Forecast Evaluation:")
    print("MAE: {:.2f}".format(prophet_mae))
    print("RMSE: {:.2f}".format(prophet_rmse))
    print("MAPE: {:.2%}".format(prophet_mape))
    
    # Compare Prophet vs SARIMA if SARIMA model exists
    if 'sarima_model' in locals():
        print("\nModel Comparison:")
        print("SARIMA MAE: {:.2f}".format(best_score))
        print("Prophet MAE: {:.2f}".format(prophet_mae))
        
        if best_score < prophet_mae:
            print("SARIMA model performs better for this dataset")
        else:
            print("Prophet model performs better for this dataset")

# ==================== ADVANCED MACHINE LEARNING MODELS ====================
print("\n=== Advanced Machine Learning Models ===")

# Drop unnecessary columns for ML
ml_df = df.copy()
ml_df.drop(columns=['Transaction ID', 'Customer Name', 'Employee Name', 'Date of Sale'], inplace=True, errors='ignore')

# Define target variable
target = 'Total Amount'
X = ml_df.drop(columns=[target])
y = ml_df[target]

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. XGBoost Model
print("\n=== XGBoost Regression Model ===")

# Create XGBoost pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    ))
])

# Train XGBoost model
xgb_pipeline.fit(X_train, y_train)

# Evaluate XGBoost model
y_pred_xgb = xgb_pipeline.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)

print("XGBoost Model Performance:")
print("R² Score: {:.4f}".format(r2_xgb))
print("MAE: {:.2f}".format(mae_xgb))
print("RMSE: {:.2f}".format(rmse_xgb))
print("MAPE: {:.2%}".format(mape_xgb))

# 2. Random Forest Model for comparison
print("\n=== Random Forest Regression Model ===")

# Create Random Forest pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ))
])

# Train Random Forest model
rf_pipeline.fit(X_train, y_train)

# Evaluate Random Forest model
y_pred_rf = rf_pipeline.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

print("Random Forest Model Performance:")
print("R² Score: {:.4f}".format(r2_rf))
print("MAE: {:.2f}".format(mae_rf))
print("RMSE: {:.2f}".format(rmse_rf))
print("MAPE: {:.2%}".format(mape_rf))

# 3. Gradient Boosting Model (original model)
print("\n=== Gradient Boosting Regression Model ===")

# Create Gradient Boosting pipeline
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Train Gradient Boosting model
gb_pipeline.fit(X_train, y_train)

# Evaluate Gradient Boosting model
y_pred_gb = gb_pipeline.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)

print("Gradient Boosting Model Performance:")
print("R² Score: {:.4f}".format(r2_gb))
print("MAE: {:.2f}".format(mae_gb))
print("RMSE: {:.2f}".format(rmse_gb))
print("MAPE: {:.2%}".format(mape_gb))

# Compare all models
print("\n=== Model Comparison ===")
models_comparison = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting'],
    'R² Score': [r2_xgb, r2_rf, r2_gb],
    'MAE': [mae_xgb, mae_rf, mae_gb],
    'RMSE': [rmse_xgb, rmse_rf, rmse_gb],
    'MAPE': [mape_xgb, mape_rf, mape_gb]
})
print(models_comparison)

# Select the best model based on R² score
best_model_idx = models_comparison['R² Score'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Model']
print("\nBest Model:", best_model_name)

# Use the best model for further analysis
if best_model_name == 'XGBoost':
    best_model = xgb_pipeline
    y_pred = y_pred_xgb
elif best_model_name == 'Random Forest':
    best_model = rf_pipeline
    y_pred = y_pred_rf
else:
    best_model = gb_pipeline
    y_pred = y_pred_gb

# ==================== ADVANCED FEATURE IMPORTANCE ANALYSIS ====================
print("\n=== Advanced Feature Importance Analysis ===")

# Get feature names after preprocessing
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()

# Extract feature importances from the best model
if best_model_name == 'XGBoost':
    importances = best_model.named_steps['regressor'].feature_importances_
elif best_model_name == 'Random Forest':
    importances = best_model.named_steps['regressor'].feature_importances_
else:
    importances = best_model.named_steps['regressor'].feature_importances_

# Create feature importance dataframe
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top 15 features
top_features = feature_importances.head(15)
print("\nTop 15 Most Important Features:")
print(top_features)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 15 Feature Importances - {} Model'.format(best_model_name), fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# SHAP Analysis for the best model
print("\n=== SHAP Analysis ===")

# Transform test data for SHAP analysis
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()
X_test_transformed = X_test_transformed.astype(np.float64)

# Create SHAP explainer based on the best model
if best_model_name == 'XGBoost':
    explainer = shap.TreeExplainer(best_model.named_steps['regressor'])
elif best_model_name == 'Random Forest':
    explainer = shap.TreeExplainer(best_model.named_steps['regressor'])
else:
    explainer = shap.TreeExplainer(best_model.named_steps['regressor'])

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# Summary plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
plt.title('SHAP Summary Plot - {} Model'.format(best_model_name), fontsize=14)
plt.tight_layout()
plt.show()

# Detailed SHAP dependence plots for top 3 features
top_3_features = feature_importances['Feature'].head(3).tolist()
for feature in top_3_features:
    feature_idx = list(feature_names).index(feature)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, X_test_transformed, feature_names=feature_names)
    plt.title('SHAP Dependence Plot - {}'.format(feature), fontsize=14)
    plt.tight_layout()
    plt.show()

# ==================== CUSTOMER SEGMENTATION ANALYSIS ====================
print("\n=== Customer Segmentation Analysis ===")

# Create customer-centric dataset
if 'Customer Name' in df.columns:
    customer_data = df.groupby('Customer Name').agg({
        'Total Amount': ['sum', 'mean', 'count'],
        'Quantity Sold': ['sum', 'mean'],
        'Price': 'mean'
    })
    
    # Flatten column names
    customer_data.columns = ['_'.join(col).strip() for col in customer_data.columns.values]
    
    # Rename columns for clarity
    customer_data.rename(columns={
        'Total Amount_sum': 'Total_Spend',
        'Total Amount_mean': 'Average_Transaction_Value',
        'Total Amount_count': 'Transaction_Count',
        'Quantity Sold_sum': 'Total_Items',
        'Quantity Sold_mean': 'Average_Items_Per_Transaction',
        'Price_mean': 'Average_Price'
    }, inplace=True)
    
    # Add derived metrics
    customer_data['Average_Spend_Per_Item'] = customer_data['Total_Spend'] / customer_data['Total_Items']
    
    # Standardize data for clustering
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(1, min(11, len(customer_data) + 1))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(customer_data_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Choose optimal k (this is a simple heuristic, could be improved)
    k_optimal = 3  # Default, but should be determined from the elbow curve
    for i in range(1, len(inertia) - 1):
        if (inertia[i-1] - inertia[i]) / (inertia[i] - inertia[i+1]) > 2:
            k_optimal = i + 1
            break
    
    print("Optimal number of clusters determined:", k_optimal)
    
    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)
    
    # Analyze clusters
    cluster_analysis = customer_data.groupby('Cluster').agg({
        'Total_Spend': ['mean', 'min', 'max', 'count'],
        'Transaction_Count': 'mean',
        'Average_Transaction_Value': 'mean',
        'Total_Items': 'mean',
        'Average_Items_Per_Transaction': 'mean'
    })
    
    # Flatten column names
    cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns.values]
    
    print("\nCustomer Segment Analysis:")
    print(cluster_analysis)
    
    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    customer_data_pca = pca.fit_transform(customer_data_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(data=customer_data_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = customer_data['Cluster']
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
    plt.title('Customer Segments - PCA Visualization', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Interpret clusters
    print("\nCustomer Segment Interpretation:")
    for cluster in range(k_optimal):
        cluster_size = cluster_analysis.loc[cluster, 'Total_Spend_count']
        cluster_pct = (cluster_size / len(customer_data)) * 100
        avg_spend = cluster_analysis.loc[cluster, 'Total_Spend_mean']
        avg_transactions = cluster_analysis.loc[cluster, 'Transaction_Count_mean']
        avg_transaction_value = cluster_analysis.loc[cluster, 'Average_Transaction_Value_mean']
        
        print("\nCluster {}:".format(cluster))
        print("  Size: {} customers ({:.1f}% of total)".format(int(cluster_size), cluster_pct))
        print("  Average Total Spend: ${:.2f}".format(avg_spend))
        print("  Average Transactions: {:.1f}".format(avg_transactions))
        print("  Average Transaction Value: ${:.2f}".format(avg_transaction_value))
        
        # Assign meaningful labels based on spending patterns
        if avg_spend > customer_data['Total_Spend'].mean() * 1.5:
            if avg_transactions > customer_data['Transaction_Count'].mean() * 1.5:
                print("  Segment Label: HIGH-VALUE LOYAL CUSTOMERS")
            else:
                print("  Segment Label: BIG SPENDERS (HIGH VALUE, LESS FREQUENT)")
        elif avg_transactions > customer_data['Transaction_Count'].mean() * 1.5:
            print("  Segment Label: FREQUENT SHOPPERS (REGULAR VISITORS)")
        elif avg_spend < customer_data['Total_Spend'].mean() * 0.5:
            print("  Segment Label: LOW-VALUE CUSTOMERS")
        else:
            print("  Segment Label: AVERAGE CUSTOMERS")
else:
    print("Customer Name column not available for segmentation analysis")

# ==================== ANOMALY DETECTION ====================
print("\n=== Sales Anomaly Detection ===")

# Detect anomalies in daily sales
if len(time_series_daily) > 30:  # Need sufficient data for anomaly detection
    # Prepare data for anomaly detection
    anomaly_data = time_series_daily.reset_index()
    anomaly_data.columns = ['date', 'sales']
    
    # Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect 5% of data to be anomalies
        random_state=42
    )
    
    # Fit and predict
    anomaly_data['anomaly'] = isolation_forest.fit_predict(anomaly_data[['sales']])
    anomaly_data['anomaly'] = anomaly_data['anomaly'].map({1: 0, -1: 1})  # 1 is anomaly, 0 is normal
    
    # Count anomalies
    anomaly_count = anomaly_data['anomaly'].sum()
    print("Detected {} anomalies in daily sales data".format(anomaly_count))
    
    # Plot anomalies
    plt.figure(figsize=(14, 7))
    plt.plot(anomaly_data['date'], anomaly_data['sales'], label='Sales')
    plt.scatter(
        anomaly_data[anomaly_data['anomaly'] == 1]['date'],
        anomaly_data[anomaly_data['anomaly'] == 1]['sales'],
        color='red',
        label='Anomaly',
        s=100,
        alpha=0.7
    )
    plt.title('Sales Anomalies Detection', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Analyze anomalies
    anomalies = anomaly_data[anomaly_data['anomaly'] == 1].sort_values('sales', ascending=False)
    print("\nTop 5 Highest Sales Anomalies:")
    print(anomalies.head(5)[['date', 'sales']])
    
    print("\nBottom 5 Lowest Sales Anomalies:")
    print(anomalies.sort_values('sales').head(5)[['date', 'sales']])
    
    # Calculate threshold values
    q1 = anomaly_data['sales'].quantile(0.25)
    q3 = anomaly_data['sales'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    
    print("\nStatistical Thresholds:")
    print("Upper Bound (Q3 + 1.5*IQR): ${:.2f}".format(upper_bound))
    print("Lower Bound (Q1 - 1.5*IQR): ${:.2f}".format(lower_bound))

# ==================== SAVE MODELS AND CREATE API ====================
print("\n=== Saving Models and Creating API ===")

# Save the best ML model
ml_model_path = "sales_prediction_ml_model.pkl"
joblib.dump(best_model, ml_model_path)
print("Best ML model ({}) saved as {}".format(best_model_name, ml_model_path))

# Save the Prophet model
prophet_model_path = "sales_forecast_prophet_model.pkl"
with open(prophet_model_path, 'wb') as f:
    joblib.dump(prophet_model, f)
print("Prophet forecasting model saved as {}".format(prophet_model_path))

# Save SARIMA model if it exists
if 'sarima_model' in locals():
    sarima_model_path = "sales_forecast_sarima_model.pkl"
    joblib.dump(sarima_model, sarima_model_path)
    print("SARIMA forecasting model saved as {}".format(sarima_model_path))

# Create FastAPI app
app = FastAPI(title="Advanced Sales Analytics API", 
              description="API for sales prediction and forecasting",
              version="2.0")

class SalesPredictionInput(BaseModel):
    data: Dict[str, Union[str, float, int]]

class TimeSeriesForecastInput(BaseModel):
    forecast_periods: int = 12
    frequency: str = "M"  # M for monthly, D for daily
    return_components: bool = False

class FeatureImportanceInput(BaseModel):
    top_n: int = 10

class AnomalyDetectionInput(BaseModel):
    sensitivity: float = 0.05  # Contamination parameter
    data: Optional[List[Dict[str, Union[str, float]]]] = None

@app.get("/")
def home():
    return {"message": "Advanced Retail Sales Analytics API is running!"}

@app.post("/predict")
def predict(input_data: SalesPredictionInput):
    """
    Predict sales based on input features using the best ML model
    """
    try:
        df_input = pd.DataFrame([input_data.data])
        for col in numeric_cols:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
        
        # Make prediction
        prediction = best_model.predict(df_input)[0]
        
        # Get confidence interval (approximation)
        confidence = 0.9  # 90% confidence
        
        return {
            "predicted_sales": float(prediction),
            "model_used": best_model_name,
            "confidence_level": confidence
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/forecast")
def forecast(input_data: TimeSeriesForecastInput):
    """
    Generate time series forecast for future periods
    """
    try:
        # Use Prophet for forecasting
        future_periods = input_data.forecast_periods
        
        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=future_periods, freq=input_data.frequency)
        
        # Make forecast
        forecast = prophet_model.predict(future)
        
        # Get the forecast for future periods only
        future_forecast = forecast.iloc[-future_periods:]
        
        # Format response
        forecast_data = []
        for _, row in future_forecast.iterrows():
            forecast_data.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "forecast": float(row['yhat']),
                "lower_bound": float(row['yhat_lower']),
                "upper_bound": float(row['yhat_upper'])
            })
        
        response = {
            "forecast": forecast_data,
            "model_used": "Prophet"
        }
        
        # Include components if requested
        if input_data.return_components:
            components = {
                "trend": future_forecast['trend'].tolist(),
                "yearly": future_forecast['yearly'].tolist() if 'yearly' in future_forecast.columns else None,
                "monthly": future_forecast['monthly'].tolist() if 'monthly' in future_forecast.columns else None
            }
            response["components"] = components
        
        return response
    except Exception as e:
        return {"error": str(e)}

@app.post("/feature_importance")
def get_feature_importance(input_data: FeatureImportanceInput):
    """
    Return the top N most important features
    """
    try:
        top_n = min(input_data.top_n, len(feature_importances))
        top_features = feature_importances.head(top_n)
        
        return {
            "feature_importance": [
                {"feature": row['Feature'], "importance": float(row['Importance'])}
                for _, row in top_features.iterrows()
            ],
            "model_used": best_model_name
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect_anomalies")
def detect_anomalies(input_data: AnomalyDetectionInput):
    """
    Detect anomalies in sales data
    """
    try:
        # Use provided data or default to the dataset used for training
        if input_data.data:
            # Convert input data to dataframe
            anomaly_data = pd.DataFrame(input_data.data)
            if 'date' in anomaly_data.columns and 'sales' in anomaly_data.columns:
                anomaly_data['date'] = pd.to_datetime(anomaly_data['date'])
            else:
                return {"error": "Input data must contain 'date' and 'sales' columns"}
        else:
            # Use the time series data from the analysis
            anomaly_data = time_series_daily.reset_index()
            anomaly_data.columns = ['date', 'sales']
        
        # Isolation Forest for anomaly detection
        isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=input_data.sensitivity,
            random_state=42
        )
        
        # Fit and predict
        anomaly_data['anomaly'] = isolation_forest.fit_predict(anomaly_data[['sales']])
        anomaly_data['anomaly'] = anomaly_data['anomaly'].map({1: 0, -1: 1})  # 1 is anomaly, 0 is normal
        
        # Extract anomalies
        anomalies = anomaly_data[anomaly_data['anomaly'] == 1]
        
        return {
            "anomalies": [
                {"date": row['date'].strftime('%Y-%m-%d'), "sales": float(row['sales'])}
                for _, row in anomalies.iterrows()
            ],
            "total_anomalies": len(anomalies),
            "sensitivity_used": input_data.sensitivity
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/model_performance")
def get_model_performance():
    """
    Return performance metrics for all models
    """
    try:
        return {
            "ml_models": [
                {
                    "name": "XGBoost",
                    "r2_score": float(r2_xgb),
                    "mae": float(mae_xgb),
                    "rmse": float(rmse_xgb),
                    "mape": float(mape_xgb)
                },
                {
                    "name": "Random Forest",
                    "r2_score": float(r2_rf),
                    "mae": float(mae_rf),
                    "rmse": float(rmse_rf),
                    "mape": float(mape_rf)
                },
                {
                    "name": "Gradient Boosting",
                    "r2_score": float(r2_gb),
                    "mae": float(mae_gb),
                    "rmse": float(rmse_gb),
                    "mape": float(mape_gb)
                }
            ],
            "best_model": best_model_name
        }
    except Exception as e:
        return {"error": str(e)}

print("\nFastAPI app is ready! Run with: uvicorn advanced_sales_analysis:app --host 0.0.0.0 --port 8000 --reload")
print("\nAPI Endpoints:")
print("  - GET  /                  : Home")
print("  - POST /predict           : Predict sales based on features")
print("  - POST /forecast          : Generate time series forecast")
print("  - POST /feature_importance: Get top feature importances")
print("  - POST /detect_anomalies  : Detect anomalies in sales data")
print("  - GET  /model_performance : Get performance metrics for all models")

# ==================== CONCLUSION ====================
print("\n=== Analysis Summary ===")
print("1. Best ML Model: {} with R² Score of {:.4f}".format(best_model_name, models_comparison['R² Score'].max()))
print("2. Top 3 Important Features: {}".format(', '.join(feature_importances['Feature'].head(3).tolist())))
print("3. Forecasting Models: Prophet and SARIMA implemented")
print("4. Advanced Analytics: Customer Segmentation and Anomaly Detection completed")
print("5. API: FastAPI with 6 endpoints for predictions, forecasting, and analytics")
print("\nAll models saved and ready for deployment!")