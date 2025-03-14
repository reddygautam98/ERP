import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# New imports for time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error

# Load dataset from local file path
file_path = r"C:\Users\reddy\Downloads\ERP\ERP_Retail_Industry_500rows.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found!")

print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# Convert date column
df['Date of Sale'] = pd.to_datetime(df['Date of Sale'])

# Create a copy of the original dataframe with date for time series analysis
df_time_series = df.copy()

# Continue with existing preprocessing
df['Year'] = df['Date of Sale'].dt.year
df['Month'] = df['Date of Sale'].dt.month
df['Day'] = df['Date of Sale'].dt.day

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

# ==================== TIME SERIES ANALYSIS ====================
print("\n=== Time Series Analysis ===")

# Aggregate data by date for time series analysis
time_series_daily = df_time_series.groupby('Date of Sale')['Total Amount'].sum().reset_index()
time_series_daily = time_series_daily.set_index('Date of Sale').sort_index()

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_daily.index, time_series_daily['Total Amount'])
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create monthly time series for better pattern visibility
time_series_monthly = time_series_daily.resample('M').sum()

# Visualize monthly time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_monthly.index, time_series_monthly['Total Amount'], marker='o')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.tight_layout()
plt.show()

# Time Series Decomposition
if len(time_series_monthly) >= 12:  # Need at least a year of data for meaningful decomposition
    decomposition = seasonal_decompose(time_series_monthly, model='additive', period=12)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(decomposition.observed)
    plt.title('Observed')
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonality')
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residuals')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough monthly data for seasonal decomposition (need at least 12 months)")
    # Try with a smaller period if data is limited
    if len(time_series_monthly) >= 4:
        decomposition = seasonal_decompose(time_series_monthly, model='additive', period=min(len(time_series_monthly) // 2, 4))
        
        plt.figure(figsize=(12, 10))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Observed')
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonality')
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Residuals')
        plt.tight_layout()
        plt.show()

# Check stationarity with Augmented Dickey-Fuller test
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if result[1] <= 0.05:
        print("Stationary (reject H0)")
    else:
        print("Non-stationary (fail to reject H0)")

print("\nStationarity Test for Monthly Sales:")
check_stationarity(time_series_monthly['Total Amount'])

# ACF and PACF plots for time series modeling
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(time_series_monthly['Total Amount'].dropna(), ax=plt.gca(), lags=min(20, len(time_series_monthly) // 2))
plt.title('Autocorrelation Function')

plt.subplot(122)
plot_pacf(time_series_monthly['Total Amount'].dropna(), ax=plt.gca(), lags=min(20, len(time_series_monthly) // 2))
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# Time Series Forecasting with SARIMA
# Split data for training and testing
if len(time_series_monthly) >= 12:
    train_size = int(len(time_series_monthly) * 0.8)
    train_data = time_series_monthly.iloc[:train_size]
    test_data = time_series_monthly.iloc[train_size:]
    
    # Fit SARIMA model
    try:
        # Simple SARIMA model - parameters would ideally be tuned based on ACF/PACF
        model = SARIMAX(train_data['Total Amount'], 
                        order=(1, 1, 1),  # (p,d,q) - ARIMA parameters
                        seasonal_order=(1, 1, 1, 12),  # (P,D,Q,s) - Seasonal parameters
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        results = model.fit(disp=False)
        print(results.summary())
        
        # Forecast
        forecast_steps = len(test_data)
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Plot forecast vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data['Total Amount'], label='Training Data')
        plt.plot(test_data.index, test_data['Total Amount'], label='Actual Sales')
        plt.plot(test_data.index, forecast_mean, label='Forecasted Sales')
        plt.fill_between(test_data.index, 
                         forecast_ci.iloc[:, 0], 
                         forecast_ci.iloc[:, 1], 
                         color='k', alpha=0.1, label='95% Confidence Interval')
        plt.title('SARIMA Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Evaluate forecast
        mae = mean_absolute_error(test_data['Total Amount'], forecast_mean)
        mse = mean_squared_error(test_data['Total Amount'], forecast_mean)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test_data['Total Amount'], forecast_mean)
        
        print("\nTime Series Forecast Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2%}")
    except Exception as e:
        print(f"Error in SARIMA modeling: {e}")
        print("Proceeding with simpler time series analysis")
else:
    print("Not enough data for reliable time series forecasting (need at least 12 months)")

# Sales by Day of Week
df_time_series['DayOfWeek'] = df_time_series['Date of Sale'].dt.day_name()
day_of_week_sales = df_time_series.groupby('DayOfWeek')['Total Amount'].sum()

# Reorder days for better visualization
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_sales = day_of_week_sales.reindex(days_order)

plt.figure(figsize=(10, 6))
sns.barplot(x=day_of_week_sales.index, y=day_of_week_sales.values)
plt.title('Total Sales by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sales by Month
df_time_series['Month'] = df_time_series['Date of Sale'].dt.month_name()
month_sales = df_time_series.groupby('Month')['Total Amount'].sum()

# Reorder months for better visualization
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
month_sales = month_sales.reindex([m for m in months_order if m in month_sales.index])

plt.figure(figsize=(12, 6))
sns.barplot(x=month_sales.index, y=month_sales.values)
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==================== CONTINUE WITH ORIGINAL CODE ====================

# Drop unnecessary columns
df.drop(columns=['Transaction ID', 'Customer Name', 'Employee Name', 'Date of Sale'], inplace=True)

# Define target variable
target = 'Total Amount'
X = df.drop(columns=[target])
y = df[target]

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

# Machine Learning Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.1, 0.2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R2 Score: {r2}, MAE: {mae}, MSE: {mse}")

# Get feature names after preprocessing
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()
trained_regressor = grid_search.best_estimator_.named_steps['regressor']

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': trained_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# Residuals Distribution
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=20, kde=True)
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# SHAP Analysis
X_test_transformed = grid_search.best_estimator_.named_steps['preprocessor'].transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()
X_test_transformed = X_test_transformed.astype(np.float64)

explainer = shap.TreeExplainer(trained_regressor)
shap_values = explainer.shap_values(X_test_transformed)
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

# Save model
model_path = os.path.join(os.path.dirname(file_path), "sales_prediction_model.pkl")
joblib.dump(grid_search.best_estimator_, model_path)
print(f"Model saved as {model_path}")

# Save time series model if it was created
try:
    if 'results' in locals():
        ts_model_path = os.path.join(os.path.dirname(file_path), "time_series_model.pkl")
        joblib.dump(results, ts_model_path)
        print(f"Time series model saved as {ts_model_path}")
except Exception as e:
    print(f"Could not save time series model: {e}")

# FastAPI Deployment
app = FastAPI()

class SalesPredictionInput(BaseModel):
    data: Dict[str, Union[str, float, int]]

class TimeSeriesForecastInput(BaseModel):
    forecast_periods: int = 12
    frequency: str = "M"  # M for monthly, D for daily

@app.get("/")
def home():
    return {"message": "Retail Sales Prediction API is running!"}

@app.post("/predict")
def predict(input_data: SalesPredictionInput):
    df_input = pd.DataFrame([input_data.data])
    for col in numeric_cols:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
    model = joblib.load(model_path)
    prediction = model.predict(df_input)[0]
    return {"predicted_sales": float(prediction)}

@app.post("/forecast")
def forecast(input_data: TimeSeriesForecastInput):
    try:
        # Load time series model
        ts_model_path = os.path.join(os.path.dirname(file_path), "time_series_model.pkl")
        if os.path.exists(ts_model_path):
            ts_model = joblib.load(ts_model_path)
            
            # Generate forecast
            forecast = ts_model.get_forecast(steps=input_data.forecast_periods)
            forecast_mean = forecast.predicted_mean.tolist()
            forecast_ci = forecast.conf_int()
            
            # Create date range for forecast
            last_date = time_series_monthly.index[-1]
            if input_data.frequency == "M":
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                            periods=input_data.forecast_periods, 
                                            freq='M')
            else:
                future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), 
                                            periods=input_data.forecast_periods, 
                                            freq='D')
            
            # Format dates for response
            date_strings = [d.strftime('%Y-%m-%d') for d in future_dates]
            
            return {
                "forecast_dates": date_strings,
                "forecast_values": forecast_mean,
                "lower_bound": forecast_ci.iloc[:, 0].tolist(),
                "upper_bound": forecast_ci.iloc[:, 1].tolist()
            }
        else:
            return {"error": "Time series model not found. Run the analysis first."}
    except Exception as e:
        return {"error": f"Forecast error: {str(e)}"}

print("FastAPI app is ready! Run with: uvicorn script_name:app --host 0.0.0.0 --port 8000 --reload")

# Run the FastAPI app with: uvicorn advanced_prediction_analysis:app --host