# Time Series Forecasting for Vehicle Sales

Week 3 task, Automotive Data Science Analyst internship (YuvaIntern).

## What this project does
1. Simulates 84 months (2019-2025) of vehicle sales with trend, seasonality and an irregular shock
2. Prepares the data: monthly index, missing values, outlier detection, log transform
3. Decomposes the series (STL) and tests stationarity (ADF)
4. Compares Seasonal Naive, Holt-Winters and SARIMA on a 12-month hold-out (MAE, RMSE, MAPE)
5. Checks residual diagnostics and forecasts 2026 with 95% intervals
6. Builds the Word report

## Run
    pip install -r requirements.txt
    python forecasting_vehicle_sales.py

## Outputs
raw_vehicle_sales.csv, clean_vehicle_sales.csv, forecast_2026.csv, fig1-fig7 PNG charts and Week3_Vehicle_Sales_Forecasting_Report.docx
