# --------------------------------------------
# ARIMA Forecasting with Real Economic Data
# Inflation Time Series (India-style)
# --------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------------------
# Load Data
# --------------------------------------------
data = pd.read_csv("inflation_india.csv")

# Convert year to datetime index
data["year"] = pd.to_datetime(data["year"], format="%Y")
data.set_index("year", inplace=True)

# --------------------------------------------
# Plot Original Series
# --------------------------------------------
plt.figure()
plt.plot(data["inflation"])
plt.title("Inflation Rate Over Time")
plt.ylabel("Inflation (%)")
plt.xlabel("Year")
plt.show()

# --------------------------------------------
# Stationarity Test (ADF)
# --------------------------------------------
adf_test = adfuller(data["inflation"])
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])

# --------------------------------------------
# Fit ARIMA Model
# ARIMA(1,1,1) is standard for macro data
# --------------------------------------------
model = ARIMA(data["inflation"], order=(1, 1, 1))
results = model.fit()

print(results.summary())

# --------------------------------------------
# Forecast Next 5 Years
# --------------------------------------------
forecast = results.forecast(steps=5)

# Create forecast index
forecast_years = pd.date_range(
    start=data.index[-1] + pd.DateOffset(years=1),
    periods=5,
    freq="Y"
)

forecast_series = pd.Series(forecast.values, index=forecast_years)

# --------------------------------------------
# Plot Forecast
# --------------------------------------------
plt.figure()
plt.plot(data["inflation"], label="Observed")
plt.plot(forecast_series, label="Forecast", linestyle="--")
plt.title("ARIMA Inflation Forecast")
plt.ylabel("Inflation (%)")
plt.xlabel("Year")
plt.legend()
plt.show()
