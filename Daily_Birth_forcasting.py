# Importing necessary packages
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt

# Reading the dataset
df = pd.read_csv("daily-total-female-births.csv")
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' to datetime format
df.columns = ['ds', 'y']  # Rename columns to Prophet format

# Visualizing the data (before forecasting)
plt.plot(df['ds'], df['y'])
plt.title('Daily Female Births in 1959')  # Title for the plot
plt.xlabel('Date')  # X-axis label
plt.ylabel('Number of Female Births')  # Y-axis label
plt.grid(True)
plt.show()

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Creating the Prophet model
    m = Prophet(yearly_seasonality=True, 
                daily_seasonality=False, 
                changepoint_range=0.9, 
                changepoint_prior_scale=0.5, 
                seasonality_mode='multiplicative')

    # Fit the model
    m.fit(df)

# Create future dataframe (50 periods ahead, daily frequency)
future = m.make_future_dataframe(periods=50)  # Removed 'df' argument
forecast = m.predict(future)

m.plot_components(forecast)
#making prediction
m.plot(forecast)