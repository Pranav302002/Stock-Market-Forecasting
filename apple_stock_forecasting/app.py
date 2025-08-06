

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
import os

#  UI Setup 
st.set_page_config(page_title="Apple Stock Forecasting", layout="wide")
st.title(" Apple Stock Forecasting Dashboard")
st.markdown("Forecast Apple stock using **ARIMA**, **Prophet**, **SARIMA**, and **LSTM**")

#  Load Data 
DATA_PATH = "data/AAPL.csv"
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

#  Preprocessing 
df['Daily Return'] = df['Close'].pct_change()
df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()

# -- Sidebar--
model_choice = st.sidebar.selectbox("🔍 Select Forecasting Model", ["ARIMA", "Prophet", "SARIMA", "LSTM"])
forecast_days = st.sidebar.slider("📅 Forecast Days", 30, 365, 90)

#  Visual 1 ------------------------
st.subheader("1. Closing Price Over Time")
st.plotly_chart(px.line(df, x='Date', y='Close', title="Closing Price"))

# ------------------------ Visual 2 ------------------------
st.subheader("2. Daily Returns Histogram")
fig2, ax2 = plt.subplots()
df['Daily Return'].dropna().hist(bins=50, ax=ax2)
st.pyplot(fig2)

# ------------------------ Visual 3 ------------------------
st.subheader("3. Cumulative Returns")
st.line_chart(df.set_index("Date")["Cumulative Return"])

# ------------------------ Forecasting ------------------------
st.header(f"📊 {model_choice} Forecast for {forecast_days} Days")

# Helper to plot forecast
def plot_forecast(hist_dates, hist_values, future_dates, forecasts, title):
    fig = px.line(title=title)
    fig.add_scatter(x=hist_dates, y=hist_values, mode='lines', name='Historical')
    fig.add_scatter(x=future_dates, y=forecasts, mode='lines', name='Forecast')
    st.plotly_chart(fig)

# ------------------------ ARIMA ------------------------
if model_choice == "ARIMA":
    series = df.set_index("Date")['Close'].dropna()
    model = auto_arima(series, seasonal=False)
    forecast = model.predict(n_periods=forecast_days)
    future_dates = pd.date_range(start=series.index[-1], periods=forecast_days+1, freq='B')[1:]
    plot_forecast(series.index, series, future_dates, forecast, "ARIMA Forecast")

    st.metric("ARIMA RMSE", f"{math.sqrt(mean_squared_error(series, model.predict_in_sample())):.2f}")

# ------------------------ Prophet ------------------------
elif model_choice == "Prophet":
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    plot_forecast(prophet_df['ds'], prophet_df['y'], forecast['ds'], forecast['yhat'], "Prophet Forecast")

    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# ------------------------ SARIMA ------------------------
elif model_choice == "SARIMA":
    sarima_series = df.set_index("Date")["Close"].dropna()
    sarima_model = SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_result = sarima_model.fit(disp=False)
    sarima_forecast = sarima_result.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=sarima_series.index[-1], periods=forecast_days+1, freq='B')[1:]
    plot_forecast(sarima_series.index, sarima_series, future_dates, sarima_forecast, "SARIMA Forecast")

# ------------------------ LSTM ------------------------
elif model_choice == "LSTM":
    data = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    look_back = 60
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)-forecast_days):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Forecasting
    inputs = scaled_data[-look_back:].reshape(1, look_back, 1)
    lstm_forecast = []
    for _ in range(forecast_days):
        pred = model.predict(inputs, verbose=0)
        lstm_forecast.append(pred[0,0])
        inputs = np.append(inputs[:,1:,:], [[[pred[0,0]]]], axis=1)

    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1,1)).flatten()
    last_date = df['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='B')[1:]

    plot_forecast(df['Date'], df['Close'], forecast_dates, lstm_forecast, "LSTM Forecast")

# ------------------------ Visuals 4-10 ------------------------
st.subheader("4. Volume Over Time")
st.plotly_chart(px.area(df, x='Date', y='Volume'))

st.subheader("5. Open vs Close")
st.plotly_chart(px.line(df, x='Date', y=['Open', 'Close']))

st.subheader("6. High-Low Spread")
df['Spread'] = df['High'] - df['Low']
st.plotly_chart(px.line(df, x='Date', y='Spread'))

st.subheader("7. 30-Day Moving Average")
df['MA30'] = df['Close'].rolling(window=30).mean()
st.line_chart(df.set_index("Date")[["Close", "MA30"]])

st.subheader("8. 7-Day Volatility")
df['Volatility'] = df['Daily Return'].rolling(window=7).std()
st.line_chart(df.set_index("Date")["Volatility"])

st.subheader("9. Log Returns")
df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
st.line_chart(df.set_index("Date")["Log Return"])

st.subheader("10. Candlestick Chart")
import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
st.plotly_chart(fig)
