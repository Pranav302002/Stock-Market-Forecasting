Apple Stock Market Forecasting Dashboard
Built using Streamlit, Prophet, ARIMA, SARIMA, and LSTM




🔍 Overview
This project provides an interactive and professional dashboard for analyzing and forecasting Apple stock prices using various time series models.

✅ Features
 Visualizations: Line chart, histogram, returns, and more

 Forecasting Models:

ARIMA

Prophet

SARIMA

LSTM (RNN)

 Streamlit Dashboard with model selection & time sliders

10+ Interactive Visualizations using Plotly


 
 Folder Structure
kotlin
Copy code
stock_market_forecasting/
├── data/
│   └── AAPL.csv
├── models/
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── sarima_model.py
│   └── lstm_model.py
├── app.py
├── requirements.txt
└── README.md

 
 Forecasting Output
Model	Type	Visualization
ARIMA	Classical	Plotly line chart
Prophet	Additive	Forecast bands
SARIMA	Seasonal ARIMA	Rolling forecast
LSTM	Deep Learning	Neural forecast line

 Sample Visualizations

 Dataset
Source: Apple Stock on Kaggle

Tech Stack
Python

Pandas, NumPy

Matplotlib, Plotly

Facebook Prophet, pmdarima, keras

Streamlit

Skills Gained
Time Series Forecasting

Model Evaluation (RMSE)

Deep Learning with LSTM

Streamlit UI/UX

💬 Feedback
Feel free to fork the repo, open issues, or drop suggestions!
