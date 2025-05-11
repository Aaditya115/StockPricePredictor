# Stock Price Predictor

## 📑 Summary
The **Stock Price Predictor** app allows users to input a stock ticker symbol, fetch the historical stock data, and predict the next day's stock closing price using a **Random Forest Regressor** model. The app also evaluates the model's performance using **Mean Absolute Error (MAE)** and visualizes the predicted vs. actual closing prices.

This project uses **Streamlit** for the interactive user interface and **Yahoo Finance (yfinance)** to fetch stock data. The model is trained on historical stock data, and users can see real-time predictions for the next day's stock price.

## 📚 Overview
This app allows users to:
- **Input a Stock Ticker**: Enter a stock ticker symbol to fetch its historical data.
- **Prediction**: Predict the next day's closing price based on historical data.
- **Model Evaluation**: View the **Mean Absolute Error (MAE)** for the model’s predictions on a test set.
- **Visualization**: See a plot comparing the actual vs. predicted stock prices.
