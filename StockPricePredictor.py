import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st

# Streamlit UI Setup
st.title('Stock Price Predictor')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker', 'AAPL')

if ticker:
    # Fetch stock data for the user-provided ticker
    stock_data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

    if stock_data.empty:
        st.error(f"No data found for ticker {ticker}")
    else:
        # Show the historical stock data
        st.subheader(f"Historical Data for {ticker}")
        st.write(stock_data.tail())  # Show the last few rows of data

        # Feature Engineering: Use 'Open', 'High', 'Low', 'Close', 'Volume' to predict 'Close' next day
        stock_data['Target'] = stock_data['Close'].shift(-1)
        stock_data.dropna(inplace=True)

        # Prepare the features (X) and target (y)
        X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = stock_data['Target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, predictions)
        st.write(f"Mean Absolute Error (MAE) on test set: {mae:.2f}")

        # Plot the actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test, label='Actual', color='blue')
        plt.plot(y_test.index, predictions, label='Predicted', color='red')
        plt.title(f"Actual vs Predicted Stock Prices for {ticker}")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        st.pyplot(plt)

        # Predict the next day's closing price
        latest_data = stock_data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
        predicted_price = model.predict(latest_data)
        st.write(f"The predicted closing price for the next day is: ${predicted_price[0]:.2f}")
