import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

# Function to download stock data from Yahoo Finance with retry logic
def get_stock_data(ticker, start_date, end_date):
    attempts = 5
    for attempt in range(attempts):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if stock_data.empty:
                print(f"No data available for {ticker}. Retrying...")
                time.sleep(5)
            else:
                return stock_data
        except Exception as e:
            print(f"Error: {e}. Retrying ({attempt + 1}/{attempts})...")
            time.sleep(5)  # wait for 5 seconds before retrying
    raise Exception(f"Failed to download data for {ticker} after {attempts} attempts.")

# Prepare features and target
def prepare_data(stock_data):
    stock_data['Prev Close'] = stock_data['Close'].shift(1)  # Previous day's close
    stock_data.dropna(inplace=True)  # Remove rows with NaN values

    X = stock_data[['Prev Close']]  # Feature: previous day's closing price
    y = stock_data['Close']        # Target: current day's closing price

    return X, y

# Main function to predict stock price
def predict_stock_price(ticker, start_date, end_date):
    # Get stock data from Yahoo Finance
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data.empty:
        st.error(f"No data available for {ticker}.")
        return None, None
    
    # Prepare the data for modeling
    X, y = prepare_data(stock_data)
    
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model with Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    
    # Ensure both y_test and y_pred are 1-dimensional arrays
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    
    # Create a DataFrame with predicted and actual prices
    predictions_df = pd.DataFrame({
        'Date': stock_data.index[-len(y_test):],
        'Actual Price': y_test,
        'Predicted Price': y_pred
    })
    
    # Display predicted stock prices in a table
    st.write("\nPredicted Stock Prices:")
    st.write(predictions_df)

    # Plot the actual vs predicted stock prices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data.index[-len(y_test):], y_test, color='blue', label='True Price')
    ax.plot(stock_data.index[-len(y_pred):], y_pred, color='red', label='Predicted Price')
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)
    
    return mse, predictions_df

# Streamlit app
def main():
    st.title("Stock Price Prediction")
    
    # User input for ticker and date range
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2023-01-01'))

    # Button to run the prediction
    if st.button("Predict Stock Prices"):
        mse, predictions_df = predict_stock_price(ticker, start_date, end_date)
        if mse is not None and predictions_df is not None:
            st.write(f"\nReturned MSE: {mse}")
            st.write(f"\nReturned Predictions DataFrame:")
            st.write(predictions_df)
        else:
            st.error("There was an error in the prediction process.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
