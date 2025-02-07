import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

# Load the saved model
loaded_model = keras.models.load_model("sol_price_prediction_model.h5")

# Load the saved scaler
with open("sol_price_scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

# Streamlit app
st.set_page_config(layout="wide")
st.title("Stock Price Prediction App")

# Stock symbol input
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, SOL-USD):", "SOL-USD")

if st.button("Predict"):
    if stock_symbol:
        try:
            # 1. Get Data from Yahoo Finance
            data = yf.download(stock_symbol, start='2022-01-01')

            if data.empty:
                st.error("Invalid stock symbol or no data found.")
            else:
                # 2. Feature Engineering
                data['EMA'] = data['Close'].ewm(span=7, adjust=False).mean()
                data['STD'] = data['Close'].rolling(window=7).std()
                data['MA7'] = data['Close'].rolling(window=7).mean()
                data['MA21'] = data['Close'].rolling(window=21).mean()
                # Bollinger Bands
                data['Upper_Band'] = data['Close'].rolling(window=20).mean() + data['Close'].rolling(window=20).std() * 2
                data['Lower_Band'] = data['Close'].rolling(window=20).mean() - data['Close'].rolling(window=20).std() * 2

                # MACD
                data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()  # 9-day EMA of MACD
                data = data.dropna()

                features = ['Close', 'MACD']
                dataset = data[features].values

                # 3. Scaling
                scaled_data = loaded_scaler.transform(dataset)

                look_back = 7

                # 4. Prediction (using the loaded model)
                n_predictions = 7
                predictions = []

                temp_scaled_data = scaled_data.copy()

                for i in range(n_predictions):
                    last_lookback_days = temp_scaled_data[-look_back:]

                    X_pred = np.array([last_lookback_days])
                    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]))

                    predicted_data_scaled = loaded_model.predict(X_pred)

                    dummy_array_future = np.zeros((predicted_data_scaled.shape[0], dataset.shape[1]))
                    dummy_array_future[:, 0] = predicted_data_scaled[:, 0]
                    predicted_price = loaded_scaler.inverse_transform(dummy_array_future)[0, 0]
                    predictions.append(predicted_price)

                    dummy_row_scaled = np.zeros((1, temp_scaled_data.shape[1]))
                    dummy_row_scaled[0, 0] = predicted_data_scaled[0, 0]
                    temp_scaled_data = np.concatenate((temp_scaled_data, dummy_row_scaled), axis=0)

                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_predictions)
                # --- Original Plot (Last 7 Days) ---
                fig_7day, ax_7day = plt.subplots(figsize=(27, 10), dpi=150)

                ax_7day.plot(data['Close'].tail(30).index, data['Close'].tail(30).values,
                             label='Actual Price (Last 30 Days)')
                ax_7day.plot(future_dates, predictions, label='Predicted Price (Next 7 Days)', marker='o',
                             linestyle='--')

                # Increased fontsize here
                for x, y in zip(data['Close'].tail(30).index, data['Close'].tail(30).values):
                    ax_7day.text(x, y, f'{float(y):.2f}', ha='center', va='bottom', fontsize=14)  # Increased to 12

                for x, y in zip(future_dates, predictions):
                    ax_7day.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=14)  # Increased to 12

                ax_7day.set_xlabel('Date')
                ax_7day.set_ylabel('Price')
                ax_7day.set_title('Actual vs. Predicted Price (7 Days)')
                ax_7day.legend(fontsize=20)
                ax_7day.grid(True, alpha=0.7)
                fig_7day.tight_layout()
                st.pyplot(fig_7day)

                # --- New Combined Plot (Last 14 Days + Indicators) ---
                fig_combined, ax_combined = plt.subplots(figsize=(27, 10), dpi=150)

                # Plot actual values for the last 14 days
                ax_combined.plot(data.index[-21:], data['Close'][-21:], label='Actual Price (Last 14 Days)')

                # Plot predictions for the next 7 days
                ax_combined.plot(future_dates, predictions, label='Predicted Price (Next 7 Days)', marker='o',
                                 linestyle='--')

                # Plot Technical Indicators (for the last 14 days)
                ax_combined.plot(data.index[-21:], data['EMA'][-21:], label='EMA (7)', alpha=0.7)
                ax_combined.plot(data.index[-21:], data['MA7'][-21:], label='MA7', alpha=0.7)
                ax_combined.plot(data.index[-21:], data['MA21'][-21:], label='MA21', alpha=0.7)

                # Plot Bollinger Bands (for the last 14 days)
                ax_combined.plot(data.index[-21:], data['Upper_Band'][-21:], label='Upper Bollinger Band (20)',
                                 linestyle='--', alpha=0.7)
                ax_combined.plot(data.index[-21:], data['Lower_Band'][-21:], label='Lower Bollinger Band (20)',
                                 linestyle='--', alpha=0.7)

                ax_combined.set_xlabel('Date')
                ax_combined.set_ylabel('Price')
                ax_combined.set_title('Actual vs. Predicted Price (Last 14 Days & Next 7 Days) with Indicators')
                ax_combined.legend(fontsize=20)
                ax_combined.grid(True, alpha=0.7)
                fig_combined.tight_layout()
                st.pyplot(fig_combined)  # Display the combined plot

                # --- MACD Plot ---
                fig_macd, ax_macd = plt.subplots(figsize=(27, 8), dpi=150)

                ax_macd.plot(data.index[-21:], data['MACD'][-21:], label='MACD')
                ax_macd.plot(data.index[-21:], data['Signal'][-21:], label='Signal Line', linestyle='--')
                ax_macd.set_xlabel('Date')
                ax_macd.set_ylabel('MACD Value')
                ax_macd.set_title('MACD (Last 14 Days)')
                ax_macd.legend(fontsize=20)
                ax_macd.grid(True, alpha=0.7)
                fig_macd.tight_layout()
                st.pyplot(fig_macd)  # Display the MACD plot


        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.warning("Please enter a stock symbol.")