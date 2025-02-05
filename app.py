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
                data = data.dropna()

                features = ['Close', 'EMA', 'STD', 'MA7', 'MA21']
                dataset = data[features].values

                # 3. Scaling
                scaled_data = loaded_scaler.transform(dataset)

                look_back = 21

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

                # 5. Plotting
                st.subheader("Price Prediction")
                last_7_days = data['Close'].tail(30)
                future_dates = pd.date_range(start=last_7_days.index[-1] + pd.Timedelta(days=1), periods=n_predictions)

                fig, ax = plt.subplots(figsize=(27, 10), dpi=150)

                ax.plot(last_7_days.index, last_7_days.values, label='Actual Price (Last 7 Days)')
                ax.plot(future_dates, predictions, label='Predicted Price (Next 7 Days)', marker='o', linestyle='--')

                for x, y in zip(last_7_days.index, last_7_days.values):
                    ax.text(x, y, f'{float(y):.2f}', ha='center', va='bottom', fontsize=8)

                for x, y in zip(future_dates, predictions):
                    ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title('Actual vs. Predicted Price (7 Days)')
                ax.legend(fontsize=20)
                ax.grid(True, alpha=0.7)
                fig.tight_layout()
                st.pyplot(fig)

                st.subheader("Predicted Prices")
                predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
                st.dataframe(predictions_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.warning("Please enter a stock symbol.")