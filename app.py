import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the saved model
loaded_model = keras.models.load_model("sol_price_prediction_model.h5")

# Streamlit app
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
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)

                look_back = 21

                def create_dataset(data, look_back=7):
                    X, y = [], []
                    for i in range(look_back, len(data)):
                        X.append(data[i - look_back:i, :])
                        y.append(data[i, 0])
                    return np.array(X), np.array(y)

                X, y = create_dataset(scaled_data, look_back=look_back)
                X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

                # 4. Prediction (using the loaded model)
                n_predictions = 7
                predictions = []
                all_predicted_scaled = []  # Store scaled predictions

                for i in range(n_predictions):
                    last_lookback_days = scaled_data[-look_back:]
                    X_pred = np.array([last_lookback_days])
                    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]))
                    predicted_data_scaled = loaded_model.predict(X_pred)
                    all_predicted_scaled.append(predicted_data_scaled)


                    dummy_data = np.zeros((1, dataset.shape[1]))
                    dummy_data[0, 0] = predicted_data_scaled[0, 0]
                    predicted_price_full = scaler.inverse_transform(dummy_data)
                    predicted_price = predicted_price_full[0, 0]
                    predictions.append(predicted_price)

                    # Correctly update scaled_data
                    dummy_row_scaled = np.zeros((1, scaled_data.shape[1]))
                    dummy_row_scaled[0, 0] = predicted_data_scaled[0, 0]
                    scaled_data = np.concatenate((scaled_data, dummy_row_scaled), axis=0)


                    data_extended = pd.DataFrame(scaler.inverse_transform(scaled_data), columns=features)
                    data_extended['EMA'] = data_extended['Close'].ewm(span=7, adjust=False).mean()
                    data_extended['STD'] = data_extended['Close'].rolling(window=7).std()
                    data_extended['MA7'] = data_extended['Close'].rolling(window=7).mean()
                    data_extended['MA21'] = data_extended['Close'].rolling(window=21).mean()
                    data_extended = data_extended.dropna()
                    scaled_data = scaler.fit_transform(data_extended[features].values)

                # 5. Plotting (Revert to Matplotlib with labels)
                st.subheader("Price Prediction")
                last_7_days = data['Close'].tail(7)
                future_dates = pd.date_range(start=last_7_days.index[-1] + pd.Timedelta(days=1), periods=n_predictions)

                fig, ax = plt.subplots(figsize=(20, 8))

                ax.plot(last_7_days.index, last_7_days.values, label='Actual Price (Last 7 Days)')
                ax.plot(future_dates, predictions, label='Predicted Price (Next 7 Days)', marker='o', linestyle='--')

                for x, y in zip(last_7_days.index, last_7_days.values):
                    ax.text(x, y, f'{float(y):.2f}', ha='center', va='bottom', fontsize=8)

                for x, y in zip(future_dates, predictions):
                    ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title('Actual vs. Predicted Price (7 Days)')
                ax.legend(fontsize=14)
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