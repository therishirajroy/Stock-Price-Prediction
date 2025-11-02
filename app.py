import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# --- Helper Functions ---

@st.cache_data
def load_data(ticker_symbol, period='5y'):
    """
    Loads historical stock data from Yahoo Finance for a specified period.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if data.empty:
            st.error(f"No data found for ticker '{ticker_symbol}'. Please check the symbol.")
            return None
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.date
        # Ensure we only have numeric data for scaling
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None

@st.cache_resource
def load_keras_model(model_path):
    """
    Loads a saved Keras model from disk.
    Uses caching to avoid reloading on every run.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please make sure it's in the same directory as the script.")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model {model_path}: {e}")
        return None


def get_all_predictions(df, lstm_model, cnnlstm_model, input_days=21):
    """
    Generates predictions for the entire historical dataset using scaled data and loaded models.
    """
    if lstm_model is None or cnnlstm_model is None:
        return df # Return original df if models failed to load

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_cols = ['Open', 'High', 'Low', 'Close']
    
    # Initialize Scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    lstm_preds_scaled = []
    cnnlstm_preds_scaled = []
    
    # Iterate from the first possible prediction point to the end
    progress_bar = st.progress(0, text="Generating predictions...")
    total_predictions = len(scaled_data) - input_days
    if total_predictions <= 0:
        st.warning("Not enough historical data to generate predictions for the selected period and look-back days.")
        progress_bar.empty()
        return df

    for i in range(input_days, len(scaled_data)):
        model_input = scaled_data[i-input_days:i, :]
        # Reshape for Keras model (samples, timesteps, features)
        model_input_reshaped = np.reshape(model_input, (1, model_input.shape[0], model_input.shape[1]))
        
        # Get scaled prediction from models. Handle cases where the model returns a sequence (2D array).
        # We assume we need the first prediction in the sequence.
        lstm_pred_output = lstm_model.predict(model_input_reshaped, verbose=0)[0]
        lstm_pred = lstm_pred_output[0] if lstm_pred_output.ndim > 1 else lstm_pred_output
        
        cnnlstm_pred_output = cnnlstm_model.predict(model_input_reshaped, verbose=0)[0]
        cnnlstm_pred = cnnlstm_pred_output[0] if cnnlstm_pred_output.ndim > 1 else cnnlstm_pred_output
        
        lstm_preds_scaled.append(lstm_pred)
        cnnlstm_preds_scaled.append(cnnlstm_pred)
        
        # Update progress bar
        progress_bar.progress((i - input_days + 1) / total_predictions, text="Generating predictions...")
    
    progress_bar.empty()

    # --- Inverse transform LSTM predictions ---
    lstm_preds_scaled = np.array(lstm_preds_scaled)
    dummy_array_lstm = np.zeros((len(lstm_preds_scaled), len(feature_cols)))
    dummy_array_lstm[:, :len(target_cols)] = lstm_preds_scaled
    lstm_preds_actual = scaler.inverse_transform(dummy_array_lstm)[:, :len(target_cols)]
    
    # --- Inverse transform CNN-LSTM predictions ---
    cnnlstm_preds_scaled = np.array(cnnlstm_preds_scaled)
    dummy_array_cnnlstm = np.zeros((len(cnnlstm_preds_scaled), len(feature_cols)))
    dummy_array_cnnlstm[:, :len(target_cols)] = cnnlstm_preds_scaled
    cnnlstm_preds_actual = scaler.inverse_transform(dummy_array_cnnlstm)[:, :len(target_cols)]

    # --- Create prediction dataframes ---
    lstm_pred_df = pd.DataFrame(lstm_preds_actual, columns=[f'{col} LSTM' for col in target_cols])
    cnnlstm_pred_df = pd.DataFrame(cnnlstm_preds_actual, columns=[f'{col} CNN-LSTM' for col in target_cols])
    
    # Add NaNs for the initial rows
    empty_rows = pd.DataFrame(np.nan, index=range(input_days), columns=lstm_pred_df.columns)
    lstm_pred_df = pd.concat([empty_rows, lstm_pred_df]).reset_index(drop=True)
    
    empty_rows_cnn = pd.DataFrame(np.nan, index=range(input_days), columns=cnnlstm_pred_df.columns)
    cnnlstm_pred_df = pd.concat([empty_rows_cnn, cnnlstm_pred_df]).reset_index(drop=True)
    
    # Combine with the original dataframe
    final_df = pd.concat([df, lstm_pred_df, cnnlstm_pred_df], axis=1)
    
    return final_df


def create_plot(df):
    """
    Creates an interactive plot with lines for actual and predicted close prices.
    """
    fig = go.Figure()

    # Add trace for Actual Close price
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], mode='lines', name='Actual Close Price',
        line=dict(color='royalblue', width=2)
    ))

    # Add trace for LSTM Predicted Close price
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close LSTM'], mode='lines', name='LSTM Predicted Close',
        line=dict(color='orange', width=2, dash='dot')
    ))

    # Add trace for CNN-LSTM Predicted Close price
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close CNN-LSTM'], mode='lines', name='CNN-LSTM Predicted Close',
        line=dict(color='fuchsia', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Actual vs. Model Predicted Close Prices',
        xaxis_title='Date', yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=True,
        legend=dict(x=0.01, y=0.99, traceorder='normal', font=dict(size=12))
    )
    return fig

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("Stock Price Prediction Dashboard")
st.write("This application uses LSTM and CNN-LSTM models to predict stock prices.")
st.info("ℹ️ **Important:** Make sure your `best_lstm_model.keras` and `best_cnnlstm_model.keras` files are in the same folder as this script.")


# --- Sidebar for User Input ---
st.sidebar.header("Settings")
ticker_symbol = st.sidebar.selectbox("Choose a Stock Ticker:", ['Select','AAPL', 'GOOGL', 'MSFT', 'AMZN', \
                                                                'TSLA', 'NVDA', 'META','NTPC.NS', 'JIOFIN.NS',\
                                                                "ETERNAL.NS","TATAMOTORS.NS","HDFCBANK.NS","RELIANCE.NS"])
data_period = st.sidebar.selectbox("Select Data Period:", ['Select','50d','100d','1y', '2y', '5y', '10y', 'max'], index=0)

# --- Main Page Content ---
if ticker_symbol!='Select' and data_period!='Select':
    # Load models first
    lstm_model = load_keras_model('best_lstm_model.keras')
    cnnlstm_model = load_keras_model('best_cnnlstm_model.keras')
    
    data_df = load_data(ticker_symbol, period=data_period)

    if data_df is not None and lstm_model is not None and cnnlstm_model is not None:
        predictions_df = get_all_predictions(data_df.copy(), lstm_model, cnnlstm_model)

        st.header(f"Predictions for {ticker_symbol}")
        
        # Filter the dataframe to only show dates where predictions are available
        plot_df = predictions_df.dropna().copy()

        if plot_df.empty:
             st.warning("Not enough historical data to generate predictions for the selected period and look-back days. Please select a longer data period or decrease the number of past days to use.")
        else:
            prediction_plot = create_plot(plot_df)
            st.plotly_chart(prediction_plot, use_container_width=True)

        st.header("Prediction vs. Actual Data")
        st.write("Comparison of actual close prices with model predictions. The most recent data is at the top.")
        
        display_cols = ['Date', 'Close', 'Close LSTM', 'Close CNN-LSTM']
        display_df = predictions_df[display_cols].copy()
        display_df.rename(columns={'Close': 'Actual Close'}, inplace=True)

        # Format numbers to 2 decimal places, handling NaNs
        for col in display_df.columns:
            if col != 'Date':
                display_df[col] = display_df[col].apply(lambda x: f'{x:,.2f}' if pd.notnull(x) else 'N/A')
        
        st.dataframe(display_df.sort_values(by='Date', ascending=False).reset_index(drop=True).head())

