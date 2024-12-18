from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import logging
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'lstm_multivar_model.h5')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'multivar_scaler.pkl')
historical_data_path = os.path.join(os.path.dirname(__file__), 'models', 'processed_data.csv')

try:
    model = load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load scaler: {str(e)}")
    scaler = None

try:
    historical_data = pd.read_csv(historical_data_path, parse_dates=['Date'])
    historical_data.set_index('Date', inplace=True)
    historical_data = historical_data.sort_index()
    logger.info("Historical data loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load historical data: {str(e)}")
    historical_data = None

features = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']
look_back = 12

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or historical_data is None:
        return jsonify({'error': 'Model or scaler or historical data not initialized properly.'}), 500

    try:
        data = request.get_json()
        year = data['year']
        month = data['month']

        target_date = pd.to_datetime(f"{year}-{month:02d}-01")

        start_date = target_date - pd.DateOffset(months=12)
        input_data = historical_data.loc[start_date:target_date - pd.DateOffset(months=1), features]

        if len(input_data) < 12:
            return jsonify({'error': 'Not enough historical data to predict. Need at least 12 months of data.'}), 400

        input_values = input_data.values

        scaled_input = scaler.transform(input_values)

        X = scaled_input.reshape(1, look_back, len(features))

        prediction_scaled = model.predict(X)
        pred_scaled_value = prediction_scaled[0,0]  

        # inverse transform
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0,0] = pred_scaled_value
        dummy_array[0,1:] = scaled_input[-1, 1:]

        original_values = scaler.inverse_transform(dummy_array)
        original_prediction = original_values[0,0]  

        predicted_int = round(original_prediction)

        return jsonify({"prediction": predicted_int})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'status': 'API is running',
        'endpoints': {
            '/predict': 'POST - Make predictions with year and month'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
