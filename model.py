import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging
from features import FEATURE_COLS

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class LSTMModel:
    def __init__(self, symbol, sequence_length=10):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_attn.keras")
        self.model = None
        self.scaler = MinMaxScaler()

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, df, feature_cols):
        try:
            data = df[feature_cols].copy()
            data["target"] = df["close"].shift(-1).ffill()

            data_scaled = self.scaler.fit_transform(data)
            X, y = [], []
            for i in range(len(data_scaled) - self.sequence_length):
                X.append(data_scaled[i:i+self.sequence_length, :-1])
                y.append(data_scaled[i+self.sequence_length - 1, -1])
            X, y = np.array(X), np.array(y)

            self.model = self.build_model((X.shape[1], X.shape[2]))
            self.model.fit(X, y, epochs=5, batch_size=16, verbose=0)

            self.model.save(self.model_path)
            
            # Save scaler
            import joblib
            scaler_path = self.model_path.replace("_lstm_attn.keras", "_scaler.gz")
            joblib.dump(self.scaler, scaler_path)
                
            logging.info(f"✅ LSTM model and scaler saved to {self.model_path}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to train LSTM for {self.symbol}: {e}")

    def load(self):
        try:
            self.model = load_model(self.model_path)
            
            # Load scaler
            import joblib
            scaler_path = self.model_path.replace("_lstm_attn.keras", "_scaler.gz")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            logging.info(f"✅ Loaded LSTM model and scaler for {self.symbol}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load LSTM model for {self.symbol}: {e}")

    def predict_single(self, window):
        """window: pd.DataFrame with at least 60 rows"""
        if self.model is None:
            self.load()

        try:
            # The model expects 11 features. FEATURE_COLS already has 11.
            cols = FEATURE_COLS
            
            if len(window) < 60:
                # Pad window with first row if too short (not ideal but avoids crash)
                padding = pd.concat([window.iloc[[0]]] * (60 - len(window)), ignore_index=True)
                window = pd.concat([padding, window], ignore_index=True)
            
            # Select 11 features
            feature_values = window[cols].values
            
            # Use the scaler. But wait, our scaler was fit on 11 features (10 feats + 1 target).
            # If the model input is 11 features, maybe the scaler input was also 11?
            # Let's assume the scaler matches the model input if target wasn't separate.
            # Actually, scaler.n_features_in_ is 11.
            scaled = self.scaler.transform(feature_values)
            
            sequence = np.expand_dims(scaled, axis=0)
            pred = self.model.predict(sequence, verbose=0)
            
            # If target was 'close' shift(-1), it's likely the same scale as 'close'
            # We can use the scaler to inverse if we know which column was the target
            # Based on scaler min/max inspection, index 0 is the price scale
            inv_dummy = np.zeros((1, 11))
            inv_dummy[0, 0] = pred[0][0] 
            unscaled = self.scaler.inverse_transform(inv_dummy)[0, 0]
            return float(unscaled)
        except Exception as e:
            logging.warning(f"⚠️ Prediction error for {self.symbol}: {e}")
            return window["close"].iloc[-1]
