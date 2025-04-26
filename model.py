import os
import gzip
import pickle
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor  # ✅ Not XGBModel
from xgb_model import train_xgb, load_xgb

FEATURE_COLS = [
    "SMA_10", "EMA_20", "Returns", "Volatility",
    "RSI", "MACD", "BBU", "BBL", "ATR",
    "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_lstm_attention(input_dim):
    """Functional LSTM model with self-attention."""
    inputs = Input(shape=(60, input_dim), name="input_layer")
    lstm_out = LSTM(64, return_sequences=True, name="lstm_1")(inputs)

    # Self-attention (query = value = LSTM output)
    context = Attention(name="self_attention")([lstm_out, lstm_out])
    lstm_flat = LSTM(32, name="lstm_2")(context)
    output = Dense(1, name="output")(lstm_flat)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(symbol, df_feat, feature_cols):
    """Train LSTM model with attention and save."""
    data = df_feat[feature_cols].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data) - 1):
        X.append(scaled_data[i - 60:i])
        y.append(scaled_data[i + 1][0])
    X, y = np.array(X), np.array(y)

    model = build_lstm_attention(X.shape[2])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_attn.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.gz")

    print(f"🧠 Training LSTM for {symbol}...")
    print(f"📦 Saving to: {model_path}")

    model.save(model_path)
    with gzip.open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)


def load_models(symbols):
    """Load saved models and scalers."""
    model_cache = {}
    for symbol in symbols:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_attn.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.gz")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = tf.keras.models.load_model(model_path)
                with gzip.open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                model_cache[symbol] = {"model": model, "scaler": scaler}
            except Exception as e:
                logging.warning(f"⚠️ Failed to load model for {symbol}: {e}")
    return model_cache
def load_xgb(symbol):
    """Load a trained XGBoost model from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_xgb.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.warning(f"⚠️ Failed to load XGBoost model for {symbol}: {e}")
        return None
