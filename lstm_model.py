import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 60


def train_lstm_model(symbol, df, feature_cols):
    data = df[feature_cols].values
    target = df["close"].shift(-1).fillna(method="ffill").values

    generator = TimeseriesGenerator(data, target, length=SEQ_LEN, batch_size=32)

    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, len(feature_cols))),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(generator, epochs=25, verbose=1, callbacks=[EarlyStopping(patience=3)])

    path = os.path.join(MODEL_DIR, f"{symbol}_lstm.h5")
    model.save(path)
    print(f"📦 Saved LSTM model to {path}")


def load_lstm_model(symbol):
    path = os.path.join(MODEL_DIR, f"{symbol}_lstm.h5")
    if os.path.exists(path):
        return load_model(path)
    else:
        raise FileNotFoundError(f"No LSTM model found for {symbol} at {path}")


def predict_with_lstm(model, df, feature_cols):
    data = df[feature_cols].values
    generator = TimeseriesGenerator(data, data[:, 0], length=SEQ_LEN, batch_size=1)
    if len(generator) == 0:
        raise ValueError("Not enough data for LSTM prediction")
    pred = model.predict(generator[-1][0], verbose=0)
    return float(pred[0][0])
