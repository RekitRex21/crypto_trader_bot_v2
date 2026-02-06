import os
import pickle
import logging
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class XGBoostModel:
    def __init__(self, symbol, feature_cols):
        self.symbol = symbol
        self.feature_cols = feature_cols
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, f"{symbol}_xgb.pkl")

    def train(self, df):
        try:
            X = df[self.feature_cols]
            y = df["close"].shift(-1).ffill()

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            self.model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                objective="reg:squarederror",
                random_state=42
            )

            self.model.fit(X_train, y_train)

            logging.info(f"📈 Training XGBoost for {self.symbol}...")
            logging.info(f"📦 Saving XGB to: {self.model_path}")

            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            logging.info(f"✅ XGBoost model saved for {self.symbol}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to train XGBoost for {self.symbol}: {e}")

    def load(self):
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logging.info(f"📤 Loaded XGBoost model for {self.symbol}")
            return True
        except Exception as e:
            logging.warning(f"⚠️ Failed to load XGBoost model for {self.symbol}: {e}")
            return False

    def predict(self, X_row):
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        if X_row is None or X_row.size == 0:
            raise ValueError("Empty input row for prediction")
        prediction = self.model.predict(X_row)
        return float(prediction[0])
