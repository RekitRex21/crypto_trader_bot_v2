import os
import pickle
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_xgb(symbol, df, feature_cols):
    """Train an XGBoost regressor and save it."""
    try:
        X = df[feature_cols]
        y = df["Close"].shift(-1).ffill()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(X_train, y_train)

        print(f"📈 Training XGBoost for {symbol}...")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{symbol}_xgb.pkl")
        print(f"📦 Saving XGB to: {model_path}")
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logging.info(f"✅ XGBoost model saved for {symbol}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to train XGBoost for {symbol}: {e}")
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
