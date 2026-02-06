import pandas as pd
import numpy as np
import logging
from features import add_features, FEATURE_COLS
from model import LSTMModel
from xgb_model import XGBoostModel
from technical_votes import signal_with_confidence as tech_predict

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, symbol, df, strategy="ensemble", debug=False):
        self.symbol = symbol
        self.df = df.copy()
        self.strategy = strategy
        self.debug = debug
        self.lstm_model = LSTMModel(self.symbol)
        self.lstm_model.load()
        self.xgb_model = XGBoostModel(symbol, FEATURE_COLS)
        self.xgb_model.load()
        self.trades = []
        self.balance = 1000.0
        self.position = 0.0
        self.entry_price = 0.0

    def get_trade_signal(self, index=-1):
        """Calculate signal for a specific row (default: latest)."""
        # Determine the row and window
        if index == -1:
            row = self.df.iloc[-1]
            current_slice = self.df
            window = self.df.tail(60)
        else:
            row = self.df.iloc[index]
            current_slice = self.df.iloc[:index+1]
            window = self.df.iloc[:index+1].tail(60)

        # 1. XGBoost
        try:
            feat_vals = row[FEATURE_COLS].values.reshape(1, -1)
            xgb_score = self.xgb_model.predict(feat_vals)
        except Exception as e:
            if self.debug: logger.warning(f"[{self.symbol}] XGB prediction error: {e}")
            xgb_score = row["close"]

        # 2. Technical
        tech_signal, _ = tech_predict(current_slice) # Keep the original tech_predict alias
        tech_score = 1 if tech_signal == "buy" else (-1 if tech_signal == "sell" else 0)

        # 3. LSTM
        try:
            lstm_pred = self.lstm_model.predict_single(window)
        except Exception as e:
            if self.debug: logger.warning(f"[{self.symbol}] LSTM prediction error: {e}")
            lstm_pred = row["close"]

        # Vote
        votes = 0
        if xgb_score > row["close"]: votes += 1
        if tech_score > 0: votes += 1
        if lstm_pred > row["close"]: votes += 1

        if self.debug:
            logger.info(f"DEBUG [{self.symbol}]: Price: {row['close']:.2f}, XGB: {xgb_score:.2f}, Tech: {tech_score}, LSTM: {lstm_pred:.2f}, Votes: {votes}")

        if votes >= 2: return "BUY"
        elif votes <= 1: return "SELL"
        return "HOLD"

    def run(self):
        self.df = add_features(self.df)
        self.df.dropna(inplace=True)
        if self.df.empty:
            logger.warning(f"[{self.symbol}] No data after feature engineering")
            return None, None

        for idx, row in self.df.iterrows():
            # Get the signal using the new method
            signal = self.get_trade_signal(index=self.df.index.get_loc(idx))
            
            # The 'decide' method's logic for position management is still needed here
            decision = "HOLD"
            if signal == "BUY" and self.position == 0:
                decision = "BUY"
            elif signal == "SELL" and self.position > 0:
                decision = "SELL"

            self.execute_trade(decision, row["close"], row.get("Date", idx))

        trades_df = pd.DataFrame(self.trades)
        return trades_df, self.balance

    def decide(self, xgb_score, tech_score, lstm_pred, row):
        # This method is now deprecated or can be refactored to call get_trade_signal
        # For now, keeping it as is, but the run method above uses get_trade_signal directly.
        # If decide is still used elsewhere, it should be updated to use get_trade_signal.
        votes = 0
        if xgb_score > row["close"]:
            votes += 1
        if tech_score > 0:
            votes += 1
        if lstm_pred > row["close"]:
            votes += 1

        if self.debug:
            logger.info(f"DEBUG [{self.symbol}]: Price: {row['close']:.2f}, XGB: {xgb_score:.2f}, Tech: {tech_score}, LSTM: {lstm_pred:.2f}, Votes: {votes}")

        if votes >= 2 and self.position == 0:
            return "BUY"
        elif votes <= 1 and self.position > 0:
            return "SELL"
        return "HOLD"

    def execute_trade(self, decision, price, date):
        if decision == "BUY":
            self.position = self.balance / price
            self.entry_price = price
            self.balance = 0
            self.trades.append({"Date": date, "Action": "BUY", "Price": price})
        elif decision == "SELL" and self.position > 0:
            self.balance = self.position * price
            self.position = 0
            self.trades.append({"Date": date, "Action": "SELL", "Price": price})

def metrics(trades_df):
    if trades_df.empty:
        return {"Total Trades": 0, "Profit/Loss": 0.0}
    
    buys = trades_df[trades_df["Action"] == "BUY"]
    sells = trades_df[trades_df["Action"] == "SELL"]
    
    return {
        "Total Trades": len(trades_df),
        "Buy Orders": len(buys),
        "Sell Orders": len(sells),
        "Last Price": trades_df["Price"].iloc[-1] if not trades_df.empty else 0
    }
