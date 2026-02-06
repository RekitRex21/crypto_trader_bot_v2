import pandas as pd
import numpy as np

class TechnicalVotes:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def macd_signal(self):
        # Note: we don't have macd_signal alone anymore, we have MACD (which is diff)
        if "MACD" not in self.df.columns:
            return "hold"
        macd = self.df["MACD"].iloc[-1]
        prev_macd = self.df["MACD"].iloc[-2]
        if macd > 0 and prev_macd <= 0:
            return "buy"
        elif macd < 0 and prev_macd >= 0:
            return "sell"
        return "hold"

    def sma_signal(self, fast=10):
        fast_col = f"SMA_{fast}"
        if fast_col not in self.df.columns:
            return "hold"
        sma_fast = self.df[fast_col]
        # Simplified SMA signal: price vs SMA
        if self.df["close"].iloc[-1] > sma_fast.iloc[-1] and self.df["close"].iloc[-2] <= sma_fast.iloc[-2]:
            return "buy"
        elif self.df["close"].iloc[-1] < sma_fast.iloc[-1] and self.df["close"].iloc[-2] >= sma_fast.iloc[-2]:
            return "sell"
        return "hold"

    def rsi_signal(self):
        if "RSI" not in self.df.columns:
            return "hold"
        rsi = self.df["RSI"].iloc[-1]
        if rsi < 30:
            return "buy"
        elif rsi > 70:
            return "sell"
        return "hold"

    def bollinger_signal(self):
        if "close" not in self.df.columns or "BBU" not in self.df.columns or "BBL" not in self.df.columns:
            return "hold"
        close = self.df["close"].iloc[-1]
        if close < self.df["BBL"].iloc[-1]:
            return "buy"
        elif close > self.df["BBU"].iloc[-1]:
            return "sell"
        return "hold"

    def signal(self):
        signals = {
            "macd": self.macd_signal(),
            "sma": self.sma_signal(),
            "rsi": self.rsi_signal(),
            "bollinger": self.bollinger_signal()
        }

        buy_votes = sum(1 for s in signals.values() if s == "buy")
        sell_votes = sum(1 for s in signals.values() if s == "sell")

        if buy_votes >= 2:
            return "buy"
        elif sell_votes >= 2:
            return "sell"
        return "hold"

    def signal_with_confidence(self):
        signals = {
            "macd": self.macd_signal(),
            "sma": self.sma_signal(),
            "rsi": self.rsi_signal(),
            "bollinger": self.bollinger_signal()
        }

        buy_votes = sum(1 for s in signals.values() if s == "buy")
        sell_votes = sum(1 for s in signals.values() if s == "sell")
        total_votes = len(signals)
        confidence = max(buy_votes, sell_votes) / total_votes if total_votes > 0 else 0

        if buy_votes >= 2:
            return "buy", round(confidence, 2)
        elif sell_votes >= 2:
            return "sell", round(confidence, 2)
        return "hold", round(confidence, 2)

# Simple API for the rest of the bot to call:
def predict(df: pd.DataFrame):
    return {"buy": 1, "sell": -1, "hold": 0}[TechnicalVotes(df).signal()]

def signal_with_confidence(df: pd.DataFrame):
    return TechnicalVotes(df).signal_with_confidence()
