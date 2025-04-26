"""Simple rule‑based TA prediction (binary bullish score)."""
import pandas as pd

def predict(df_row: pd.Series) -> float:
    score = 0
    # RSI oversold
    if df_row["RSI"] < 30:
        score += 1
    # Price above EMA20
    if df_row["Close"] > df_row["EMA_20"]:
        score += 1
    # MACD positive
    if df_row["MACD"] > 0:
        score += 1
    return score / 3  # 0‑1