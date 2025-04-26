"""Feature engineering – TA indicators."""
import pandas as pd
import pandas_ta as ta

FEATURE_COLS = [
    "SMA_10", "EMA_20", "Returns", "Volatility",
    "RSI", "MACD", "BBU", "BBL", "ATR", "STOCHRSIk_14_14_3_3",
"STOCHRSId_14_14_3_3",
]

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA_10"] = ta.sma(out["Close"], length=10)
    out["EMA_20"] = ta.ema(out["Close"], length=20)
    out["Returns"] = out["Close"].pct_change()
    out["Volatility"] = out["Returns"].rolling(24).std()
    out["RSI"] = ta.rsi(out["Close"], length=14)
    macd = ta.macd(out["Close"])
    out["MACD"] = macd["MACD_12_26_9"]
    bb = ta.bbands(out["Close"], length=20, std=2)
    out["BBU"] = bb["BBU_20_2.0"]
    out["BBL"] = bb["BBL_20_2.0"]
    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)
    stochrsi = ta.stochrsi(out["Close"])
    if stochrsi is not None:
        out = pd.concat([out, stochrsi], axis=1)
    else:
        print("⚠️ Warning: StochRSI returned None — not enough data?")
    return out.dropna()
