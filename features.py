import pandas as pd
import ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add full technical indicator feature set to DataFrame."""
    df = df.copy()

    # Returns and Volatility
    df["Returns"] = df["close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=10).std()

    # SMA and EMA
    df["SMA_10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)

    # RSI
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)

    # MACD
    df["MACD"] = ta.trend.macd_diff(df["close"])

    # Bollinger Bands
    df["BBU"] = ta.volatility.bollinger_hband(df["close"])
    df["BBL"] = ta.volatility.bollinger_lband(df["close"])

    # ATR
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["STOCHRSIk_14_14_3_3"] = stoch_rsi.stochrsi_k()
    df["STOCHRSId_14_14_3_3"] = stoch_rsi.stochrsi_d()

    return df.dropna()

# List of feature columns used in modeling
FEATURE_COLS = [
    "SMA_10", "EMA_20", "Returns", "Volatility", "RSI", "MACD", 
    "BBU", "BBL", "ATR", "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
]
