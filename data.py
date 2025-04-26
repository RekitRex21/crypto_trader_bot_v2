# data.py  —  clean OHLCV fetcher for CryptoCompare
import os
import logging
from datetime import datetime
import requests
import pandas as pd

API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "YOUR_CRYPTOCOMPARE_API_KEY")
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"

def get_crypto_ohlcv(symbol: str, days: int = 365, quote: str = "USD") -> pd.DataFrame:
    """
    Return a tidy OHLCV DataFrame with columns:
    Date | Open | High | Low | Close | Volume
    """
    url = f"{BASE_URL}?fsym={symbol}&tsym={quote}&limit={days}&api_key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = r.json()["Data"]["Data"]
    except Exception as e:
        logging.error(f"📡 Fetch failed for {symbol}: {e}")
        return pd.DataFrame()

    if not raw:
        logging.warning(f"🛑 No data returned for {symbol}")
        return pd.DataFrame()

    df = (
        pd.DataFrame(raw)
        .rename(
            columns={
                "time": "Timestamp",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volumeto": "Volume",
            }
        )
        .assign(Date=lambda x: pd.to_datetime(x["Timestamp"], unit="s"))
        .loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return df
