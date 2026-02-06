import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

def get_price_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical price data for a crypto symbol from CryptoCompare."""
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": days,
        "api_key": API_KEY
    }

    try:
        logger.info(f"[{symbol}] Fetching data from CryptoCompare…")
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data["Response"] != "Success":
            logger.warning(f"[{symbol}] API response error: {data.get('Message', 'No message')}")
            return pd.DataFrame()

        df = pd.DataFrame(data["Data"]["Data"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={
            "time": "Date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volumefrom": "volumefrom",
            "volumeto": "volumeto"
        }, inplace=True)
        df.set_index("Date", inplace=True)

        logger.info(f"[{symbol}] Data shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"[{symbol}] Failed to fetch data: {e}")
        return pd.DataFrame()
