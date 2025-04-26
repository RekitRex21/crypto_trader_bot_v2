# live.py
# ----------------------------------------------------------------------
"""Alpaca execution layer for crypto paper/live trading.

Expected env-vars
-----------------
APCA_API_KEY_ID        ← your Alpaca key
APCA_API_SECRET_KEY    ← your Alpaca secret
APCA_API_BASE_URL      ← override if you switch off paper trading
"""

from __future__ import annotations
import os
from typing import Dict

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaConnector:
    """Minimal wrapper around Alpaca-py for spot-crypto paper trading."""

    def __init__(self, *, paper: bool = True):
        # --- order client (trading) -----------------------------------
        self.client = TradingClient(
            api_key=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
            paper=paper,
        )

        # --- market-data client ---------------------------------------
        self.historic = CryptoHistoricalDataClient(
            api_key=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
        )

    # ------------------------------------------------------------------
    # prices
    # ------------------------------------------------------------------
    def latest_price(self, symbol: str) -> float:
        """
        Robust fetch of the most recent trade price.

        Works for BOTH old (string) and new (request-object) signatures of
        Alpaca-py 0.13+. Falls back to a 1-minute bar if trade fetch fails.
        """

        # --- preferred: new signature (request object) ----------------
        try:
            req = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.historic.get_crypto_latest_trade(req)
            return float(trade.price)
        except Exception:
            pass  # fall through to older path

        # --- legacy (pre-0.13) string signature -----------------------
        try:
            trade = self.historic.get_crypto_latest_trade(symbol)
            return float(trade.price)
        except Exception:
            pass  # fall through to bar fallback

        # --- final fallback: 1-minute bar -----------------------------
        bars_req = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=1
        )
        bars = self.historic.get_crypto_bars(bars_req)

        # SDK returns a Bars object with .df (DataFrame) in ≥0.13
        if hasattr(bars, "df"):
            return float(bars.df["close"].iloc[-1])

        # older SDK: dict-like container → bars[symbol][0]
        return float(bars[symbol][0].close)

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------
    def buy_usd_notional(self, symbol: str, usd_size: float) -> str:
        """Market-buy for a USD notional amount. Returns order-id."""
        px = self.latest_price(symbol)
        qty = round(usd_size / px, 8)
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        return self.client.submit_order(order).id

    def sell_quantity(self, symbol: str, qty: float) -> str:
        """Market-sell a crypto quantity. Returns order-id."""
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        return self.client.submit_order(order).id


# Registry consumed by main.py
CONNECTORS: Dict[str, type] = {"alpaca": AlpacaConnector}
