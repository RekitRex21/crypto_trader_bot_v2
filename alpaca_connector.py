import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestTradeRequest

load_dotenv()
logger = logging.getLogger(__name__)

LOG_PATH = "trades_log.txt"

def log_trade(action: str, symbol: str, amount: float):
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} | {action.upper()} | {symbol} | ${amount:.2f}\n")

class AlpacaConnector:
    """Wrapper around Alpaca paper/live trading for crypto."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = TradingClient(
            api_key=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
            paper=paper,
        )
        self.data_client = CryptoHistoricalDataClient(
            api_key=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
        )

    def buy_usd_notional(self, symbol: str, amount_usd: float):
        """Place a market buy order for a USD amount."""
        try:
            # Check purchasing power or safeguards here if needed
            order = MarketOrderRequest(
                symbol=symbol,
                notional=amount_usd,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            self.client.submit_order(order)
            logger.info(f"✅ Bought ${amount_usd:.2f} of {symbol}")
            log_trade("BUY", symbol, amount_usd)
        except Exception as e:
            logger.error(f"❌ Failed to buy {symbol}: {e}")

    def sell_usd_notional(self, symbol: str, amount_usd: float):
        """Place a market sell order for a USD amount."""
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                notional=amount_usd,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            self.client.submit_order(order)
            logger.info(f"✅ Sold ${amount_usd:.2f} of {symbol}")
            log_trade("SELL", symbol, amount_usd)
        except Exception as e:
            logger.error(f"❌ Failed to sell {symbol}: {e}")

    def latest_price(self, symbol: str) -> float:
        """Fetch the latest trade price for a symbol."""
        try:
            req = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
            res = self.data_client.get_crypto_latest_trade(req)
            return res[symbol].price
        except Exception as e:
            logger.error(f"❌ Failed to fetch price for {symbol}: {e}")
            return 0.0

    def close_all(self):
        """Close all open positions."""
        try:
            self.client.close_all_positions()
            logger.info("✅ Closed all open positions")
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
