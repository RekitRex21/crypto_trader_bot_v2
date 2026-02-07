"""
Alpaca exchange connector for crypto trading.
Supports both paper and live trading via alpaca-py.
"""
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from exchange_connectors.base import (
    BaseExchangeConnector, 
    OrderResult, 
    Position,
    OrderSide
)

logger = logging.getLogger(__name__)


class AlpacaConnector(BaseExchangeConnector):
    """
    Alpaca exchange connector for crypto trading.
    Wraps alpaca-py for standardized interface.
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        super().__init__(api_key, secret_key, paper)
        self.exchange_name = "alpaca"
        self.client: Optional[TradingClient] = None
        self.data_client: Optional[CryptoHistoricalDataClient] = None
    
    def connect(self) -> bool:
        """Initialize Alpaca clients."""
        try:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
            self.data_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            # Verify connection
            account = self.client.get_account()
            logger.info(f"✅ Alpaca connected: Status={account.status}, Paper={self.paper}")
            return True
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get Alpaca account information."""
        account = self.client.get_account()
        return {
            'exchange': self.exchange_name,
            'status': str(account.status),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'trading_blocked': account.trading_blocked,
        }
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Alpaca."""
        alpaca_symbol = self.normalize_symbol(symbol)
        try:
            quote = self.data_client.get_crypto_latest_quote(alpaca_symbol)
            return float(quote[alpaca_symbol].ask_price)
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert to Alpaca format (BTC/USD -> BTCUSD)."""
        return symbol.replace("/", "").upper()
    
    def buy_market(self, symbol: str, quantity: float) -> OrderResult:
        """Execute market buy order."""
        alpaca_symbol = self.normalize_symbol(symbol)
        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=quantity,
                side=AlpacaOrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            result = self.client.submit_order(order)
            logger.info(f"✅ Alpaca BUY {quantity} {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result.id),
                symbol=symbol,
                side="buy",
                quantity=float(result.qty or quantity),
                filled_price=float(result.filled_avg_price or 0),
                raw_response=result.__dict__,
            )
        except Exception as e:
            logger.error(f"❌ Alpaca buy failed: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def sell_market(self, symbol: str, quantity: float) -> OrderResult:
        """Execute market sell order."""
        alpaca_symbol = self.normalize_symbol(symbol)
        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=quantity,
                side=AlpacaOrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            result = self.client.submit_order(order)
            logger.info(f"✅ Alpaca SELL {quantity} {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result.id),
                symbol=symbol,
                side="sell",
                quantity=float(result.qty or quantity),
                filled_price=float(result.filled_avg_price or 0),
                raw_response=result.__dict__,
            )
        except Exception as e:
            logger.error(f"❌ Alpaca sell failed: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def buy_notional(self, symbol: str, usd_amount: float) -> OrderResult:
        """Buy using USD amount (fractional order)."""
        alpaca_symbol = self.normalize_symbol(symbol)
        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                notional=usd_amount,
                side=AlpacaOrderSide.BUY,
                time_in_force=TimeInForce.DAY,  # Required for fractional
            )
            result = self.client.submit_order(order)
            logger.info(f"✅ Alpaca BUY ${usd_amount} of {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result.id),
                symbol=symbol,
                side="buy",
                quantity=float(result.filled_qty or 0),
                filled_price=float(result.filled_avg_price or 0),
                raw_response=result.__dict__,
            )
        except Exception as e:
            logger.error(f"❌ Alpaca notional buy failed: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def sell_notional(self, symbol: str, usd_amount: float) -> OrderResult:
        """Sell using USD amount (fractional order)."""
        alpaca_symbol = self.normalize_symbol(symbol)
        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                notional=usd_amount,
                side=AlpacaOrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            result = self.client.submit_order(order)
            logger.info(f"✅ Alpaca SELL ${usd_amount} of {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result.id),
                symbol=symbol,
                side="sell",
                quantity=float(result.filled_qty or 0),
                filled_price=float(result.filled_avg_price or 0),
                raw_response=result.__dict__,
            )
        except Exception as e:
            logger.error(f"❌ Alpaca notional sell failed: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        positions = []
        try:
            alpaca_positions = self.client.get_all_positions()
            for pos in alpaca_positions:
                positions.append(Position(
                    symbol=pos.symbol,
                    side="long" if float(pos.qty) > 0 else "short",
                    quantity=abs(float(pos.qty)),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    unrealized_pnl=float(pos.unrealized_pl),
                    exchange=self.exchange_name,
                ))
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        return positions
    
    def close_position(self, symbol: str) -> OrderResult:
        """Close position for a specific symbol."""
        alpaca_symbol = self.normalize_symbol(symbol)
        try:
            result = self.client.close_position(alpaca_symbol)
            logger.info(f"✅ Alpaca closed position: {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result.id) if hasattr(result, 'id') else None,
                symbol=symbol,
                side="close",
            )
        except Exception as e:
            logger.error(f"❌ Failed to close position {symbol}: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions."""
        results = []
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.info("✅ Alpaca closed all positions")
            results.append(OrderResult(success=True, symbol="ALL", side="close"))
        except Exception as e:
            logger.error(f"❌ Failed to close all positions: {e}")
            results.append(OrderResult(success=False, symbol="ALL", error=str(e)))
        return results
    
    def get_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Alpaca."""
        alpaca_symbol = self.normalize_symbol(symbol)
        
        # Map timeframe string to Alpaca TimeFrame
        tf_map = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "4h": TimeFrame(4, TimeFrameUnit.Hour),
            "1d": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))
        
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=tf,
                start=datetime.now() - timedelta(days=limit),
            )
            bars = self.data_client.get_crypto_bars(request)
            
            df = bars.df.reset_index()
            df.columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
