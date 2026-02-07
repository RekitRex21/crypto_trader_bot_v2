"""
Binance US exchange connector for crypto trading.
Uses CCXT library for standardized API access.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import ccxt

from exchange_connectors.base import (
    BaseExchangeConnector,
    OrderResult,
    Position,
)

logger = logging.getLogger(__name__)


class BinanceUSConnector(BaseExchangeConnector):
    """
    Binance US exchange connector using CCXT.
    Supports spot trading for US-compliant crypto pairs.
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = False):
        super().__init__(api_key, secret_key, paper)
        self.exchange_name = "binance_us"
        self.exchange: Optional[ccxt.binanceus] = None
        
        if paper:
            logger.warning("⚠️ Binance US does not have a native paper trading mode. Using sandbox if available.")
    
    def connect(self) -> bool:
        """Initialize Binance US client."""
        try:
            self.exchange = ccxt.binanceus({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            
            # Enable sandbox for paper trading if requested
            if self.paper:
                self.exchange.set_sandbox_mode(True)
            
            # Verify connection by fetching balance
            balance = self.exchange.fetch_balance()
            logger.info(f"✅ Binance US connected. Total assets: {len(balance['info']['balances'])}")
            return True
        except Exception as e:
            logger.error(f"❌ Binance US connection failed: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get Binance US account information."""
        try:
            balance = self.exchange.fetch_balance()
            # Sum up USD equivalents
            total_usd = float(balance.get('USD', {}).get('total', 0))
            free_usd = float(balance.get('USD', {}).get('free', 0))
            
            return {
                'exchange': self.exchange_name,
                'status': 'active',
                'equity': total_usd,
                'buying_power': free_usd,
                'cash': free_usd,
                'trading_blocked': False,
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {'exchange': self.exchange_name, 'error': str(e)}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Binance US."""
        binance_symbol = self.normalize_symbol(symbol)
        try:
            ticker = self.exchange.fetch_ticker(binance_symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert to Binance format (BTC/USD -> BTC/USD)."""
        # Binance US uses BTC/USD format (same as our standard)
        return symbol.upper()
    
    def buy_market(self, symbol: str, quantity: float) -> OrderResult:
        """Execute market buy order."""
        binance_symbol = self.normalize_symbol(symbol)
        try:
            result = self.exchange.create_market_buy_order(binance_symbol, quantity)
            logger.info(f"✅ Binance US BUY {quantity} {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result['id']),
                symbol=symbol,
                side="buy",
                quantity=float(result.get('filled', quantity)),
                filled_price=float(result.get('average', 0)),
                fees=float(result.get('fee', {}).get('cost', 0)),
                raw_response=result,
            )
        except Exception as e:
            logger.error(f"❌ Binance US buy failed: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def sell_market(self, symbol: str, quantity: float) -> OrderResult:
        """Execute market sell order."""
        binance_symbol = self.normalize_symbol(symbol)
        try:
            result = self.exchange.create_market_sell_order(binance_symbol, quantity)
            logger.info(f"✅ Binance US SELL {quantity} {symbol}")
            return OrderResult(
                success=True,
                order_id=str(result['id']),
                symbol=symbol,
                side="sell",
                quantity=float(result.get('filled', quantity)),
                filled_price=float(result.get('average', 0)),
                fees=float(result.get('fee', {}).get('cost', 0)),
                raw_response=result,
            )
        except Exception as e:
            logger.error(f"❌ Binance US sell failed: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def buy_notional(self, symbol: str, usd_amount: float) -> OrderResult:
        """Buy using USD amount."""
        # Calculate quantity from price
        price = self.get_current_price(symbol)
        if price <= 0:
            return OrderResult(success=False, symbol=symbol, error="Could not get price")
        
        quantity = usd_amount / price
        return self.buy_market(symbol, quantity)
    
    def sell_notional(self, symbol: str, usd_amount: float) -> OrderResult:
        """Sell using USD amount."""
        price = self.get_current_price(symbol)
        if price <= 0:
            return OrderResult(success=False, symbol=symbol, error="Could not get price")
        
        quantity = usd_amount / price
        return self.sell_market(symbol, quantity)
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions (non-zero balances)."""
        positions = []
        try:
            balance = self.exchange.fetch_balance()
            for asset, data in balance.items():
                if isinstance(data, dict) and data.get('total', 0) > 0:
                    # Skip USD/stablecoins
                    if asset in ['USD', 'USDT', 'USDC', 'BUSD']:
                        continue
                    
                    symbol = f"{asset}/USD"
                    try:
                        price = self.get_current_price(symbol)
                        positions.append(Position(
                            symbol=symbol,
                            side="long",
                            quantity=float(data['total']),
                            entry_price=0,  # Not available from balance
                            current_price=price,
                            unrealized_pnl=0,  # Would need entry price
                            exchange=self.exchange_name,
                        ))
                    except Exception:
                        pass  # Symbol might not be tradeable
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        return positions
    
    def close_position(self, symbol: str) -> OrderResult:
        """Close position by selling all holdings."""
        try:
            balance = self.exchange.fetch_balance()
            base_asset = symbol.split('/')[0]
            quantity = float(balance.get(base_asset, {}).get('free', 0))
            
            if quantity <= 0:
                return OrderResult(success=True, symbol=symbol, side="close")  # Nothing to close
            
            return self.sell_market(symbol, quantity)
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return OrderResult(success=False, symbol=symbol, error=str(e))
    
    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions."""
        results = []
        positions = self.get_open_positions()
        
        for pos in positions:
            result = self.close_position(pos.symbol)
            results.append(result)
        
        if not positions:
            results.append(OrderResult(success=True, symbol="ALL", side="close"))
        
        return results
    
    def get_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Binance US."""
        binance_symbol = self.normalize_symbol(symbol)
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(binance_symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
