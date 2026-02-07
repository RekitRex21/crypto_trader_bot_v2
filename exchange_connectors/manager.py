"""
Exchange manager factory for initializing and managing multiple exchange connectors.
"""
import logging
from typing import Dict, Optional, List

from core.config import config
from exchange_connectors.base import BaseExchangeConnector
from exchange_connectors.alpaca_connector import AlpacaConnector
from exchange_connectors.binance_us_connector import BinanceUSConnector
from exchange_connectors.coinbase_connector import CoinbaseConnector

logger = logging.getLogger(__name__)


class ExchangeManager:
    """
    Factory and manager for exchange connectors.
    Initializes configured exchanges and provides unified access.
    """
    
    def __init__(self):
        self.connectors: Dict[str, BaseExchangeConnector] = {}
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize all configured exchanges."""
        # Alpaca
        if config.alpaca.is_configured:
            try:
                connector = AlpacaConnector(
                    api_key=config.alpaca.api_key,
                    secret_key=config.alpaca.secret_key,
                    paper=config.alpaca.paper,
                )
                if connector.connect():
                    self.connectors['alpaca'] = connector
                    logger.info("✅ Alpaca exchange initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Alpaca: {e}")
        
        # Binance US
        if config.binance_us.is_configured:
            try:
                connector = BinanceUSConnector(
                    api_key=config.binance_us.api_key,
                    secret_key=config.binance_us.secret_key,
                    paper=False,  # Binance US doesn't have paper mode
                )
                if connector.connect():
                    self.connectors['binance_us'] = connector
                    logger.info("✅ Binance US exchange initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Binance US: {e}")
        
        # Coinbase  
        if config.coinbase.is_configured:
            try:
                connector = CoinbaseConnector(
                    api_key=config.coinbase.api_key,
                    secret_key=config.coinbase.api_secret,
                    paper=False,
                )
                if connector.connect():
                    self.connectors['coinbase'] = connector
                    logger.info("✅ Coinbase exchange initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Coinbase: {e}")
        
        if not self.connectors:
            logger.warning("⚠️ No exchanges configured. Check your .env file.")
    
    def get(self, exchange_name: str) -> Optional[BaseExchangeConnector]:
        """Get a specific exchange connector."""
        return self.connectors.get(exchange_name)
    
    def get_primary(self) -> Optional[BaseExchangeConnector]:
        """Get the first available exchange (Alpaca preferred)."""
        for name in ['alpaca', 'binance_us', 'coinbase']:
            if name in self.connectors:
                return self.connectors[name]
        return None
    
    def get_all(self) -> Dict[str, BaseExchangeConnector]:
        """Get all initialized connectors."""
        return self.connectors
    
    def list_active(self) -> List[str]:
        """List names of active exchanges."""
        return list(self.connectors.keys())
    
    def get_combined_account_info(self) -> Dict[str, Dict]:
        """Get account info from all exchanges."""
        info = {}
        for name, connector in self.connectors.items():
            try:
                info[name] = connector.get_account_info()
            except Exception as e:
                info[name] = {'error': str(e)}
        return info
    
    def close_all(self):
        """Close all exchange connections."""
        for name, connector in self.connectors.items():
            try:
                connector.close_all_positions()
                logger.info(f"Closed all positions on {name}")
            except Exception as e:
                logger.error(f"Error closing positions on {name}: {e}")


# Singleton instance
exchange_manager: Optional[ExchangeManager] = None

def get_exchange_manager() -> ExchangeManager:
    """Get or create the exchange manager singleton."""
    global exchange_manager
    if exchange_manager is None:
        exchange_manager = ExchangeManager()
    return exchange_manager
