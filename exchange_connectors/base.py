"""
Base exchange connector interface.
All exchange implementations inherit from this abstract class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class OrderResult:
    """Standardized order result across all exchanges."""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    filled_price: float = 0.0
    fees: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class Position:
    """Standardized position data."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    exchange: str


class BaseExchangeConnector(ABC):
    """
    Abstract base class for all exchange connectors.
    Provides a unified interface for trading operations.
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.exchange_name: str = "base"
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection and verify credentials."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account balance and status."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass
    
    @abstractmethod
    def buy_market(self, symbol: str, quantity: float) -> OrderResult:
        """Execute market buy order for quantity units."""
        pass
    
    @abstractmethod
    def sell_market(self, symbol: str, quantity: float) -> OrderResult:
        """Execute market sell order for quantity units."""
        pass
    
    @abstractmethod
    def buy_notional(self, symbol: str, usd_amount: float) -> OrderResult:
        """Buy using USD amount (fractional)."""
        pass
    
    @abstractmethod
    def sell_notional(self, symbol: str, usd_amount: float) -> OrderResult:
        """Sell using USD amount (fractional)."""
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        """Close position for a specific symbol."""
        pass
    
    @abstractmethod
    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions."""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> Any:
        """Fetch OHLCV historical data."""
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert generic symbol to exchange-specific format."""
        return symbol
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exchange={self.exchange_name}, paper={self.paper})"
