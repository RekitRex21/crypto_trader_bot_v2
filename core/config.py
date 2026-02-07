"""
Centralized configuration management using Pydantic.
Loads from environment variables with validation.
"""
import os
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseModel):
    """PostgreSQL connection settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="crypto_trading")
    user: str = Field(default="crypto_bot")
    password: str = Field(default="changeme123")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class AlpacaConfig(BaseModel):
    """Alpaca API configuration."""
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    paper: bool = Field(default=True)
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)


class BinanceUSConfig(BaseModel):
    """Binance US API configuration."""
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)


class CoinbaseConfig(BaseModel):
    """Coinbase API configuration."""
    api_key: str = Field(default="")
    api_secret: str = Field(default="")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""
    bot_token: str = Field(default="")
    allowed_user_ids: List[int] = Field(default_factory=list)
    
    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.allowed_user_ids)


class RiskConfig(BaseModel):
    """Risk management parameters."""
    max_position_size: float = Field(default=0.02, description="Max 2% per trade")
    max_drawdown: float = Field(default=0.15, description="15% circuit breaker")
    daily_loss_limit: float = Field(default=0.05, description="5% daily max loss")
    stop_loss_atr_multiplier: float = Field(default=2.0)
    take_profit_ratio: float = Field(default=2.0, description="Risk:Reward 1:2")
    max_trades_per_day: int = Field(default=10)
    cooldown_after_loss: int = Field(default=3600, description="Seconds")


class TradingConfig(BaseModel):
    """Trading engine settings."""
    symbols: List[str] = Field(default=["BTC/USD", "ETH/USD"])
    timeframe: str = Field(default="1h")
    loop_interval: int = Field(default=300, description="Seconds between checks")
    min_confidence: float = Field(default=0.70)
    initial_capital: float = Field(default=10000.0)


class Config(BaseModel):
    """Master configuration container."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    binance_us: BinanceUSConfig = Field(default_factory=BinanceUSConfig)
    coinbase: CoinbaseConfig = Field(default_factory=CoinbaseConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            database=DatabaseConfig(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "crypto_trading"),
                user=os.getenv("DB_USER", "crypto_bot"),
                password=os.getenv("DB_PASSWORD", "changeme123"),
            ),
            alpaca=AlpacaConfig(
                api_key=os.getenv("APCA_API_KEY_ID", ""),
                secret_key=os.getenv("APCA_API_SECRET_KEY", ""),
                paper=os.getenv("APCA_PAPER", "true").lower() == "true",
            ),
            binance_us=BinanceUSConfig(
                api_key=os.getenv("BINANCE_US_API_KEY", ""),
                secret_key=os.getenv("BINANCE_US_SECRET_KEY", ""),
            ),
            coinbase=CoinbaseConfig(
                api_key=os.getenv("COINBASE_API_KEY", ""),
                api_secret=os.getenv("COINBASE_API_SECRET", ""),
            ),
            telegram=TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                allowed_user_ids=[int(x) for x in os.getenv("TELEGRAM_USER_IDS", "").split(",") if x],
            ),
            risk=RiskConfig(
                max_position_size=float(os.getenv("RISK_MAX_POSITION_SIZE", "0.02")),
                max_drawdown=float(os.getenv("RISK_MAX_DRAWDOWN", "0.15")),
                daily_loss_limit=float(os.getenv("RISK_DAILY_LOSS_LIMIT", "0.05")),
                max_trades_per_day=int(os.getenv("RISK_MAX_TRADES_PER_DAY", "10")),
            ),
            trading=TradingConfig(
                symbols=os.getenv("TRADING_SYMBOLS", "BTC/USD,ETH/USD").split(","),
                timeframe=os.getenv("TRADING_TIMEFRAME", "1h"),
                loop_interval=int(os.getenv("TRADING_LOOP_INTERVAL", "300")),
                min_confidence=float(os.getenv("TRADING_MIN_CONFIDENCE", "0.70")),
                initial_capital=float(os.getenv("INITIAL_CAPITAL", "10000")),
            ),
        )
    
    def get_active_exchanges(self) -> List[str]:
        """Return list of configured exchanges."""
        active = []
        if self.alpaca.is_configured:
            active.append("alpaca")
        if self.binance_us.is_configured:
            active.append("binance_us")
        if self.coinbase.is_configured:
            active.append("coinbase")
        return active


# Singleton instance
config = Config.from_env()
