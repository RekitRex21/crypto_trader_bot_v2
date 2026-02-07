"""
Risk management system with position sizing, stop-loss, and circuit breakers.
"""
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

from core.config import config

logger = logging.getLogger(__name__)


@dataclass
class RiskManager:
    """
    Comprehensive risk management for trading operations.
    Implements position sizing, drawdown limits, and trade controls.
    """
    
    # State tracking
    peak_equity: Optional[float] = None
    daily_pnl: float = 0.0
    trades_today: int = 0
    last_loss_time: Optional[datetime] = None
    trade_history: List[Dict] = field(default_factory=list)
    current_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize with config values."""
        self.max_position_size = config.risk.max_position_size
        self.max_drawdown = config.risk.max_drawdown
        self.daily_loss_limit = config.risk.daily_loss_limit
        self.stop_loss_atr_multiplier = config.risk.stop_loss_atr_multiplier
        self.take_profit_ratio = config.risk.take_profit_ratio
        self.max_trades_per_day = config.risk.max_trades_per_day
        self.cooldown_after_loss = config.risk.cooldown_after_loss
        self.current_date = datetime.now().date()
    
    def can_trade(self, current_equity: float, portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk constraints.
        
        Returns:
            Tuple[bool, str]: (can_trade, reason)
        """
        # Reset daily counters if new day
        if datetime.now().date() != self.current_date:
            self._reset_daily_counters()
        
        # Initialize peak equity
        if self.peak_equity is None:
            self.peak_equity = current_equity
        else:
            self.peak_equity = max(self.peak_equity, current_equity)
        
        # Circuit breaker: max drawdown
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if current_drawdown >= self.max_drawdown:
                logger.warning(f"🛑 Circuit breaker triggered: Drawdown {current_drawdown:.1%}")
                return False, f"Circuit breaker: Drawdown {current_drawdown:.1%} exceeds {self.max_drawdown:.0%} limit"
        
        # Daily loss limit
        if current_equity > 0 and (self.daily_pnl / current_equity) <= -self.daily_loss_limit:
            logger.warning(f"🛑 Daily loss limit reached: {self.daily_pnl:.2f}")
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        # Max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.max_trades_per_day})"
        
        # Cooldown after loss
        if self.last_loss_time:
            elapsed = (datetime.now() - self.last_loss_time).total_seconds()
            cooldown_remaining = self.cooldown_after_loss - elapsed
            if cooldown_remaining > 0:
                return False, f"Cooldown period: {int(cooldown_remaining / 60)} minutes remaining"
        
        return True, "All risk checks passed"
    
    def calculate_position_size(
        self, 
        current_equity: float, 
        signal_confidence: float, 
        volatility: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion with adjustments.
        
        Args:
            current_equity: Current portfolio value
            signal_confidence: Model confidence (0.0 to 1.0)
            volatility: Current ATR / price ratio
        
        Returns:
            Position size in USD
        """
        # Base position size (% of equity)
        base_size = current_equity * self.max_position_size
        
        # Adjust for signal confidence (0.5 to 1.0 scales position 50%-100%)
        confidence_adjustment = 0.5 + (signal_confidence * 0.5)
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = 1.0 / (1.0 + (volatility * 10))  # Scale factor for volatility
        
        final_size = base_size * confidence_adjustment * volatility_adjustment
        
        logger.debug(f"Position sizing: base=${base_size:.2f}, conf_adj={confidence_adjustment:.2f}, vol_adj={volatility_adjustment:.2f} -> final=${final_size:.2f}")
        
        return final_size
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        direction: str, 
        atr: float
    ) -> float:
        """
        Calculate dynamic stop-loss based on ATR.
        
        Args:
            entry_price: Trade entry price
            direction: 'long' or 'short'
            atr: Average True Range value
        
        Returns:
            Stop-loss price
        """
        distance = atr * self.stop_loss_atr_multiplier
        
        if direction == 'long':
            stop_loss = entry_price - distance
        else:  # short
            stop_loss = entry_price + distance
        
        return stop_loss
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        stop_loss: float, 
        direction: str
    ) -> float:
        """
        Calculate take-profit using risk:reward ratio.
        
        Args:
            entry_price: Trade entry price
            stop_loss: Stop-loss price
            direction: 'long' or 'short'
        
        Returns:
            Take-profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.take_profit_ratio
        
        if direction == 'long':
            take_profit = entry_price + reward
        else:  # short
            take_profit = entry_price - reward
        
        return take_profit
    
    def log_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Update risk tracking after trade execution.
        
        Args:
            trade_result: Dict with 'pnl' key at minimum
        """
        self.trades_today += 1
        pnl = trade_result.get('pnl', 0)
        self.daily_pnl += pnl
        self.trade_history.append(trade_result)
        
        if pnl < 0:
            self.last_loss_time = datetime.now()
            logger.info(f"📉 Loss recorded: ${pnl:.2f}, cooldown activated")
    
    def _reset_daily_counters(self) -> None:
        """Reset counters at start of each trading day."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.last_loss_time = None
        self.current_date = datetime.now().date()
        logger.info("📅 Daily counters reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Return current risk status for reporting."""
        return {
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'cooldown_active': self.last_loss_time is not None,
            'max_drawdown_limit': self.max_drawdown,
            'daily_loss_limit': self.daily_loss_limit,
        }


# Singleton instance
risk_manager = RiskManager()
