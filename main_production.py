"""
Production entry point for Crypto Trading Bot V3.
Integrates all components: exchanges, strategy, risk, database, and Telegram.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from core.config import config
from core.database import db
from core.risk_manager import risk_manager
from core.logger import trading_logger
from exchange_connectors import get_exchange_manager, OrderResult
from strategies import VisionEnhancedEnsemble
from vision_engine import ChartImageGenerator
from telegram_interface import TradingTelegramBot

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine orchestrating all components.
    """
    
    def __init__(self):
        self.is_running = False
        self.equity = config.trading.initial_capital
        self.daily_pnl = 0.0
        self.trades_today = 0
        
        # Initialize components
        self.exchange_manager = None
        self.strategy = None
        self.telegram_bot = None
        self.chart_gen = ChartImageGenerator()
        
        # Active exchanges list
        self.active_exchanges: List[str] = []
    
    async def initialize(self):
        """Initialize all trading components."""
        logger.info("🚀 Initializing Trading Engine V3...")
        
        # Exchange manager
        self.exchange_manager = get_exchange_manager()
        self.active_exchanges = self.exchange_manager.list_active()
        
        if not self.active_exchanges:
            logger.warning("⚠️ No exchanges configured!")
        else:
            logger.info(f"✅ Exchanges: {', '.join(self.active_exchanges)}")
        
        # Strategy (try to load models)
        self.strategy = VisionEnhancedEnsemble(
            model_paths={
                'lstm': 'models/lstm_btc.h5',
                'xgboost': 'models/xgb_btc.pkl',
                'cnn': 'models/cnn_pattern.h5',
            }
        )
        logger.info(f"✅ Strategy initialized: {self.strategy.get_status()}")
        
        # Telegram bot
        if config.telegram.is_configured:
            self.telegram_bot = TradingTelegramBot(trading_engine=self)
            await self.telegram_bot.start()
            logger.info("✅ Telegram bot started")
        else:
            logger.info("ℹ️ Telegram not configured, skipping")
        
        # Update equity from exchanges
        await self._update_equity()
        
        logger.info(f"💰 Starting equity: ${self.equity:,.2f}")
        logger.info("🟢 Trading Engine ready")
    
    async def _update_equity(self):
        """Update equity from all exchanges."""
        total = 0.0
        for name, connector in self.exchange_manager.get_all().items():
            try:
                info = connector.get_account_info()
                total += info.get('equity', 0)
            except Exception as e:
                logger.error(f"Failed to get equity from {name}: {e}")
        
        if total > 0:
            self.equity = total
    
    async def trading_loop(self):
        """Main trading loop."""
        symbols = config.trading.symbols
        interval = config.trading.loop_interval
        
        logger.info(f"📊 Monitoring: {symbols}")
        logger.info(f"⏱️ Loop interval: {interval}s")
        
        while self.is_running:
            try:
                for symbol in symbols:
                    await self._process_symbol(symbol)
                
                # Update equity periodically
                await self._update_equity()
                
                # Reset daily counters at midnight
                risk_manager._reset_daily_counters() if datetime.now().hour == 0 else None
                
                logger.info(f"💤 Sleeping {interval}s...")
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                if self.telegram_bot:
                    await self.telegram_bot.notify_alert("System Error", str(e), "ERROR")
                await asyncio.sleep(60)  # Cooldown on error
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol through the trading pipeline."""
        logger.info(f"📊 Processing {symbol}...")
        
        # Get primary exchange
        connector = self.exchange_manager.get_primary()
        if not connector:
            logger.warning("No exchange available")
            return
        
        # Check risk constraints
        can_trade, reason = risk_manager.can_trade(self.equity, 0)
        if not can_trade:
            logger.warning(f"⚠️ Risk check failed: {reason}")
            return
        
        try:
            # Fetch market data
            df = connector.get_historical_data(symbol, config.trading.timeframe, limit=200)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return
            
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Get ensemble signal
            signal = self.strategy.get_ensemble_signal(df, config.trading.min_confidence)
            
            logger.info(
                f"[{symbol}] Signal: {signal['action'].upper()} | "
                f"Confidence: {signal['confidence']:.1%} | "
                f"Reason: {signal['reasoning']}"
            )
            
            # Execute if actionable
            if signal['action'] in ['buy', 'sell']:
                await self._execute_trade(connector, symbol, signal, df)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def _execute_trade(
        self, 
        connector, 
        symbol: str, 
        signal: Dict[str, Any], 
        df: pd.DataFrame
    ):
        """Execute trade based on signal."""
        action = signal['action']
        confidence = signal['confidence']
        
        # Calculate ATR for position sizing
        atr = self._calculate_atr(df)
        volatility = atr / df['close'].iloc[-1]
        
        # Position size
        position_size = risk_manager.calculate_position_size(
            self.equity, confidence, volatility
        )
        
        entry_price = df['close'].iloc[-1]
        direction = 'long' if action == 'buy' else 'short'
        
        # Stop loss and take profit
        stop_loss = risk_manager.calculate_stop_loss(entry_price, direction, atr)
        take_profit = risk_manager.calculate_take_profit(entry_price, stop_loss, direction)
        
        logger.info(
            f"🎯 Executing {action.upper()} {symbol} | "
            f"Size: ${position_size:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
        )
        
        # Execute order
        if action == 'buy':
            result = connector.buy_notional(symbol, position_size)
        else:
            result = connector.sell_notional(symbol, position_size)
        
        if result.success:
            logger.info(f"✅ Order filled: {result}")
            
            # Log to database
            trade_data = {
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'exchange': connector.exchange_name,
                'side': action,
                'entry_price': entry_price,
                'quantity': result.quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'VisionEnsemble',
                'signals': signal['model_signals'],
                'pnl': 0,
                'pnl_percent': 0,
                'fees': result.fees,
            }
            
            try:
                db.log_trade(trade_data)
            except Exception as e:
                logger.warning(f"Could not log trade to DB: {e}")
            
            risk_manager.log_trade({'pnl': 0})
            self.trades_today += 1
            
            # Telegram notification
            if self.telegram_bot:
                trade_data['confidence'] = confidence
                await self.telegram_bot.notify_trade(trade_data)
        else:
            logger.error(f"❌ Order failed: {result.error}")
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        try:
            import pandas_ta as ta
            
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            if macd is not None:
                df['macd'] = macd['MACD_12_26_9']
                df['signal'] = macd['MACDs_12_26_9']
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            
        except Exception as e:
            logger.warning(f"Could not add indicators: {e}")
        
        return df.dropna()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            import pandas_ta as ta
            atr = ta.atr(df['high'], df['low'], df['close'], length=period)
            return float(atr.iloc[-1]) if atr is not None else 0.0
        except Exception:
            return 0.0
    
    # ==================== TELEGRAM HELPER METHODS ====================
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions for Telegram."""
        positions = []
        if self.exchange_manager:
            for name, connector in self.exchange_manager.get_all().items():
                try:
                    for pos in connector.get_open_positions():
                        positions.append({
                            'symbol': pos.symbol,
                            'exchange': pos.exchange,
                            'quantity': pos.quantity,
                            'entry_price': pos.entry_price,
                            'current_price': pos.current_price,
                            'unrealized_pnl': pos.unrealized_pnl,
                        })
                except Exception:
                    pass
        return positions
    
    def generate_chart(self, symbol: str) -> bytes:
        """Generate chart for Telegram."""
        if self.exchange_manager:
            connector = self.exchange_manager.get_primary()
            if connector:
                df = connector.get_historical_data(f"{symbol}/USD", '1h', 100)
                if not df.empty:
                    return self.chart_gen.ohlcv_to_bytes(df)
        return b''
    
    def run_analysis(self, symbol: str) -> Dict[str, Any]:
        """Run analysis for Telegram."""
        if self.exchange_manager and self.strategy:
            connector = self.exchange_manager.get_primary()
            if connector:
                df = connector.get_historical_data(f"{symbol}/USD", '1h', 200)
                if not df.empty:
                    df = self._add_indicators(df)
                    return self.strategy.get_ensemble_signal(df)
        
        return {'action': 'hold', 'confidence': 0, 'model_signals': {}, 'reasoning': 'No data'}
    
    def get_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for Telegram."""
        try:
            return db.get_performance_metrics(days)
        except Exception:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_return': 0}
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("🛑 Shutting down Trading Engine...")
        self.is_running = False
        
        if self.telegram_bot:
            await self.telegram_bot.stop()
        
        try:
            db.close()
        except Exception:
            pass
        
        logger.info("👋 Goodbye!")


async def main():
    """Main entry point."""
    engine = TradingEngine()
    
    # Signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        engine.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await engine.initialize()
        engine.is_running = True
        await engine.trading_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    asyncio.run(main())
