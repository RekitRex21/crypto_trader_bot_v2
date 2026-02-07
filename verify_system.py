import asyncio
import logging
import sys
from core.config import config
from core.database import db
from core.logger import trading_logger
from exchange_connectors import get_exchange_manager
from strategies import VisionEnhancedEnsemble
from telegram_interface import TradingTelegramBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verifier")

async def verify():
    logger.info("🔍 System Verification Started")
    
    # 1. Database Check
    try:
        logger.info("📡 Testing Database Connection...")
        # Simple query
        db.log_system_event("INFO", "Verifier", "Testing DB connection")
        logger.info("✅ Database connection successful and logging works.")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
    
    # 2. Exchange Manager Check
    try:
        logger.info("🏢 Testing Exchange Manager...")
        manager = get_exchange_manager()
        active = manager.list_active()
        logger.info(f"✅ Active exchanges: {active}")
        
        primary = manager.get_primary()
        if primary:
            # Test account info fetch if keys are there
            try:
                acc = primary.get_account_info()
                logger.info(f"✅ Primary ({primary.exchange_name}) account info: {acc.get('equity')}")
            except Exception as fe:
                logger.warning(f"⚠️ Could not fetch account info (likely invalid keys): {fe}")
    except Exception as e:
        logger.error(f"❌ Exchange Manager failed: {e}")
        
    # 3. Strategy Check (Fallback logic)
    try:
        logger.info("🧠 Testing Ensemble Strategy Fallbacks...")
        strategy = VisionEnhancedEnsemble()
        logger.info(f"✅ Strategy status: {strategy.get_status()}")
        
        # Mock DF for signal test
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0], 'volume': [1000.0]
        })
        # Add basic indicators
        df['rsi'] = 50.0
        df['sma_20'] = 100.0
        df['sma_50'] = 90.0
        
        signal = strategy.get_ensemble_signal(df)
        logger.info(f"✅ Signal generation successful (Fallback Mode): {signal['action']}")
    except Exception as e:
        logger.error(f"❌ Strategy test failed: {e}")

    # 4. Telegram Check (Init only)
    try:
        logger.info("📲 Testing Telegram Bot initialization...")
        if config.telegram.is_configured:
            bot = TradingTelegramBot(None)
            logger.info("✅ Telegram Bot initialized successfully.")
        else:
            logger.info("ℹ️ Telegram not configured, skipping.")
    except Exception as e:
        logger.error(f"❌ Telegram init failed: {e}")

    logger.info("🏁 Verification Complete")

if __name__ == "__main__":
    asyncio.run(verify())
