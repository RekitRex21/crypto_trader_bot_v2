"""
Structured logging system with JSON formatting and optional database/Telegram integration.
"""
import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry['data'] = record.extra_data
        
        return json.dumps(log_entry)


class TradingLogger:
    """
    Centralized logging with console, file, and optional integrations.
    """
    
    def __init__(
        self, 
        name: str = "crypto_bot",
        log_file: str = "logs/trading.log",
        level: int = logging.INFO,
        json_logs: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (JSON or plain)
        try:
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            
            if json_logs:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(console_format)
            
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not create file handler: {e}")
        
        self.db = None
        self.telegram_notifier = None
    
    def set_database(self, db_manager):
        """Enable database logging."""
        self.db = db_manager
    
    def set_telegram(self, telegram_notifier):
        """Enable Telegram alerts for critical logs."""
        self.telegram_notifier = telegram_notifier
    
    def _log_with_extras(self, level: int, message: str, extra_data: Optional[Dict] = None):
        """Log with optional extra data."""
        extra = {'extra_data': extra_data} if extra_data else {}
        self.logger.log(level, message, extra=extra)
        
        # Log to database if enabled
        if self.db and level >= logging.WARNING:
            try:
                self.db.log_system_event(
                    level=logging.getLevelName(level),
                    component='trading_bot',
                    message=message,
                    metadata=extra_data
                )
            except Exception:
                pass  # Don't let logging errors break the app
    
    def debug(self, message: str, data: Optional[Dict] = None):
        self._log_with_extras(logging.DEBUG, message, data)
    
    def info(self, message: str, data: Optional[Dict] = None):
        self._log_with_extras(logging.INFO, message, data)
    
    def warning(self, message: str, data: Optional[Dict] = None):
        self._log_with_extras(logging.WARNING, message, data)
    
    def error(self, message: str, data: Optional[Dict] = None):
        self._log_with_extras(logging.ERROR, message, data)
    
    def critical(self, message: str, data: Optional[Dict] = None):
        self._log_with_extras(logging.CRITICAL, message, data)
        
        # Send Telegram alert for critical errors
        if self.telegram_notifier:
            try:
                self.telegram_notifier.send_alert('CRITICAL', message)
            except Exception:
                pass
    
    def trade(self, action: str, symbol: str, details: Dict[str, Any]):
        """Log trade execution with structured data."""
        message = f"🔄 TRADE | {action.upper()} | {symbol}"
        self.info(message, {'action': action, 'symbol': symbol, **details})
    
    def signal(self, symbol: str, signal_type: str, confidence: float, model_data: Dict):
        """Log trading signal."""
        message = f"📊 SIGNAL | {symbol} | {signal_type.upper()} | Confidence: {confidence:.1%}"
        self.info(message, {
            'symbol': symbol,
            'signal': signal_type,
            'confidence': confidence,
            'models': model_data
        })


def setup_logging(log_level: str = "INFO") -> TradingLogger:
    """Initialize and return the trading logger."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    return TradingLogger(level=level)


# Default logger instance
trading_logger = setup_logging()
