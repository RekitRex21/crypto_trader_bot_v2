"""
PostgreSQL database layer with connection pooling and trade logging.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

from core.config import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """PostgreSQL connection manager with pooling."""
    
    def __init__(self):
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Create connection pool."""
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.user,
                password=config.database.password,
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def close(self):
        """Close all connections in pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")
    
    # ==================== TRADE OPERATIONS ====================
    
    def log_trade(self, trade_data: Dict[str, Any]) -> int:
        """Insert trade record and return ID."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (
                        timestamp, symbol, exchange, side, entry_price, exit_price,
                        quantity, pnl, pnl_percent, strategy, signals,
                        stop_loss, take_profit, exit_reason, fees
                    ) VALUES (
                        %(timestamp)s, %(symbol)s, %(exchange)s, %(side)s, %(entry_price)s, %(exit_price)s,
                        %(quantity)s, %(pnl)s, %(pnl_percent)s, %(strategy)s, %(signals)s,
                        %(stop_loss)s, %(take_profit)s, %(exit_reason)s, %(fees)s
                    ) RETURNING id
                """, {
                    **trade_data,
                    'signals': Json(trade_data.get('signals', {})),
                    'timestamp': trade_data.get('timestamp', datetime.utcnow()),
                })
                trade_id = cur.fetchone()[0]
                logger.info(f"📝 Trade logged: ID {trade_id}")
                return trade_id
    
    def get_recent_trades(self, limit: int = 50, exchange: Optional[str] = None) -> List[Dict]:
        """Retrieve recent trades."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if exchange:
                    cur.execute("""
                        SELECT * FROM trades 
                        WHERE exchange = %s
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (exchange, limit))
                else:
                    cur.execute("""
                        SELECT * FROM trades 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (limit,))
                return [dict(row) for row in cur.fetchall()]
    
    # ==================== POSITION OPERATIONS ====================
    
    def open_position(self, position_data: Dict[str, Any]) -> int:
        """Insert open position and return ID."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO positions (
                        symbol, exchange, side, entry_price, quantity,
                        stop_loss, take_profit, is_open
                    ) VALUES (
                        %(symbol)s, %(exchange)s, %(side)s, %(entry_price)s, %(quantity)s,
                        %(stop_loss)s, %(take_profit)s, TRUE
                    ) RETURNING id
                """, position_data)
                pos_id = cur.fetchone()[0]
                logger.info(f"📈 Position opened: ID {pos_id}")
                return pos_id
    
    def close_position(self, position_id: int, exit_price: float) -> None:
        """Mark position as closed."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE positions 
                    SET is_open = FALSE, closed_at = NOW()
                    WHERE id = %s
                """, (position_id,))
                logger.info(f"📉 Position closed: ID {position_id}")
    
    def get_open_positions(self, exchange: Optional[str] = None) -> List[Dict]:
        """Get all open positions."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if exchange:
                    cur.execute("""
                        SELECT * FROM positions 
                        WHERE is_open = TRUE AND exchange = %s
                        ORDER BY opened_at DESC
                    """, (exchange,))
                else:
                    cur.execute("""
                        SELECT * FROM positions 
                        WHERE is_open = TRUE
                        ORDER BY opened_at DESC
                    """)
                return [dict(row) for row in cur.fetchall()]
    
    # ==================== PERFORMANCE OPERATIONS ====================
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log daily performance snapshot."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO performance (
                        timestamp, equity, daily_return, win_rate,
                        sharpe_ratio, max_drawdown, total_trades
                    ) VALUES (
                        %(timestamp)s, %(equity)s, %(daily_return)s, %(win_rate)s,
                        %(sharpe_ratio)s, %(max_drawdown)s, %(total_trades)s
                    )
                """, {
                    **metrics,
                    'timestamp': datetime.utcnow(),
                })
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate performance over time window."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COALESCE(AVG(pnl_percent), 0) as avg_return,
                        COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0), 0) as win_rate,
                        COALESCE(SUM(pnl), 0) as total_pnl
                    FROM trades
                    WHERE timestamp > NOW() - INTERVAL '%s days'
                """, (days,))
                return dict(cur.fetchone())
    
    # ==================== LOGGING OPERATIONS ====================
    
    def log_system_event(self, level: str, component: str, message: str, metadata: Optional[Dict] = None) -> None:
        """Log system event to database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO system_logs (timestamp, level, component, message, metadata)
                    VALUES (NOW(), %s, %s, %s, %s)
                """, (level, component, message, Json(metadata or {})))


# Singleton instance
db = DatabaseManager()
