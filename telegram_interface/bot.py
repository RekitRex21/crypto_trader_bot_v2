"""
Telegram bot interface for trading bot control and monitoring.
"""
import logging
import asyncio
from typing import List, Optional, Dict, Any
from io import BytesIO

from core.config import config

logger = logging.getLogger(__name__)

# Telegram imports with error handling
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram interface disabled.")


class TradingTelegramBot:
    """
    Telegram bot for trading control and monitoring.
    
    Commands:
    - /start - Initialize bot
    - /stop - Stop trading
    - /status - Portfolio status
    - /positions - Open positions
    - /chart <symbol> - View chart
    - /analyze <symbol> - Run AI analysis
    - /performance - Performance metrics
    - /config - View/edit configuration
    """
    
    def __init__(self, trading_engine=None):
        """
        Initialize Telegram bot.
        
        Args:
            trading_engine: Reference to main trading engine
        """
        if not TELEGRAM_AVAILABLE:
            logger.error("Telegram bot cannot be initialized - missing dependencies")
            return
        
        self.token = config.telegram.bot_token
        self.allowed_users = config.telegram.allowed_user_ids
        self.trading_engine = trading_engine
        self.app: Optional[Application] = None
        
        if self.token:
            self._setup_application()
    
    def _setup_application(self):
        """Setup Telegram application and handlers."""
        self.app = Application.builder().token(self.token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("chart", self.cmd_chart))
        self.app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        
        # Callback query handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        logger.info("✅ Telegram bot handlers registered")
    
    def _check_auth(self, update: Update) -> bool:
        """Check if user is authorized."""
        user_id = update.effective_user.id
        if not self.allowed_users or user_id in self.allowed_users:
            return True
        logger.warning(f"Unauthorized access attempt from user {user_id}")
        return False
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not self._check_auth(update):
            await update.message.reply_text("❌ Unauthorized")
            return
        
        if self.trading_engine:
            self.trading_engine.is_running = True
        
        message = """
🤖 **Crypto Trading Bot V3 Activated**

Available commands:
• /status - View portfolio status
• /positions - View open positions
• /chart BTC - View price chart
• /analyze ETH - Run AI analysis
• /performance - View metrics
• /stop - Stop trading

Bot is now monitoring markets.
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("📈 Positions", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("🔍 Analyze BTC", callback_data="analyze_BTC"),
                InlineKeyboardButton("🔍 Analyze ETH", callback_data="analyze_ETH"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command."""
        if not self._check_auth(update):
            return
        
        if self.trading_engine:
            self.trading_engine.is_running = False
        
        await update.message.reply_text("🛑 Trading stopped. Use /start to resume.")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._check_auth(update):
            return
        
        status = self._get_status()
        
        message = f"""
📊 **Bot Status**

🤖 State: {'🟢 ACTIVE' if status.get('is_running') else '🔴 STOPPED'}
💰 Equity: ${status.get('equity', 0):,.2f}
📈 Daily P&L: {status.get('daily_pnl_percent', 0):+.2f}%
🎯 Win Rate: {status.get('win_rate', 0):.1f}%
📊 Trades Today: {status.get('trades_today', 0)}

**Exchanges Active:** {', '.join(status.get('exchanges', ['None']))}
**Strategy:** Vision Ensemble V3
        """
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if not self._check_auth(update):
            return
        
        positions = self._get_positions()
        
        if not positions:
            await update.message.reply_text("📭 No open positions")
            return
        
        lines = ["📈 **Open Positions**\n"]
        
        for pos in positions:
            emoji = "🟢" if pos.get('unrealized_pnl', 0) >= 0 else "🔴"
            lines.append(
                f"{emoji} **{pos['symbol']}** ({pos['exchange']})\n"
                f"   Qty: {pos['quantity']:.6f}\n"
                f"   Entry: ${pos['entry_price']:,.2f}\n"
                f"   Current: ${pos['current_price']:,.2f}\n"
                f"   P&L: ${pos['unrealized_pnl']:+,.2f}\n"
            )
        
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    
    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chart command."""
        if not self._check_auth(update):
            return
        
        symbol = context.args[0] if context.args else "BTC"
        await update.message.reply_text(f"📊 Generating chart for {symbol}...")
        
        try:
            chart_bytes = self._generate_chart(symbol)
            if chart_bytes:
                await update.message.reply_photo(
                    photo=BytesIO(chart_bytes),
                    caption=f"📊 {symbol}/USD Chart"
                )
            else:
                await update.message.reply_text(f"❌ Could not generate chart for {symbol}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command - run AI analysis."""
        if not self._check_auth(update):
            return
        
        symbol = context.args[0] if context.args else "BTC"
        await update.message.reply_text(f"🔍 Running AI analysis for {symbol}...")
        
        try:
            analysis = self._run_analysis(symbol)
            
            emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}[analysis['action']]
            
            message = f"""
{emoji} **AI Analysis: {symbol}/USD**

**Signal:** {analysis['action'].upper()}
**Confidence:** {analysis['confidence']:.1%}

**Model Votes:**
• LSTM: {analysis['model_signals'].get('lstm', {}).get('signal', 'N/A')}
• XGBoost: {analysis['model_signals'].get('xgboost', {}).get('signal', 'N/A')}
• CNN Vision: {analysis['model_signals'].get('cnn', {}).get('signal', 'N/A')}

**Reasoning:** {analysis.get('reasoning', 'N/A')}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Analysis error: {str(e)}")
    
    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command."""
        if not self._check_auth(update):
            return
        
        days = int(context.args[0]) if context.args else 30
        metrics = self._get_performance(days)
        
        message = f"""
📈 **Performance Report ({days} days)**

📊 Total Trades: {metrics.get('total_trades', 0)}
🎯 Win Rate: {metrics.get('win_rate', 0):.1%}
💰 Total P&L: ${metrics.get('total_pnl', 0):+,.2f}
📈 Avg Return: {metrics.get('avg_return', 0):.2%}
        """
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        message = """
🤖 **Crypto Bot V3 Commands**

/start - Start trading bot
/stop - Stop trading
/status - Portfolio status
/positions - View open positions
/chart <symbol> - Price chart
/analyze <symbol> - AI analysis
/performance [days] - Metrics
/help - This message
        """
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()
        
        if not self._check_auth(update):
            return
        
        data = query.data
        
        if data == "status":
            await self.cmd_status(update, context)
        elif data == "positions":
            await self.cmd_positions(update, context)
        elif data.startswith("analyze_"):
            symbol = data.split("_")[1]
            context.args = [symbol]
            await self.cmd_analyze(update, context)
    
    # ==================== NOTIFICATION METHODS ====================
    
    async def notify_trade(self, trade_data: Dict[str, Any]):
        """Send trade execution alert."""
        emoji = "🟢" if trade_data.get('side') == 'buy' else "🔴"
        
        message = f"""
{emoji} **Trade Executed**

📊 {trade_data.get('symbol')}
💼 {trade_data.get('side', '').upper()} @ ${trade_data.get('entry_price', 0):,.2f}
📦 Quantity: {trade_data.get('quantity', 0):.6f}
🎯 Stop Loss: ${trade_data.get('stop_loss', 0):,.2f}
🎯 Take Profit: ${trade_data.get('take_profit', 0):,.2f}
📊 Confidence: {trade_data.get('confidence', 0):.1%}
        """
        
        await self._send_to_all(message)
    
    async def notify_alert(self, alert_type: str, message: str, level: str = 'INFO'):
        """Send system alert."""
        emoji = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '🛑', 'SUCCESS': '✅'}.get(level, 'ℹ️')
        formatted = f"{emoji} **{alert_type}**\n\n{message}"
        await self._send_to_all(formatted)
    
    async def _send_to_all(self, message: str):
        """Send message to all allowed users."""
        if not self.app:
            return
        
        for user_id in self.allowed_users:
            try:
                await self.app.bot.send_message(
                    chat_id=user_id, 
                    text=message, 
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
    
    # ==================== HELPER METHODS ====================
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current status from trading engine."""
        if self.trading_engine:
            return {
                'is_running': getattr(self.trading_engine, 'is_running', False),
                'equity': getattr(self.trading_engine, 'equity', 0),
                'daily_pnl_percent': 0,
                'win_rate': 0,
                'trades_today': 0,
                'exchanges': getattr(self.trading_engine, 'active_exchanges', []),
            }
        return {'is_running': False, 'equity': 0, 'exchanges': []}
    
    def _get_positions(self) -> List[Dict]:
        """Get open positions from trading engine."""
        if self.trading_engine and hasattr(self.trading_engine, 'get_positions'):
            return self.trading_engine.get_positions()
        return []
    
    def _generate_chart(self, symbol: str) -> bytes:
        """Generate chart image bytes."""
        if self.trading_engine and hasattr(self.trading_engine, 'generate_chart'):
            return self.trading_engine.generate_chart(symbol)
        return b''
    
    def _run_analysis(self, symbol: str) -> Dict[str, Any]:
        """Run ensemble analysis."""
        if self.trading_engine and hasattr(self.trading_engine, 'run_analysis'):
            return self.trading_engine.run_analysis(symbol)
        return {'action': 'hold', 'confidence': 0, 'model_signals': {}, 'reasoning': 'No engine'}
    
    def _get_performance(self, days: int) -> Dict[str, Any]:
        """Get performance metrics."""
        if self.trading_engine and hasattr(self.trading_engine, 'get_performance'):
            return self.trading_engine.get_performance(days)
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_return': 0}
    
    # ==================== LIFECYCLE ====================
    
    async def start(self):
        """Start the bot."""
        if self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            logger.info("🤖 Telegram bot started")
    
    async def stop(self):
        """Stop the bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("🤖 Telegram bot stopped")
    
    def run(self):
        """Run bot in blocking mode."""
        if self.app:
            self.app.run_polling()
