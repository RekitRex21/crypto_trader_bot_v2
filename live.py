import time
import logging
import pandas as pd
from data import get_live_price
from features import add_features
from model import load_model
from xgb_model import predict_with_xgb
from portfolio import Portfolio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def live_trade(symbols, feature_cols, interval=60):
    portfolio = Portfolio(initial_capital=1000.0)
    trades = []

    logger.info("📡 Starting live paper trading...")

    while True:
        for symbol in symbols:
            try:
                price_data = get_live_price(symbol)
                df = pd.DataFrame([price_data])
                df = add_features(df)
                df.dropna(inplace=True)

                if df.empty:
                    logger.warning(f"No valid features for {symbol}")
                    continue

                model = load_model(symbol, model_type="xgb")
                X_live = df[feature_cols].values
                prediction = predict_with_xgb(model, X_live)
                signal = prediction[0]
                price = df["close"].iloc[-1]

                if signal == 1 and portfolio.can_buy(symbol):
                    qty = portfolio.buy(symbol, price)
                    if qty > 0:
                        trades.append({"symbol": symbol, "action": "BUY", "price": price, "qty": qty, "time": time.time()})
                        logger.info(f"🟢 BUY {symbol} at {price:.4f} | Qty: {qty}")

                elif signal == -1 and portfolio.can_sell(symbol):
                    qty, pnl = portfolio.sell(symbol, price)
                    trades.append({"symbol": symbol, "action": "SELL", "price": price, "qty": qty, "pnl": pnl, "time": time.time()})
                    logger.info(f"🔴 SELL {symbol} at {price:.4f} | Qty: {qty} | PnL: {pnl:.2f}")

            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")

        logger.info(f"💼 Current Balance: ${portfolio.capital:.2f}")
        time.sleep(interval)

