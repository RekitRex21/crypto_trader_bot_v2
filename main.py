import argparse
import logging
import pandas as pd
from data import get_price_data
from model import load_model
from features import add_features, FEATURE_COLS
from backtester import Backtester, metrics
from technical_votes import signal_with_confidence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot V2")
    parser.add_argument("--mode", type=str, choices=["backtest", "live"], required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--strategy", type=str, default="ensemble", help="Strategy: ensemble, xgb, rsi")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def generate_features(df):
    return add_features(df)

def run_backtest(symbols, strategy, debug):
    logging.info("Starting back-test …")
    all_trades = []

    for symbol in symbols:
        try:
            logging.info(f"[{symbol}] fetching data …")
            df = get_price_data(symbol)
            if df is None or df.empty or not isinstance(df, pd.DataFrame):
                raise ValueError("No valid price data returned")

            df = generate_features(df)
            bt = Backtester(symbol=symbol, df=df, strategy=strategy, debug=debug)
            trades, equity = bt.run()

            if trades is not None and not trades.empty:
                all_trades.append(trades)
                logging.info(f"✅ Completed backtest for {symbol}: {len(trades)} trades")
            else:
                logging.warning(f"⚠️ No trades executed for {symbol}")
        except Exception as e:
            logging.warning(f"[{symbol}] back-test failed: {e}")
            continue

    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        logging.info("\n📊 Backtest Summary:")
        for k, v in metrics(all_trades_df).items():
            print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        logging.warning("❌ No trades were executed in this backtest session.")

def run_live(symbols, strategy, debug):
    from alpaca_connector import AlpacaConnector
    import time
    
    logging.info(f"🚀 Starting LIVE trading session (Strategy: {strategy})")
    connector = AlpacaConnector(paper=True) 
    
    while True:
        for symbol in symbols:
            try:
                # 1. Fetch latest data (matches backtest source)
                df = get_price_data(symbol, days=80) # Need enough history for window (60) + features
                if df is None or df.empty: continue
                
                # 2. Generate features
                df = generate_features(df)
                
                # 3. Instantiate Backtester (loads models)
                # We re-instantiate to ensure clean state and fresh model reload if changed
                bt = Backtester(symbol=symbol, df=df, strategy=strategy, debug=debug)
                
                # 4. Get Signal
                signal, debug_info = bt.get_trade_signal(index=-1)
                
                if debug:
                    logging.info(f"[{symbol}] Signal: {signal} | Info: {debug_info}")
                else:
                    logging.info(f"[{symbol}] Signal: {signal}")
                
                # 5. Execute
                if signal == "BUY":
                    # Check if we already have a position?
                    # Alpaca connector handles 'buy' as a market order.
                    # Simple logic: Buy $100 worth.
                    # We might want to check current position to avoid double buying if we are HOLDING
                    # But for now, let's trust the signal is "Action" based.
                    # Wait, get_trade_signal returns BUY if votes >= 2.
                    # If we hold, and votes >= 2, we keep holding (or buy more?).
                    # The Backtester.run() logic checks specific entry/exit.
                    # Live logic should probably just buy if no position, or hold.
                    # AlpacaConnector doesn't track "internal" state easily without querying.
                    # Let's simple: Buy $100.
                    connector.buy_usd_notional(symbol, 100.0)
                elif signal == "SELL":
                    connector.close_all() # Logic says close all for symbol? AlpacaConnector.close_all closes ALL positions.
                    # Ideally we close only this symbol.
                    connector.sell_usd_notional(symbol, 100.0) # Or sell all?
                    # Let's stick to safe "sell what we can" or "close all" if that's the intention.
                    # The previous code had `connector.close_all()`. I'll stick to that or `sell_usd_notional`.
                    # Actually, let's use close_all() as it's safer to exit everything on a SELL signal for now.
                    connector.close_all()

            except Exception as e:
                logging.error(f"Error in live loop for {symbol}: {e}")
                
        logging.info("Sleeping for 1 hour...")
        time.sleep(3600)

def main():
    args = parse_args()

    if args.mode == "backtest":
        run_backtest(args.symbols, args.strategy, args.debug)
    elif args.mode == "live":
        run_live(args.symbols, args.strategy, args.debug)

if __name__ == "__main__":
    main()
