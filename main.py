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
                df = get_price_data(symbol, days=81) # Need enough history for window (60) + features
                if df is None or df.empty:
                    logging.warning(f"[{symbol}] No data fetched")
                    continue
                
                # 2. Generate features
                df = generate_features(df)
                
                # 3. Instantiate Backtester (loads models)
                bt = Backtester(symbol=symbol, df=df, strategy=strategy, debug=debug)
                
                # 4. Get Signal
                signal, debug_info = bt.get_trade_signal(index=-1)
                
                msg = f"[{symbol}] Price: {debug_info['price']:.2f} | Votes: {debug_info['votes']} | Signal: {signal}"
                logging.info(msg)
                
                # 5. Execute
                if signal == "BUY":
                    logging.info(f"🚀 Executing BUY for {symbol}")
                    connector.buy_usd_notional(symbol, 100.0)
                elif signal == "SELL":
                    # Check if we have a position before closing
                    try:
                        pos = connector.client.get_open_position(symbol)
                        logging.info(f"🔴 Signal SELL for {symbol}. Closing position of {pos.qty} units.")
                        connector.close_all() # Or specifically close this one
                    except Exception:
                        logging.info(f"⚪ Signal SELL for {symbol} but no open position found. Skipping.")
                
            except Exception as e:
                logging.error(f"Error in live loop for {symbol}: {e}")
                
        if debug:
            logging.info("Deep dive complete. Exiting search for one-shot verification.")
            break
        
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
