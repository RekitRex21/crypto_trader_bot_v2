# main.py
# ----------------------------------------------------------------------
"""CLI entry point for back-testing or live paper trading."""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv(".env")
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich import print

from data import get_crypto_ohlcv
from features import add_indicators, FEATURE_COLS
from model import load_models, train_model
from xgb_model import train_xgb
from backtester import Backtester, metrics
from portfolio import Portfolio
from ensemble import EnsemblePredictor
from live import CONNECTORS

# ---------------- logging / env ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
def _backtest_cli(args):
    logger.info("Starting back-test …")

    symbols = args.symbols
    model_cache = load_models(symbols)
    all_trades = []

    for sym in symbols:
        logger.info(f"[{sym}] fetching data …")
        raw = get_crypto_ohlcv(sym)
        df_feat = add_indicators(raw)

        # ---------- wrapper so trades is always defined --------------
        for attempt in (1, 2):          # max two tries per symbol
            try:
                bt = Backtester(
                    sym,
                    df_feat,
                    model_cache,
                    start_idx=args.warmup if hasattr(args, "warmup") else 120,
                    debug=args.debug,
                )
                trades, _ = bt.run()
                break                   # success → exit retry-loop
            except Exception as e:
                logger.warning(f"[{sym}] back-test failed (attempt {attempt}): {e}")
                if attempt == 1:
                    # try once more after training models
                    logger.info(f"[{sym}] training fresh models …")
                    train_model(sym, df_feat, FEATURE_COLS)
                    train_xgb(sym, df_feat, FEATURE_COLS)
                    model_cache = load_models([sym]) | model_cache
                    continue
                # second failure → give up
                trades = pd.DataFrame()
                break

        # ---------- book-keeping ------------------------------------
        if not trades.empty:
            all_trades.append(trades)
            print(f"\n✅ Completed [{sym}] with {len(trades)} trades")
            for k, v in metrics(trades).items():
                print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
        else:
            print(f"\n⚠️  No trades executed for {sym}")

    if all_trades:
        combined = pd.concat(all_trades)
        combined.to_csv("trades_output.csv", index=False)
        print("\n📦 All trades saved to trades_output.csv")

# ------------------------------------------------------------------ #
def _live_cli(args):
    """Simple continuous paper-trading loop."""
    from technical_votes import predict as tech_predict
    from backtester import _create_sequence

    connector_cls = CONNECTORS[args.broker]
    broker = connector_cls(paper=True)

    symbols = args.symbols
    model_cache = load_models(symbols)

    # ensure models exist
    for sym in symbols:
        if sym not in model_cache:
            raw = get_crypto_ohlcv(sym)
            df_feat = add_indicators(raw)
            train_model(sym, df_feat, FEATURE_COLS)
            train_xgb(sym, df_feat, FEATURE_COLS)
    model_cache = load_models(symbols)

    print("[yellow]🎯 Live paper loop — Ctrl-C to exit[/yellow]")
    try:
        while True:
            for sym in symbols:
                price = broker.latest_price(f"{sym}/USD")
                df_live = add_indicators(get_crypto_ohlcv(sym, days=90))
                if len(df_live) < 61:
                    continue

                # --- predictions ---
                cache = model_cache[sym]
                seq = _create_sequence(df_live, len(df_live) - 1)
                lstm_pred = cache["scaler"].inverse_transform(
                    [[cache["model"].predict(seq, verbose=0)[0][0]]]
                )[0][0]

                from xgb_model import load_xgb
                xgb_model = load_xgb(sym)
                xgb_pred = float(
                    xgb_model.predict(df_live.iloc[-1][FEATURE_COLS].values.reshape(1, -1))[0]
                )

                tech_pred = price * (1 + (tech_predict(df_live.iloc[-1]) - 0.5) * 0.02)

                blended = EnsemblePredictor({"lstm": 1, "xgb": 1, "technical": 1}).predict(
                    {"lstm": lstm_pred, "xgb": xgb_pred, "technical": tech_pred}
                )
                gain = (blended - price) / price

                # --- trade logic ---
                if gain > 0.002:
                    broker.buy_usd_notional(f"{sym}/USD", 100)
                    print(f"[green]{sym} BUY 100 USD notional at {price:.2f} (pred {blended:.2f})[/green]")

    except KeyboardInterrupt:
        print("\nStopped.")

def cli():
    parser = argparse.ArgumentParser(description="Crypto Bot v2.5")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTC"], help="Coin tickers")
    parser.add_argument("--broker", default="alpaca", choices=CONNECTORS.keys())
    parser.add_argument("--capital", type=float, default=10_000, help="Start equity (USD)")
    parser.add_argument("--debug", action="store_true", help="Verbose back-test output")
    parser.add_argument("--strategy", default="ensemble", choices=["ensemble"],
                        help="(reserved for future)")
    args = parser.parse_args()

    if args.mode == "backtest":
        _backtest_cli(args)
    else:
        _live_cli(args)


if __name__ == "__main__":
    cli()
