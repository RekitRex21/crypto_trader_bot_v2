# backtester.py
# ----------------------------------------------------------------------
"""Walk-forward back-testing engine and performance metrics."""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from portfolio import Portfolio
from ensemble import EnsemblePredictor
from technical_votes import predict as tech_predict
from xgb_model import load_xgb, train_xgb
from model import load_models, train_model, FEATURE_COLS


# ------------------------------------------------------------------ #
def _create_sequence(df: pd.DataFrame, idx: int, window: int = 60):
    """Return (1, window, n_features) array for the LSTM."""
    if idx < window:
        return None
    seq = df.iloc[idx - window : idx][FEATURE_COLS].values
    return seq.reshape(1, window, -1)


class Backtester:
    """Single-symbol walk-forward back-test."""

    def __init__(
        self,
        symbol: str,
        df_feat: pd.DataFrame,
        model_cache: Dict[str, Dict],
        *,
        start_idx: int = 200,
        debug: bool = False,
    ):
        self.symbol = symbol
        self.df = df_feat.reset_index(drop=True)
        self.start_idx = start_idx
        self.debug = debug

        cache = model_cache.get(symbol, {})
        self.lstm_model = cache.get("model")
        self.scaler = cache.get("scaler")
        self.xgb_model = load_xgb(symbol)

        # Adaptive blender
        self.ensemble = EnsemblePredictor({"lstm": 0.5, "xgb": 0.3, "technical": 0.2})

    # -------------------------------------------------------------- #
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the back-test.
        Returns: (trades_df, equity_curve_df)
        """
        # train on-the-fly if models missing
        if self.lstm_model is None or self.scaler is None:
            train_model(self.symbol, self.df, FEATURE_COLS)
            self.lstm_model, self.scaler = load_models([self.symbol])[self.symbol].values()
        if self.xgb_model is None:
            train_xgb(self.symbol, self.df, FEATURE_COLS)
            self.xgb_model = load_xgb(self.symbol)

        port = Portfolio(1_000)
        open_pos: Optional[dict] = None
        trades_count = 0

        for i, row in self.df.iterrows():
            if i < self.start_idx:
                continue

            x_seq = _create_sequence(self.df, i)
            if x_seq is None:
                continue

            # ---------- predictions ----------
            lstm_norm = self.lstm_model.predict(x_seq, verbose=0)[0][0]
            lstm_pred = (
                lstm_norm * (self.scaler.data_max_[0] - self.scaler.data_min_[0])
                + self.scaler.data_min_[0]
            )

            xgb_pred = float(
                self.xgb_model.predict(row[FEATURE_COLS].values.reshape(1, -1))[0]
            )

            tech_pred = row["Close"] * (1 + (tech_predict(row) - 0.5) * 0.02)

            preds = {"lstm": lstm_pred, "xgb": xgb_pred, "technical": tech_pred}
            self.ensemble.update_errors(preds, row["Close"])
            current_price = row["Close"]
            price_pred = self.ensemble.predict(preds)
            price_pred = np.clip(price_pred, current_price*0.9, current_price*1.1)
            pred_gain = (price_pred - current_price) / current_price

            if self.debug and i % 5 == 0:
                print(
                    f"{self.symbol} {row['Date'].date()} | "
                    f"Px={current_price:.2f} → {price_pred:.2f} "
                    f"Δ={pred_gain:+.2%}"
                )

            # ---------- entry ----------
            if open_pos is None and pred_gain > 0.012:  # >= 1 % upside
                tp_price = current_price + 3 * row["ATR"]
                sl_price = current_price - 1.5 * row["ATR"]

                open_pos = port.enter_trade(
                    self.symbol,
                    row["Date"],
                    current_price,
                    row["ATR"],
                    tp_price,
                )
                open_pos["Stop"] = sl_price  # override default stop

            # ---------- exit ----------
            elif open_pos:
                if row["High"] >= open_pos["TP"] or row["Low"] <= open_pos["Stop"]:
                    exit_px = (
                        open_pos["TP"]
                        if row["High"] >= open_pos["TP"]
                        else open_pos["Stop"]
                    )
                    port.exit_trade(open_pos, row["Date"], exit_px)
                    open_pos = None
                    trades_count += 1

            port.log_equity(row["Date"])

            if self.debug and (i - self.start_idx) % 100 == 0:
                print(
                    f"{self.symbol} progress {i - self.start_idx}/{len(self.df) - self.start_idx} "
                    f"| trades {trades_count}"
                )

        return port.trades_df(), port.equity_df()


# ------------------------------------------------------------------ #
def metrics(df_trades: pd.DataFrame, initial_capital: float = 1_000) -> dict:
    """Basic performance statistics."""
    if df_trades.empty:
        return {k: 0 for k in (
            "final_capital",
            "total_return_pct",
            "win_rate",
            "sharpe",
            "max_drawdown",
            "avg_hold_hrs",
        )}

    final_cap = df_trades["Capital After Trade"].iloc[-1]
    returns = df_trades["Profit"] / (df_trades["Buy Price"] * df_trades["Qty"])
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(365 * 24)
        if returns.std() > 0
        else 0
    )
    mdd = (
        df_trades["Capital After Trade"].cummax() - df_trades["Capital After Trade"]
    ).max()
    hold_hrs = (
        (df_trades["Sell Date"] - df_trades["Buy Date"])
        .dt.total_seconds()
        .mean()
        / 3600
    )

    return {
        "final_capital": final_cap,
        "total_return_pct": (final_cap - initial_capital) / initial_capital * 100,
        "win_rate": (df_trades["Profit"] > 0).mean() * 100,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "avg_hold_hrs": hold_hrs,
    }
