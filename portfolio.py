"""Capital management, ATR‑based sizing, TP/SL, fee & slippage."""
from datetime import datetime
import pandas as pd
FEE_RATE = 0.001  # 0.1 % each side
SLIPPAGE = 0.0005
RISK_PCT = 0.05   # risk 2 % of capital per trade

class Portfolio:
    def __init__(self, capital: float = 1000):
        self.capital = capital
        self.equity_curve = []  # list of (date, capital)
        self.trades = []
    def _cost(self, price):
        return price * (1 + FEE_RATE + SLIPPAGE)
    def _proceeds(self, price):
        return price * (1 - FEE_RATE - SLIPPAGE)
    def enter_trade(self, symbol: str, date, price, atr, take_profit):
        position_value = self.capital * RISK_PCT
        qty = position_value / self._cost(price)
        stop_loss = price - 2 * atr
        self.capital -= position_value
        return {
            "Coin": symbol,
            "Buy Date": date,
            "Buy Price": price,
            "Qty": qty,
            "Stop": stop_loss,
            "TP": take_profit,
        }
    def exit_trade(self, pos, date, price):
        proceeds = self._proceeds(price) * pos["Qty"]
        buy_cost = pos["Qty"] * self._cost(pos["Buy Price"])
        profit = proceeds - buy_cost
        self.capital += proceeds
        self.trades.append({
            **pos,
            "Sell Date": date,
            "Sell Price": price,
            "Profit": profit,
            "Capital After Trade": self.capital,
        })
    def log_equity(self, date):
        self.equity_curve.append({"Date": date, "Capital": self.capital})
    def trades_df(self):
        return pd.DataFrame(self.trades)
    def equity_df(self):
        return pd.DataFrame(self.equity_curve)