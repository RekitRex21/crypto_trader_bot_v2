import pandas as pd
import numpy as np
from datetime import datetime

class Portfolio:
    def __init__(self, initial_capital=1000.0):
        self.cash = initial_capital
        self.holdings = {}
        self.history = []
        self.current_timestamp = None

    def can_buy(self, symbol):
        return symbol not in self.holdings

    def can_sell(self, symbol):
        return symbol in self.holdings

    def buy(self, symbol, price, timestamp=None):
        if price <= 0 or self.cash <= 0:
            return 0
        quantity = self.cash / price
        self.holdings[symbol] = {
            "quantity": quantity,
            "entry_price": price,
            "timestamp": pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now(),
        }
        self.cash = 0
        return quantity

    def sell(self, symbol, price, timestamp=None):
        if symbol not in self.holdings or price <= 0:
            return 0, 0, 0
        data = self.holdings.pop(symbol)
        quantity = data["quantity"]
        entry_price = data["entry_price"]
        entry_time = data["timestamp"]

        proceeds = quantity * price
        pnl = proceeds - (quantity * entry_price)
        self.cash += proceeds

        # Calculate hold time in hours
        exit_time = pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now()
        hold_time = (exit_time - entry_time).total_seconds() / 3600.0

        return quantity, pnl, hold_time

    def equity_curve(self):
        equity = []
        total = self.cash
        for _ in range(len(self.history)):
            equity.append(total)
        return equity
