import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from technical_votes import TechnicalVotes

class TestTechnicalLogic(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe with enough rows for technical indicators
        self.data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "macd": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            "macd_signal": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.0], # Crossover at end
            "bb_upper": [120]*11,
            "bb_lower": [90]*11,
            "rsi": [50]*11,
            "sma_10": [100]*11,
            "sma_30": [95]*11
        })

    def test_macd_buy_signal(self):
        # Ensure macd > macd_signal and prev_macd <= prev_signal
        df = self.data.copy()
        # End: macd=1.1, macd_signal=1.0. Prev: macd=1.0, macd_signal=1.1.
        tv = TechnicalVotes(df)
        self.assertEqual(tv.macd_signal(), "buy")

    def test_rsi_bounds(self):
        df = self.data.copy()
        df.loc[df.index[-1], "rsi"] = 25
        tv = TechnicalVotes(df)
        self.assertEqual(tv.rsi_signal(), "buy")
        
        df.loc[df.index[-1], "rsi"] = 75
        tv = TechnicalVotes(df)
        self.assertEqual(tv.rsi_signal(), "sell")

if __name__ == "__main__":
    unittest.main()
