🚀 Crypto Trader Bot v2.5 (Aggressive AI Edition)
By RekitRex

---
📚 About:

A next-level AI crypto trading bot using:
- LSTM Neural Networks 🧠
- XGBoost Tree Boosters 🎯
- Technical Analysis 📈
- Adaptive Regimes ⚙️

Pre-trained models are included — ready to backtest or expand.

---
🛠️ Setup Instructions:

1. Unzip the folder.
2. Open a terminal inside the folder.
3. Create a virtual environment:

   python3 -m venv .venv

4. Activate the environment:

   source .venv/bin/activate

5. Install required packages:

   pip install -r requirements.txt

---
🎯 How to Run:

Example backtest on Top AI Coins:

   python main.py --mode backtest --symbols NEAR TAO ICP RNDR FET GRT AKT OCEAN AIOZ GLM --debug

Example backtest on Classics:

   python main.py --mode backtest --symbols BTC ETH SOL DOGE ADA --debug

---
📈 Results:

- Trades are automatically saved to a CSV.
- Models are saved in /models/ folder.

---
🔧 Notes:

- Ensure `.venv/` and `.csv` files stay local (they are ignored by GitHub).
- For live paper trading, edit `live.py` with Alpaca API keys.

---
🧠 Happy Trading,
RekitRex
