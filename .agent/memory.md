# Persistent Memory: Crypto Bot V2

## 🎯 Current Objective
Audit the existing codebase to ensure it is in "working order" and compliant with the Antigravity Master System Prompt.

## 🏗️ Technical Architecture
- **Language**: Python
- **Exchange Integration**: Alpaca (via `alpaca_connector.py`)
- **ML Models**: 
  - LSTM (`lstm_model.py`)
  - XGBoost (`xgb_model.py`)
  - Ensemble (`ensemble.py`)
- **Logic**: Orchestrated via `main.py` and `live.py`.

## 📈 Recent Progress
- [x] Analyze codebase structure.
- [x] Initialize `specs.md` and `.agent/memory.md`.
- [x] Audit `.aiexclude` compliance.
- [/] Setup Environment (.env keys).

## ⚠️ Known Issues
- Pending audit of `.env` interactions (privacy compliance).
- Uncertain status of dependencies (need to verify `requirements.txt`).
- No unit tests visible in root (need to scan for testable logic).

## 🚀 Next Steps
- [x] Audit `.env` and create `.aiexclude`.
- [x] Verify execution environment (Python 3.12 venv).
- [x] Configure `.env` with API keys.
- [/] Deep dive into model and orchestration logic.
