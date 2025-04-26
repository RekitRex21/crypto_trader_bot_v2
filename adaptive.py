"""Detect market regime and weight ensemble models."""
import numpy as np

def detect_regime(volatility: float, adx: float):
    if volatility > 0.05:
        return "volatile"
    if adx > 25:
        return "trending"
    return "normal"

def regime_weights(regime: str):
    if regime == "volatile":
        return {"lstm": 0.4, "xgb": 0.3, "technical": 0.3}
    if regime == "trending":
        return {"lstm": 0.6, "xgb": 0.2, "technical": 0.2}
    return {"lstm": 0.5, "xgb": 0.3, "technical": 0.2}