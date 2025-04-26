# ensemble.py
# ----------------------------------------------------------------------
"""Adaptive model blending with inverse-MAE weighting."""

from collections import deque
import numpy as np


class RollingMAE:
    def __init__(self, window: int = 30):
        self.window = window
        self.errors = deque(maxlen=window)

    def update(self, pred: float, actual: float):
        self.errors.append(abs(pred - actual))

    def mae(self) -> float:
        return np.mean(self.errors) if self.errors else np.inf


class EnsemblePredictor:
    """
    Blend model predictions.

    Parameters
    ----------
    base_weights : dict[str, float]
        Initial static weights, e.g. {"lstm": 0.5, "xgb": 0.3, "technical": 0.2}
    """

    def __init__(self, base_weights: dict[str, float]):
        self.base_weights = base_weights
        self.trackers = {k: RollingMAE() for k in base_weights}

    # -------------------------------------------------------------- #
    def update_errors(self, preds: dict[str, float], actual: float):
        """Feed real outcome to update rolling MAE trackers."""
        for k, p in preds.items():
            self.trackers[k].update(p, actual)

    def _inverse_mae_weights(self) -> dict[str, float]:
        maes = {k: self.trackers[k].mae() for k in self.base_weights}

        inv = {k: (1 / v) if v > 0 and np.isfinite(v) else 0 for k, v in maes.items()}
        total = sum(inv.values())

        # cold start → fall back to static weights
        if total == 0:
            return self.base_weights

        return {k: v / total for k, v in inv.items()}

    def predict(self, preds: dict[str, float]) -> float:
        """Return blended price prediction."""
        w = self._inverse_mae_weights()
        return sum(preds[k] * w.get(k, 0) for k in preds)
