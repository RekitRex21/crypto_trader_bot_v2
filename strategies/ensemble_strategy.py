"""
Vision-enhanced ensemble strategy.
Combines LSTM, XGBoost, and CNN vision models for trading signals.
"""
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import config
from vision_engine import ChartImageGenerator, get_pattern_recognizer

logger = logging.getLogger(__name__)

# Optional ML imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class VisionEnhancedEnsemble:
    """
    Multi-model ensemble combining:
    - LSTM (30%): Temporal sequence patterns
    - XGBoost (30%): Nonlinear feature relationships  
    - CNN Vision (40%): Visual chart pattern recognition
    """
    
    def __init__(
        self, 
        model_paths: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble with model paths.
        
        Args:
            model_paths: Dict with 'lstm', 'xgboost', 'cnn' keys pointing to model files
            weights: Custom weights for each model (must sum to 1.0)
        """
        self.model_paths = model_paths or {}
        self.weights = weights or {
            'lstm': 0.30,
            'xgboost': 0.30,
            'cnn': 0.40,
        }
        
        # Load models
        self.lstm_model = None
        self.xgb_model = None
        self.cnn_recognizer = None
        self.chart_gen = ChartImageGenerator()
        
        self._load_models()
    
    def _load_models(self):
        """Load all available models."""
        # LSTM
        lstm_path = self.model_paths.get('lstm')
        if lstm_path and os.path.exists(lstm_path) and TF_AVAILABLE:
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                logger.info(f"✅ Loaded LSTM model from {lstm_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load LSTM model: {e}")
        
        # XGBoost
        xgb_path = self.model_paths.get('xgboost')
        if xgb_path and os.path.exists(xgb_path) and JOBLIB_AVAILABLE:
            try:
                self.xgb_model = joblib.load(xgb_path)
                logger.info(f"✅ Loaded XGBoost model from {xgb_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load XGBoost model: {e}")
        
        # CNN Vision
        cnn_path = self.model_paths.get('cnn')
        if cnn_path and os.path.exists(cnn_path):
            try:
                self.cnn_recognizer = get_pattern_recognizer(cnn_path)
                logger.info(f"✅ Loaded CNN model from {cnn_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load CNN model: {e}")
        else:
            # Initialize empty recognizer
            self.cnn_recognizer = get_pattern_recognizer(None)
    
    def get_ensemble_signal(
        self, 
        df: pd.DataFrame, 
        min_confidence: float = 0.70
    ) -> Dict[str, Any]:
        """
        Generate trading signal from ensemble of all models.
        
        Args:
            df: OHLCV dataframe with indicators
            min_confidence: Minimum confidence threshold for action
        
        Returns:
            {
                'action': 'buy'|'sell'|'hold',
                'confidence': float,
                'model_signals': dict,
                'reasoning': str
            }
        """
        model_signals = {}
        votes = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}
        
        # 1. LSTM prediction (temporal patterns)
        if self.lstm_model is not None:
            lstm_result = self._get_lstm_signal(df)
            model_signals['lstm'] = lstm_result
            votes[lstm_result['signal']] += self.weights['lstm']
        else:
            # Fallback to technical analysis if no LSTM
            tech_signal = self._get_technical_signal(df)
            model_signals['lstm'] = {'signal': tech_signal, 'value': 0.5, 'fallback': True}
            votes[tech_signal] += self.weights['lstm']
        
        # 2. XGBoost prediction (nonlinear features)
        if self.xgb_model is not None:
            xgb_result = self._get_xgb_signal(df)
            model_signals['xgboost'] = xgb_result
            votes[xgb_result['signal']] += self.weights['xgboost']
        else:
            # Fallback
            tech_signal = self._get_technical_signal(df)
            model_signals['xgboost'] = {'signal': tech_signal, 'value': 0.5, 'fallback': True}
            votes[tech_signal] += self.weights['xgboost']
        
        # 3. CNN Vision prediction (chart patterns)
        cnn_result = self._get_cnn_signal(df)
        model_signals['cnn'] = cnn_result
        votes[cnn_result['signal']] += self.weights['cnn']
        
        # Determine winning signal
        winning_signal = max(votes, key=votes.get)
        ensemble_confidence = votes[winning_signal]
        
        # Generate action
        if ensemble_confidence >= min_confidence:
            if winning_signal == 'bullish':
                action = 'buy'
            elif winning_signal == 'bearish':
                action = 'sell'
            else:
                action = 'hold'
        else:
            action = 'hold'
        
        # Generate reasoning
        reasoning = self._generate_reasoning(model_signals, winning_signal, action, ensemble_confidence)
        
        return {
            'action': action,
            'confidence': min(ensemble_confidence, 1.0),
            'model_signals': model_signals,
            'votes': votes,
            'reasoning': reasoning
        }
    
    def _get_lstm_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get LSTM prediction."""
        try:
            # Prepare sequence input
            features = ['close', 'volume']
            if 'rsi' in df.columns:
                features.extend(['rsi', 'macd'])
            
            lookback = 60
            available_features = [f for f in features if f in df.columns]
            
            if len(df) < lookback:
                return {'signal': 'neutral', 'value': 0.5}
            
            sequence = df[available_features].iloc[-lookback:].values
            # Normalize
            sequence = (sequence - np.mean(sequence, axis=0)) / (np.std(sequence, axis=0) + 1e-8)
            sequence = sequence.reshape(1, lookback, len(available_features))
            
            pred = self.lstm_model.predict(sequence, verbose=0)[0][0]
            
            if pred > 0.55:
                signal = 'bullish'
            elif pred < 0.45:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            return {'signal': signal, 'value': float(pred)}
            
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return {'signal': 'neutral', 'value': 0.5, 'error': str(e)}
    
    def _get_xgb_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get XGBoost prediction."""
        try:
            # Prepare feature vector
            features = ['rsi', 'macd', 'signal', 'sma_20', 'sma_50', 'volume']
            available = [f for f in features if f in df.columns]
            
            if not available:
                return {'signal': 'neutral', 'value': 0.5}
            
            X = df[available].iloc[-1:].fillna(0).values
            
            pred_proba = self.xgb_model.predict_proba(X)[0]
            pred = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
            
            if pred > 0.55:
                signal = 'bullish'
            elif pred < 0.45:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            return {'signal': signal, 'value': float(pred)}
            
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return {'signal': 'neutral', 'value': 0.5, 'error': str(e)}
    
    def _get_cnn_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get CNN vision prediction."""
        try:
            # Generate chart image
            chart_img = self.chart_gen.ohlcv_to_image(df, lookback=100)
            
            # Predict pattern
            result = self.cnn_recognizer.predict_pattern(chart_img)
            
            return {
                'signal': result['signal'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            }
            
        except Exception as e:
            logger.warning(f"CNN prediction failed: {e}")
            return {'signal': 'neutral', 'confidence': 0.33, 'error': str(e)}
    
    def _get_technical_signal(self, df: pd.DataFrame) -> str:
        """Fallback technical analysis signal."""
        try:
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 70:
                    return 'bearish'
                elif rsi < 30:
                    return 'bullish'
            
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma20 = df['sma_20'].iloc[-1]
                sma50 = df['sma_50'].iloc[-1]
                if sma20 > sma50:
                    return 'bullish'
                elif sma20 < sma50:
                    return 'bearish'
            
            return 'neutral'
        except Exception:
            return 'neutral'
    
    def _generate_reasoning(
        self, 
        model_signals: Dict, 
        final_signal: str, 
        action: str,
        confidence: float
    ) -> str:
        """Generate human-readable explanation."""
        reasons = []
        
        # Count model agreements
        agreements = sum(
            1 for m in model_signals.values() 
            if m.get('signal') == final_signal
        )
        reasons.append(f"{agreements}/3 models agree on {final_signal}")
        
        # CNN confidence note
        if model_signals.get('cnn', {}).get('confidence', 0) > 0.8:
            reasons.append(f"Strong visual pattern ({model_signals['cnn']['confidence']:.0%})")
        
        # Action reasoning
        if action == 'hold':
            if confidence < 0.7:
                reasons.append(f"Confidence {confidence:.0%} below threshold")
            else:
                reasons.append("Mixed/neutral signals")
        
        return " | ".join(reasons)
    
    def get_status(self) -> Dict[str, bool]:
        """Return model availability status."""
        return {
            'lstm_loaded': self.lstm_model is not None,
            'xgboost_loaded': self.xgb_model is not None,
            'cnn_loaded': self.cnn_recognizer is not None,
            'weights': self.weights
        }


# Factory function
def create_ensemble(
    lstm_path: Optional[str] = None,
    xgb_path: Optional[str] = None,
    cnn_path: Optional[str] = None
) -> VisionEnhancedEnsemble:
    """Create ensemble with specified model paths."""
    return VisionEnhancedEnsemble(
        model_paths={
            'lstm': lstm_path or 'models/lstm_btc.h5',
            'xgboost': xgb_path or 'models/xgb_btc.pkl',
            'cnn': cnn_path or 'models/cnn_pattern.h5',
        }
    )
