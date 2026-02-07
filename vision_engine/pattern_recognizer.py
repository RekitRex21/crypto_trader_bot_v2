"""
CNN-based pattern recognizer for chart image classification.
Classifies charts as bullish, bearish, or neutral.
"""
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. CNN pattern recognition disabled.")


class PatternRecognizerCNN:
    """
    ResNet-inspired CNN for chart pattern classification.
    Outputs: bullish, bearish, or neutral signal with confidence.
    """
    
    CLASSES = ['bullish', 'bearish', 'neutral']
    
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 3):
        self.num_classes = num_classes
        self.model: Optional[Any] = None
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow required for pattern recognition")
            return
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self._build_model()
            logger.info("Built new CNN model (untrained)")
    
    def _build_model(self):
        """
        Build ResNet-inspired CNN architecture.
        Input: (224, 224, 3) normalized image
        Output: (3,) softmax probabilities
        """
        inputs = layers.Input(shape=(224, 224, 3))
        
        # Initial conv block
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Residual-style blocks
        for filters in [64, 128, 256]:
            # Main path
            shortcut = x
            x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Residual connection
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2)(x)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def load_model(self, model_path: str) -> bool:
        """Load pre-trained model from file."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Loaded CNN model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = self._build_model()
            return False
    
    def save_model(self, model_path: str):
        """Save model to file."""
        if self.model:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path)
            logger.info(f"✅ Saved CNN model to {model_path}")
    
    def predict_pattern(self, chart_image: np.ndarray) -> Dict[str, Any]:
        """
        Predict pattern from chart image.
        
        Args:
            chart_image: numpy array (224, 224, 3) or (N, 224, 224, 3)
        
        Returns:
            {
                'signal': 'bullish'|'bearish'|'neutral',
                'confidence': float,
                'probabilities': {class: prob}
            }
        """
        if self.model is None:
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'probabilities': {c: 0.33 for c in self.CLASSES},
                'error': 'Model not loaded'
            }
        
        # Ensure batch dimension
        if chart_image.ndim == 3:
            chart_image = np.expand_dims(chart_image, axis=0)
        
        # Predict
        predictions = self.model.predict(chart_image, verbose=0)[0]
        
        max_idx = np.argmax(predictions)
        
        return {
            'signal': self.CLASSES[max_idx],
            'confidence': float(predictions[max_idx]),
            'probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.CLASSES, predictions)
            }
        }
    
    def predict_batch(self, images: np.ndarray) -> list:
        """Predict patterns for batch of images."""
        if self.model is None:
            return [{'signal': 'neutral', 'confidence': 0.0} for _ in range(len(images))]
        
        predictions = self.model.predict(images, verbose=0)
        
        results = []
        for pred in predictions:
            max_idx = np.argmax(pred)
            results.append({
                'signal': self.CLASSES[max_idx],
                'confidence': float(pred[max_idx]),
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(self.CLASSES, pred)
                }
            })
        return results
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        save_path: Optional[str] = None
    ):
        """
        Train the CNN model.
        
        Args:
            X_train: Training images (N, 224, 224, 3)
            y_train: Training labels (N, 3) one-hot encoded
            X_val: Validation images
            y_val: Validation labels
            epochs: Training epochs
            batch_size: Batch size
            save_path: Optional path to save best model
        """
        if self.model is None:
            logger.error("No model to train")
            return
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                min_lr=1e-7
            ),
        ]
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    save_path,
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            )
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=3,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zoom_range=0.02,
        )
        
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


# Singleton for reuse
_recognizer: Optional[PatternRecognizerCNN] = None

def get_pattern_recognizer(model_path: Optional[str] = None) -> PatternRecognizerCNN:
    """Get or create pattern recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = PatternRecognizerCNN(model_path=model_path)
    return _recognizer
