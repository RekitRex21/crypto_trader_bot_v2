"""
Chart image generator for converting OHLCV data to candlestick chart images.
Used for CNN pattern recognition.
"""
import logging
from typing import Tuple, Optional
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image

logger = logging.getLogger(__name__)


class ChartImageGenerator:
    """
    Converts OHLCV dataframes to normalized candlestick chart images.
    Output format is suitable for CNN input (224x224x3 normalized).
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.style = self._create_style()
    
    def _create_style(self):
        """Create a clean chart style optimized for pattern recognition."""
        return mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up='#00C805',      # Green for up candles
                down='#FF3B30',    # Red for down candles
                edge='inherit',
                wick='inherit',
                volume='inherit',
            ),
            rc={
                'font.size': 8,
                'axes.labelsize': 8,
                'xtick.labelsize': 6,
                'ytick.labelsize': 6,
            }
        )
    
    def ohlcv_to_image(
        self, 
        df: pd.DataFrame, 
        lookback: int = 100, 
        include_indicators: bool = True,
        include_volume: bool = True
    ) -> np.ndarray:
        """
        Convert OHLCV dataframe to candlestick chart image.
        
        Args:
            df: DataFrame with OHLCV columns and datetime index
            lookback: Number of candles to display
            include_indicators: Whether to add moving averages
            include_volume: Whether to include volume bars
        
        Returns:
            numpy array (224, 224, 3) normalized [0, 1]
        """
        # Take last N candles
        chart_data = df.iloc[-lookback:].copy()
        
        # Ensure proper column names for mplfinance
        chart_data.columns = [c.lower() for c in chart_data.columns]
        
        # Prepare additional plots (moving averages)
        addplot = []
        if include_indicators and len(chart_data) >= 50:
            try:
                sma20 = chart_data['close'].rolling(20).mean()
                sma50 = chart_data['close'].rolling(50).mean()
                addplot = [
                    mpf.make_addplot(sma20, color='#2196F3', width=1),
                    mpf.make_addplot(sma50, color='#FF9800', width=1),
                ]
            except Exception as e:
                logger.debug(f"Could not add indicators: {e}")
        
        try:
            # Generate chart
            fig, axes = mpf.plot(
                chart_data,
                type='candle',
                style=self.style,
                volume=include_volume,
                addplot=addplot if addplot else None,
                returnfig=True,
                figsize=(8, 6),
                tight_layout=True,
                datetime_format='%m-%d',
                xrotation=0,
            )
            
            # Convert figure to numpy array
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            # Load and resize image
            img = Image.open(buf).convert('RGB')
            img = img.resize(self.image_size, Image.LANCZOS)
            
            # Normalize to [0, 1]
            img_array = np.array(img) / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to generate chart image: {e}")
            # Return blank image on failure
            return np.zeros((*self.image_size, 3))
    
    def ohlcv_to_bytes(
        self, 
        df: pd.DataFrame, 
        lookback: int = 100,
        include_indicators: bool = True
    ) -> bytes:
        """
        Generate chart image and return as PNG bytes.
        Useful for Telegram bot sending.
        
        Args:
            df: OHLCV dataframe
            lookback: Number of candles
            include_indicators: Add moving averages
        
        Returns:
            PNG image bytes
        """
        chart_data = df.iloc[-lookback:].copy()
        chart_data.columns = [c.lower() for c in chart_data.columns]
        
        addplot = []
        if include_indicators and len(chart_data) >= 50:
            try:
                sma20 = chart_data['close'].rolling(20).mean()
                sma50 = chart_data['close'].rolling(50).mean()
                addplot = [
                    mpf.make_addplot(sma20, color='#2196F3', width=1.5),
                    mpf.make_addplot(sma50, color='#FF9800', width=1.5),
                ]
            except Exception:
                pass
        
        try:
            fig, axes = mpf.plot(
                chart_data,
                type='candle',
                style=self.style,
                volume=True,
                addplot=addplot if addplot else None,
                returnfig=True,
                figsize=(12, 8),
                tight_layout=True,
            )
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate chart bytes: {e}")
            return b''
    
    def generate_batch(
        self, 
        df: pd.DataFrame, 
        window_size: int = 100, 
        stride: int = 24
    ) -> np.ndarray:
        """
        Generate multiple chart images using sliding window.
        Useful for training dataset generation.
        
        Args:
            df: Full OHLCV dataframe
            window_size: Candles per chart
            stride: Step size between windows
        
        Returns:
            numpy array of shape (N, 224, 224, 3)
        """
        images = []
        
        for i in range(window_size, len(df), stride):
            window_df = df.iloc[i - window_size:i]
            img = self.ohlcv_to_image(window_df, lookback=window_size)
            images.append(img)
        
        return np.array(images)


# Convenience function
def generate_chart_image(df: pd.DataFrame, lookback: int = 100) -> np.ndarray:
    """Generate single chart image."""
    generator = ChartImageGenerator()
    return generator.ohlcv_to_image(df, lookback)
