"""
Utility functions for the fake news detection app.
Provides helper functions for formatting results and creating sample data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

def format_prediction_result(prediction: int, confidence: float) -> Dict[str, any]:
    """Format prediction result for display."""
    label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    color = "red" if prediction == 1 else "green"
    icon = "🚨" if prediction == 1 else "✅"
    
    return {
        'label': label,
        'prediction': prediction,
        'confidence_percentage': f"{confidence*100:.1f}%",
        'color': color,
        'icon': icon
    }

def create_sample_news_data() -> pd.DataFrame:
    """Create extended sample news data for demonstration."""
    sample_data = [
        ("The World Health Organization announced new guidelines for global health.", 0),
        ("Researchers at Stanford University published a study on climate change.", 0),
        ("BREAKING: Drinking coffee backwards cures all diseases! Doctors are FURIOUS!", 1),
        ("SHOCKING: Ancient aliens built the pyramids using weird technology!", 1),
    ]
    return pd.DataFrame(sample_data, columns=['text', 'label'])

def get_model_info() -> Dict[str, str]:
    """Get information about available models."""
    return {
        'naive_bayes': 'Fast, good for basic text analysis.',
        'random_forest': 'Robust, handles complex patterns.',
        'logistic_regression': 'Classic and very interpretable.',
        'svm': 'Great for finding tricky patterns in text.'
    }
