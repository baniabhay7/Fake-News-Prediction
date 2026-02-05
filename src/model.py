"""
Machine learning models for fake news detection.
This module implements various ML algorithms and provides training/prediction functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import DataLoader, FeatureExtractor
import config


class FakeNewsClassifier:
    """Main classifier for fake news detection."""
    
    def __init__(self, model_name: str = "naive_bayes"):
        """
        Initialize classifier with specified model.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.model = self._create_model(model_name)
        self.feature_extractor = None
        self.is_trained = False
    
    def _create_model(self, model_name: str) -> Any:
        """
        Create ML model based on name.
        
        Args:
            model_name (str): Model name
            
        Returns:
            Sklearn model object
        """
        models = {
            "naive_bayes": MultinomialNB(alpha=1.0),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                max_depth=10,
                min_samples_split=5
            ),
            "logistic_regression": LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000,
                C=1.0
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
        
        return models[model_name]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_extractor: FeatureExtractor) -> None:
        """
        Train the model.
        """
        print(f"Training {self.model_name} model...")
        
        self.model.fit(X_train, y_train)
        self.feature_extractor = feature_extractor
        self.is_trained = True
        
        print(f"Model training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def predict_text(self, text: str) -> Tuple[int, float]:
        """
        Predict single text article.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Transform text to features
        X = self.feature_extractor.transform_texts([text])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def save_model(self, model_path: str, vectorizer_path: str) -> None:
        """
        Save trained model and vectorizer.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_extractor.vectorizer, vectorizer_path)
    
    def load_model(self, model_path: str, vectorizer_path: str) -> None:
        """
        Load trained model and vectorizer.
        """
        try:
            self.model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            from src.data_processing import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.vectorizer = vectorizer
            self.feature_extractor.is_fitted = True
            
            self.is_trained = True
        except Exception as e:
            print(f"Error loading model: {e}")


class ModelComparison:
    """Compare multiple models and select the best one."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_extractor: FeatureExtractor) -> None:
        """
        Train all available models and evaluate them.
        """
        for model_name in config.MODEL_NAMES:
            classifier = FakeNewsClassifier(model_name)
            classifier.train(X_train, y_train, feature_extractor)
            metrics = classifier.evaluate(X_test, y_test)
            self.models[model_name] = classifier
            self.results[model_name] = metrics
    
    def get_best_model(self) -> Tuple[str, FakeNewsClassifier]:
        """
        Get the best performing model based on F1-score.
        """
        if not self.results:
            raise ValueError("No models trained.")
        
        best_model_name = max(self.results.keys(),
                             key=lambda x: self.results[x]['f1_score'])
        
        return best_model_name, self.models[best_model_name]
