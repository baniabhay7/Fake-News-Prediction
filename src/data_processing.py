"""
Text Processing Module for Fake News Detection

This module contains all the essential text processing functions including:
- Tokenization
- Stopword removal
- Vectorization (Count and TF-IDF)
- Complete preprocessing pipeline
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional
from collections import Counter

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Scikit-learn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK datasets"""
    nltk_downloads = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "omw-1.4",
    ]
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except:
            pass


# Initialize NLTK components
def setup_nltk():
    """Download required NLTK datasets and configure paths"""
    # Prefer local nltk_data if it exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_nltk_data = os.path.join(base_dir, ".venv", "nltk_data")
    if os.path.exists(local_nltk_data):
        nltk.data.path.append(local_nltk_data)

    nltk_downloads = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "omw-1.4",
        "punkt_tab",
    ]
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except:
            pass


import os

setup_nltk()


class TextTokenizer:
    """
    Text tokenization utility class
    """

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        """
        Simple tokenization using split()

        Args:
            text (str): Input text

        Returns:
            list: List of tokens
        """
        return text.split()

    @staticmethod
    def nltk_tokenize(text: str, lowercase: bool = True) -> List[str]:
        """
        NLTK word tokenization

        Args:
            text (str): Input text
            lowercase (bool): Convert to lowercase

        Returns:
            list: List of tokens
        """
        if lowercase:
            text = text.lower()
        return word_tokenize(text)

    @staticmethod
    def regex_tokenize(
        text: str, pattern: str = r"\b\w+\b", lowercase: bool = True
    ) -> List[str]:
        """
        Regular expression tokenization

        Args:
            text (str): Input text
            pattern (str): Regex pattern for tokenization
            lowercase (bool): Convert to lowercase

        Returns:
            list: List of tokens
        """
        if lowercase:
            text = text.lower()
        return re.findall(pattern, text)

    @staticmethod
    def sentence_tokenize(text: str) -> List[str]:
        """
        Sentence tokenization

        Args:
            text (str): Input text

        Returns:
            list: List of sentences
        """
        return sent_tokenize(text)


class StopwordRemover:
    """
    Stopword removal utility class
    """

    def __init__(
        self, language: str = "english", custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize stopword remover

        Args:
            language (str): Language for stopwords
            custom_stopwords (list): Additional custom stopwords
        """
        self.stop_words = set(stopwords.words(language))

        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

    def remove_stopwords(
        self, tokens: List[str], filter_alpha: bool = True
    ) -> List[str]:
        """
        Remove stopwords from token list

        Args:
            tokens (list): List of tokens
            filter_alpha (bool): Keep only alphabetic tokens

        Returns:
            list: Filtered tokens
        """
        filtered_tokens = []
        for token in tokens:
            if token.lower() not in self.stop_words:
                if not filter_alpha or token.isalpha():
                    filtered_tokens.append(token)
        return filtered_tokens

    def add_stopwords(self, words: List[str]):
        """Add custom stopwords"""
        self.stop_words.update(words)

    def remove_stopwords_text(self, text: str) -> str:
        """
        Remove stopwords from text and return processed text

        Args:
            text (str): Input text

        Returns:
            str: Text with stopwords removed
        """
        tokens = word_tokenize(text.lower())
        filtered_tokens = self.remove_stopwords(tokens)
        return " ".join(filtered_tokens)


class TextCleaner:
    """
    Text cleaning utility class
    """

    @staticmethod
    def basic_clean(text: str) -> str:
        """
        Basic text cleaning

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def advanced_clean(text: str) -> str:
        """
        Advanced text cleaning with more preprocessing

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text


class TextNormalizer:
    """
    Text normalization using stemming and lemmatization
    """

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem_text(self, text: str) -> str:
        """Apply stemming to text"""
        tokens = word_tokenize(text.lower())
        stemmed_tokens = self.stem_tokens(tokens)
        return " ".join(stemmed_tokens)

    def lemmatize_text(self, text: str) -> str:
        """Apply lemmatization to text"""
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        return " ".join(lemmatized_tokens)


class TextVectorizer:
    """
    Text vectorization utility class
    """

    def __init__(self, vectorizer_type: str = "tfidf", **kwargs):
        """
        Initialize vectorizer

        Args:
            vectorizer_type (str): 'count' or 'tfidf'
            **kwargs: Additional arguments for vectorizer
        """
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None

        # Default parameters
        default_params = {
            "lowercase": True,
            "stop_words": "english",
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 1,  # Changed from 2 to 1 for small datasets
            "max_df": 0.9,  # Changed from 0.8 to 0.9 for small datasets
        }

        # Update with user parameters
        default_params.update(kwargs)

        if vectorizer_type == "count":
            self.vectorizer = CountVectorizer(**default_params)
        elif vectorizer_type == "tfidf":
            # Add TF-IDF specific parameters
            if "sublinear_tf" not in default_params:
                default_params["sublinear_tf"] = True
            self.vectorizer = TfidfVectorizer(**default_params)
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")

    def fit_transform(self, texts: List[str]):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        """Transform texts using fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()

    def get_vocabulary_size(self):
        """Get vocabulary size"""
        return len(self.vectorizer.vocabulary_)


class TextPreprocessor:
    """
    High-level text preprocessing helper for tests and application code.
    Provides simple, well-tested methods: clean_text and preprocess_text.
    """

    def __init__(self):
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer()

    def clean_text(self, text: str) -> str:
        """
        Clean input text by removing URLs, mentions, hashtags and numeric tokens.
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs, emails and mentions
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"@\w+", "", text)

        # Remove hashtags
        text = re.sub(r"#\w+", "", text)

        # Remove digits
        text = re.sub(r"\d+", "", text)

        # Basic cleaning (lowercase & remove special chars)
        text = self.cleaner.basic_clean(text)

        return text

    def preprocess_text(self, text: str) -> str:
        """
        Full preprocessing pipeline used by the application tests.
        Lowercases, cleans, tokenizes, removes stopwords and lemmatizes.
        Returns a single string suitable for vectorizers.
        """
        try:
            cleaned = self.clean_text(text)
            tokens = TextTokenizer.nltk_tokenize(cleaned, lowercase=True)
            remover = StopwordRemover()
            tokens = remover.remove_stopwords(tokens)
            # Lemmatize tokens for normalization
            tokens = self.normalizer.lemmatize_tokens(tokens)
            return " ".join(tokens)
        except Exception:
            return ""

    # Back-compat shim: some parts of the codebase call `preprocess()`
    def preprocess(self, text: str) -> str:
        """
        Backwards-compatible alias for `preprocess_text`.
        """
        return self.preprocess_text(text)


class FeatureExtractor:
    """
    Simple wrapper around TfidfVectorizer used by the training utilities and tests.
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, stop_words="english"
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X

    def transform_texts(self, texts):
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        return self.vectorizer.transform(texts)


class DataLoader:
    """
    Data loading utilities used by tests and training scripts.
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path

    def load_data(self, path: str = None):
        """Load a CSV dataset from path."""
        p = path or self.data_path
        if p and os.path.exists(p):
            return pd.read_csv(p)
        raise FileNotFoundError(f"Data file not found: {p}")

    def create_sample_data(self):
        """Return a small sample dataframe for tests and demos."""
        # Try to use utilities if available
        try:
            from src.utils import create_sample_news_data

            return create_sample_news_data()
        except Exception:
            # Fallback sample data
            data = [
                ("This is a real news example.", 0),
                ("Fake news example with sensational claims!", 1),
            ]
            return pd.DataFrame(data, columns=["text", "label"])


# End of module
