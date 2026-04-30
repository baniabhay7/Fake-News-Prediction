# Configuration settings for Fake News Detection App
# Editing these values DOES change behavior — train_models.py reads from here.
import os

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"

# Model registry
MODEL_NAMES = ["naive_bayes", "logistic_regression", "random_forest", "svm"]
DEFAULT_MODEL = "svm"

# TF-IDF vectorizer settings (used by train_models.py)
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 1
MAX_DF = 1.0

# Train/test split
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Per-model hyperparameters (used by train_models.py and src/model.py)
NB_ALPHA = 0.1
LR_MAX_ITER = 1000
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
SVM_KERNEL = "linear"

# App settings
APP_TITLE = "Fake News Detection"
APP_ICON = "🔍"

# File paths
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.csv")
SAMPLE_DATA_FILE = os.path.join(DATA_DIR, "sample_data.csv")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "vectorizer.joblib")
