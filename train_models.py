"""Train all four classifiers from data/training_data.csv into models/."""

import os
import subprocess
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import config
from src.data_processing import TextPreprocessor


def ensure_training_data(data_path: str) -> str:
    if os.path.exists(data_path) and len(pd.read_csv(data_path)) >= 100:
        return data_path

    builder = os.path.join(os.path.dirname(__file__), "combine_datasets.py")
    if not os.path.exists(builder):
        sys.exit(f"ERROR: {data_path} missing and {builder} not found.")
    subprocess.run([sys.executable, builder], check=True)
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: combine_datasets.py ran but {data_path} still missing.")
    return data_path


def build_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(alpha=config.NB_ALPHA),
        "Logistic Regression": LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel=config.SVM_KERNEL,
            probability=True,
            random_state=config.RANDOM_STATE,
        ),
    }


def train_all_models(data_path: str = config.TRAINING_DATA_FILE) -> None:
    data_path = ensure_training_data(data_path)
    df = pd.read_csv(data_path)

    preprocessor = TextPreprocessor()
    df["processed_text"] = df["text"].astype(str).apply(preprocessor.preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"].values,
        df["label"].values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"].values,
    )

    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, config.VECTORIZER_FILE)

    results = []
    for name, model in build_models().items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        results.append({
            "Model": name,
            "Train Acc": accuracy_score(y_train, model.predict(X_train_vec)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        })
        filename = name.lower().replace(" ", "_") + "_model.joblib"
        joblib.dump(model, os.path.join(config.MODELS_DIR, filename))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    summary_path = os.path.join(config.MODELS_DIR, "training_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"Samples: {len(df):,} | Train: {len(X_train):,} | Test: {len(X_test):,}\n")
        f.write(
            f"TF-IDF: max_features={config.MAX_FEATURES}, "
            f"ngram_range={config.NGRAM_RANGE}, min_df={config.MIN_DF}, "
            f"max_df={config.MAX_DF}\n\n"
        )
        f.write(results_df.to_string(index=False))
        f.write("\n")


if __name__ == "__main__":
    train_all_models()
