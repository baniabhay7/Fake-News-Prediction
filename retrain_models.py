"""
Retrain all four classifiers from data/training_data.csv and write fresh
.joblib artifacts to models/.

Combines the original train_models.py (precision/recall/F1 reporting and
summary table) with the config-driven structure and combine_datasets.py
fallback of retrain_models.py.

If data/training_data.csv is missing, this script will invoke
combine_datasets.py to build it from the raw ISOT + WELFake sources.

All hyperparameters are read from config.py — edit there to tune.
"""

import os
import sys
import subprocess
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_processing import TextPreprocessor


def ensure_training_data(data_path: str) -> str:
    """Make sure data_path exists. If not, build it via combine_datasets.py."""
    if os.path.exists(data_path):
        df_check = pd.read_csv(data_path)
        if len(df_check) >= 100:
            print(f"Step 1: Using existing dataset with {len(df_check):,} samples")
            return data_path
        print(f"Step 1: Dataset too small ({len(df_check)} samples) — rebuilding...")
    else:
        print("Step 1: training_data.csv not found — building from raw sources...")

    builder = os.path.join(os.path.dirname(__file__), "combine_datasets.py")
    if not os.path.exists(builder):
        print(f"ERROR: {builder} not found and {data_path} is missing.")
        print("Provide data/training_data.csv with columns text,label or restore combine_datasets.py.")
        sys.exit(1)

    subprocess.run([sys.executable, builder], check=True)

    if not os.path.exists(data_path):
        print(f"ERROR: combine_datasets.py ran but {data_path} still missing.")
        sys.exit(1)
    return data_path


def build_models() -> dict:
    """Instantiate the four classifiers from config hyperparameters."""
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


def retrain_all_models(data_path: str = config.TRAINING_DATA_FILE) -> None:
    print("=" * 80)
    print("RETRAINING FAKE NEWS DETECTION MODELS")
    print("=" * 80)

    data_path = ensure_training_data(data_path)

    print("\nStep 2: Loading training data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} samples from {data_path}")
    print(f"  Real news: {(df['label'] == 0).sum():,} | Fake news: {(df['label'] == 1).sum():,}")

    print("\nStep 3: Preprocessing text (clean + tokenize + stopwords + lemmatize)...")
    preprocessor = TextPreprocessor()
    df["processed_text"] = df["text"].astype(str).apply(preprocessor.preprocess)

    X = df["processed_text"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print(f"\nStep 4: Vectorizing (TF-IDF, max_features={config.MAX_FEATURES}, "
          f"ngram_range={config.NGRAM_RANGE})...")
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  Feature matrix shape: {X_train_vec.shape}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, config.VECTORIZER_FILE)
    print(f"  Saved vectorizer: {config.VECTORIZER_FILE}")

    print("\nStep 5: Training models...")
    print("-" * 80)

    results = []
    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        train_acc = accuracy_score(y_train, model.predict(X_train_vec))
        test_acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"  Train accuracy: {train_acc*100:.2f}%")
        print(f"  Test accuracy:  {test_acc*100:.2f}%")
        print(f"  Precision:      {precision*100:.2f}%")
        print(f"  Recall:         {recall*100:.2f}%")
        print(f"  F1-Score:       {f1*100:.2f}%")
        print(classification_report(
            y_test, y_pred,
            target_names=["Real", "Fake"], digits=3,
        ))

        filename = model_name.lower().replace(" ", "_") + "_model.joblib"
        model_path = os.path.join(config.MODELS_DIR, filename)
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")

        results.append({
            "Model": model_name,
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        })

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    summary_path = os.path.join(config.MODELS_DIR, "training_results.txt")
    with open(summary_path, "w") as f:
        f.write("FAKE NEWS DETECTION - MODEL TRAINING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training data: {len(df):,} articles\n")
        f.write(f"Train/Test split: {len(X_train):,} / {len(X_test):,}\n")
        f.write(f"Features: {X_train_vec.shape[1]}\n")
        f.write(f"TF-IDF: max_features={config.MAX_FEATURES}, ngram_range={config.NGRAM_RANGE}, "
                f"min_df={config.MIN_DF}, max_df={config.MAX_DF}\n\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")

    print(f"\nResults saved to {summary_path}")
    print("\nDone. Restart Flask: python app.py")


if __name__ == "__main__":
    retrain_all_models()
