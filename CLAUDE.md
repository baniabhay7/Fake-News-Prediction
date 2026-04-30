# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run the Flask app (loads pre-trained models from `models/` at startup, serves on port 5000):
```bash
python app.py
```

Retrain all four models from `data/training_data.csv` and write fresh `.joblib` artifacts into `models/`:
```bash
python train_models.py
```

Run the unit tests:
```bash
python -m unittest tests.test_model      # or: pytest tests/
python -m unittest tests.test_model.TestTextPreprocessor.test_clean_text   # single test
```

First-time NLTK setup (the package imports auto-download but these helpers exist for when that fails):
```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','stopwords','wordnet','omw-1.4','punkt_tab']]"
python fix_nltk.py     # downloads into .venv/nltk_data (preferred path, see data_processing.setup_nltk)
python debug_nltk.py   # smoke-test NLTK is wired up
```

The fact checker requires a spaCy model that is **not** in `requirements.txt`:
```bash
python -m spacy download en_core_web_sm
```
If absent, `app.py` catches `ImportError`/`OSError` and runs without fact checking.

## Architecture

### Two parallel ML pipelines — know which one is live

There are two separate training/prediction code paths in this repo, and they don't share code at runtime:

1. **The runtime path (what the Flask app actually uses):** `app.py` → `joblib.load()` directly on the four files in `models/` → `src.data_processing.TextPreprocessor.preprocess()` for text cleaning. The trainer for these artifacts is **`train_models.py`** (a flat script using sklearn directly with a `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`).
2. **The OO path (referenced by tests, not by the app):** `src/model.py` defines `FakeNewsClassifier` and `ModelComparison`, plus `src/data_processing.py` defines `FeatureExtractor` and `DataLoader`. These exist for `tests/test_model.py` and a (missing) `download_dataset.py` flow, but the Flask app never instantiates them.

When changing the live prediction behavior, edit `app.py` and `train_models.py` together. `src/model.py` is **not** wired in — changes there will not affect predictions until you also rewrite `app.py`'s loader.

### Ensemble voting in `app.py`

`/predict` runs the input through all four models loaded at startup (Naive Bayes, Logistic Regression, Random Forest, SVM) and returns a **majority-vote** verdict. On a 2–2 tie it falls back to the higher average probability. Per-model probabilities and votes are returned in the JSON for the frontend to render. `AVAILABLE_MODELS` (the dict in `app.py`, not `config.MODEL_NAMES`) is the source of truth for which `.joblib` files are expected — keep them in sync if you add/remove a model.

### TextPreprocessor.preprocess is the canonical entry point

`app.py` and `train_models.py` both call `TextPreprocessor().preprocess(text)`. That method is a back-compat alias for `preprocess_text` — do not remove it. The pipeline is: clean (URL/email/mention/hashtag/digit strip + lowercase + non-alpha removal) → NLTK word tokenize → stopword removal → WordNet lemmatize → join. The `clean_text` step strips digits, so any feature that needs numbers (e.g. fact checker numerical claim detection) must run **before** preprocessing or directly on the raw input.

### Fact checker is optional and isolated

`src/fact_checker.py` (spaCy NER + Wikipedia API + regex scam patterns) is imported behind a try/except in `app.py` and assigned to a `fact_checker` global that is **currently never invoked from the `/predict` route** despite being loaded. If you wire it in, do it on the raw text (digits are still present) and treat its result as advisory — the route should still return a verdict if `FACT_CHECKER_AVAILABLE` is False.

### Known mismatches to watch for

- `app.py:208` renders `about_rebuilt.html`, but `templates/` only contains `about.html` and `index_professional.html`. The `/about` route 500s as-is.
- `data/training_data.csv` is gitignored and not committed; `train_models.py` will fail or fall back unless you supply it (or a missing `download_dataset.py`).
- `src/model.py:_create_model` only registers `naive_bayes`, `random_forest`, `logistic_regression` — no SVM — so `ModelComparison.train_all_models` (which iterates `config.MODEL_NAMES`, including `"svm"`) will raise. The live path doesn't hit this; tests don't exercise it.
- `config.py` constants (`MAX_FEATURES=10000`, `MIN_DF=2`, `MAX_DF=0.8`) are **not** what `train_models.py` actually uses (it hardcodes `max_features=5000` with default `min_df`/`max_df`). Changing `config.py` will not change retraining behavior.

### Models directory

`models/*.joblib` artifacts are committed to the repo (`.gitignore` has the `models/*.joblib` line commented out intentionally). The expected filenames are exactly: `vectorizer.joblib`, `naive_bayes_model.joblib`, `logistic_regression_model.joblib`, `random_forest_model.joblib`, `svm_model.joblib`. `train_models.py` derives these from `model_name.lower().replace(' ', '_') + '_model.joblib'` — renaming a model in the dict will silently produce a file the app can't find.
