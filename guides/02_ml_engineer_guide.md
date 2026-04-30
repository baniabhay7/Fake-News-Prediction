# Person 2 — ML Engineer Guide

**Your role**: own the machine learning core. You explain *how raw text becomes numbers, which models we trained, what hyperparameters they use, and how well they perform*. You're the most technical part of the presentation.

**Files you own**:
- [retrain_models.py](../retrain_models.py) — the training script
- [config.py](../config.py) — single source of truth for every hyperparameter
- [src/model.py](../src/model.py) — OO classifier wrapper used by tests
- All `.joblib` files in `models/` — your trained artifacts
- [models/training_results.txt](../models/training_results.txt) — performance log

**What Person 1 hands you**: `data/training_data.csv` with 11,632 balanced rows.
**What you hand off to Person 3**: 4 trained models + 1 vectorizer, all serialized as `.joblib` files in `models/`.

---

## 1. Context (1 paragraph)

Person 1 produced a balanced corpus of 11,632 labeled articles. Your job is to turn that text into something a machine learning model can ingest, train **four different classifiers** on the resulting features, evaluate them on a held-out test set, and save the trained models so the Flask app can load them. After your stage, the project has working `.joblib` files that Person 3 can serve from the API.

---

## 2. Preprocessing pipeline (before vectorization)

Even before TF-IDF, every article passes through `TextPreprocessor.preprocess()` in `src/data_processing.py`. The exact sequence:

1. **Lowercase** the text.
2. **Strip URLs** matching `http\S+`, `www\S+`, `https\S+`.
3. **Strip emails** (`\S+@\S+`), **mentions** (`@user`), **hashtags** (`#tag`).
4. **Strip digits** and any non-alphabetic character (only `[a-z\s]` survives).
5. **Tokenize** with NLTK's `word_tokenize`.
6. **Remove English stopwords** (NLTK's English stopword list — *the, a, is, …*).
7. **Lemmatize** each token with WordNet's `WordNetLemmatizer` (running → run, better → good).
8. **Re-join** tokens into a single space-separated string.

**Why this matters**: by the time the model sees the text, it's normalized — same words map to the same tokens regardless of casing or grammatical form, irrelevant noise is gone, and the vocabulary is much smaller and more informative.

**Trade-off**: digits are stripped. So anything that needs numbers (Person 3's fact checker looking for "200% tax" or "1000 km/h") has to run on the **raw** text, before this preprocessing — Person 3 will explain that.

---

## 3. Feature extraction — TF-IDF

After preprocessing, every article is still a string of words. To feed it to a classifier, we convert it to a fixed-length numerical vector using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

### How TF-IDF works (one minute)

For each (term, document) pair:

```
TF-IDF(term, doc) = TF(term, doc) × log(N / DF(term))
```

- `TF` (term frequency): how often the term appears in the document.
- `DF` (document frequency): how many documents contain the term.
- `N`: total documents.

A word that appears in *every* document (e.g. *"news"* in a fake-news corpus) gets weight near zero — it's everywhere, so it's not informative. A word that appears in only a few documents but appears strongly there gets a high weight — it's a signal.

### Our parameters

Set in `config.py`, applied in `retrain_models.py`:

```python
TfidfVectorizer(
    max_features=5000,        # config.MAX_FEATURES
    ngram_range=(1, 2),       # config.NGRAM_RANGE
    min_df=1,                 # config.MIN_DF
    max_df=1.0,               # config.MAX_DF
)
```

| Parameter | Value | Why |
|---|---|---|
| `max_features` | 5000 | Cap vocabulary at the 5,000 most informative terms. Larger → slower, more overfitting risk. |
| `ngram_range` | (1, 2) | Use both single words AND two-word phrases. Bigrams catch *"breaking news"*, *"fake media"*, *"shocking discovery"* — strong fake-news signals that single words miss. |
| `min_df` | 1 | Keep terms that appear in even one document — corpus is moderately small. |
| `max_df` | 1.0 | Don't filter common terms — let TF-IDF weighting handle them. |

**Output**: each article becomes a sparse 5,000-dimensional vector. Most entries are zero (the article only uses a small subset of the vocabulary), but the non-zero entries are weighted by importance.

---

## 4. The four classifiers — chosen on purpose

We train four models from **four different model families**. This isn't decorative — it ensures the ensemble's errors are uncorrelated, so the majority vote is more reliable than any single model.

| Model | Family | Why we picked it | Hyperparameters |
|---|---|---|---|
| **Naive Bayes** | Probabilistic | Cheap, strong baseline for bag-of-words text. Trains in seconds. | `alpha=0.1` — Laplace smoothing. Lower α lets distinctive terms drive predictions. |
| **Logistic Regression** | Linear | Linear decision boundary; well-calibrated probabilities (good for the ensemble's tie-breaker). | `max_iter=1000`, `random_state=42` |
| **Random Forest** | Tree-based ensemble | Captures non-linear feature interactions other models miss. | `n_estimators=100` (100 trees), `max_depth=20`, `random_state=42` |
| **SVM (linear kernel)** | Margin-based | The classical gold standard for high-dimensional sparse text. Best accuracy in our test. | `kernel='linear'`, `probability=True` (slower, but needed for `predict_proba`), `random_state=42` |

All hyperparameters live in [config.py](../config.py). To tune:

```python
# Edit these in config.py and re-run python retrain_models.py
NB_ALPHA          = 0.1
LR_MAX_ITER       = 1000
RF_N_ESTIMATORS   = 100
RF_MAX_DEPTH      = 20
SVM_KERNEL        = "linear"
```

---

## 5. Train/test split

```python
train_test_split(
    X, y,
    test_size=0.2,            # config.TEST_SIZE — 20% held out for testing
    random_state=42,          # config.RANDOM_STATE — reproducible split
    stratify=y,               # preserve 50/50 class balance in both splits
)
```

- **80% train** (≈9,300 rows) — used to fit each model.
- **20% test** (≈2,330 rows) — held out, never seen during training.
- **`stratify=y`** keeps each split exactly 50/50, so test accuracy is comparable to train accuracy.
- **`random_state=42`** means anyone re-running the script gets the same split — reproducibility.

---

## 6. Training results

From `models/training_results.txt`, on the held-out 20% test set:

| Model | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **SVM** | **90.4%** | 90% | 90% | 91% |
| **Logistic Regression** | **90.0%** | 90% | 90% | 90% |
| **Random Forest** | **88.0%** | 88% | 88% | 88% |
| **Naive Bayes** | **83.1%** | 84% | 83% | 83% |

### Reading the numbers

- **SVM and LogReg are essentially tied at 90%** — both are strong linear models on TF-IDF. SVM edges out marginally.
- **Random Forest is solid at 88%** — the slight gap to linear models is typical for sparse high-dim features (trees prefer dense data).
- **Naive Bayes is the weakest at 83%** — it's also the fastest and most interpretable; the gap is the price of its simplicity.

**Ensemble effect**: when you majority-vote four models with different error profiles, the ensemble's effective accuracy is at least as high as the best single model — typically a touch higher because uncorrelated errors cancel out.

---

## 7. Saving the artifacts

After training, every model and the vectorizer are serialized with `joblib.dump`:

```
models/
├── vectorizer.joblib                       # the fitted TfidfVectorizer
├── naive_bayes_model.joblib                # MultinomialNB
├── logistic_regression_model.joblib        # LogisticRegression
├── random_forest_model.joblib              # RandomForestClassifier
├── svm_model.joblib                        # SVC
└── training_results.txt                    # human-readable performance log
```

Person 3's Flask app loads these directly at startup with `joblib.load()`. **No retraining at request time** — predictions hit pre-loaded models in memory.

---

## 8. Tunability — `config.py` is the single source of truth

Everything that affects training behavior is in [config.py](../config.py). Edit there → rerun `python retrain_models.py` → new models in `models/`. Mention this in the presentation: it's a real engineering win, not just paperwork. The original codebase had hardcoded values that diverged from `config.py`; we cleaned it up so tuning is one file away.

---

## 9. What you should know cold

- **"Why these four models specifically?"**
  Four different model families (probabilistic, linear, tree-based, margin-based) so their errors are uncorrelated. That's what makes the ensemble work — voting four identical models would gain nothing.

- **"Why not deep learning / BERT / transformers?"**
  Honest answer: classical models hit 90%+ on this corpus with no GPU and 5 minutes of training. Transformers might add 2-3 percentage points but require massive compute, longer training, and a more complex deployment story. The ROI isn't there for the corpus size.

- **"How do you prevent overfitting?"**
  `max_features=5000` caps vocabulary; Random Forest's `max_depth=20` prevents trees from memorizing; SVM with linear kernel has low capacity; train/test split is held out and stratified.

- **"Why does the linear SVM beat Random Forest?"**
  TF-IDF features are very high-dimensional (5,000) and very sparse. Linear models excel here; tree-based models prefer dense, lower-dimensional features.

- **"Why is `probability=True` on the SVM slow?"**
  Default SVM doesn't output probabilities. To enable `predict_proba`, sklearn fits Platt-scaled logistic regression on top, which adds a 5-fold cross-validation step. We pay this cost because the ensemble needs probabilities for tie-breaking.

- **"What's the training time?"**
  3–10 minutes total on a laptop, dominated by SVM's CV step. Naive Bayes trains in <1 second.

- **"Could we add more models?"**
  Yes — gradient boosting (XGBoost, LightGBM) would slot in cleanly. Add to the dict in `retrain_models.py` and the `AVAILABLE_MODELS` map in `app.py`.

---

## 10. Quick reference — what to put on slides

You get **4 slides** (~4-5 minutes — you have the most technical content):

1. **Preprocessing pipeline** — the 8-step text cleaning sequence.
2. **TF-IDF feature extraction** — what it is and why bigrams matter.
3. **The four classifiers** — table with model, family, why-we-picked-it.
4. **Performance results** — accuracy table + ensemble logic teaser.

Detailed slide content is in [`slides/02_ml_engineer_slides.md`](../slides/02_ml_engineer_slides.md).

---

## 11. Hand-off to Person 3

After your section, say:

> "We've got four trained models, each with its own perspective. The next question is: how do we combine them at prediction time, and how do we serve this to a user? Over to [Person 3's name]."
