# Fake News Detection — Project Guide

End-to-end walkthrough: what the project does, every dataset explained, how the raw datasets get combined into `training_data.csv`, the exact preprocessing and model parameters, the ensemble decision logic, the optional fact checker, and how to run and retrain everything from scratch.

---

## 1. What the project does

Given a news article (a paragraph of text), classify it as **REAL NEWS** or **FAKE NEWS**.

It does this by running the input through **four independent classifiers** trained on a balanced labeled corpus, then taking a **majority vote**. Optionally, an entity- and pattern-based **fact checker** runs alongside and surfaces advisory warnings (suspicious numerical claims, scam-style phrasing, missing Wikipedia matches). The whole thing is exposed through a Flask web app at `http://localhost:5000`.

Pipeline at a glance:

```
                                                    ┌─►  Naive Bayes        ─┐
raw text  ─►  TextPreprocessor  ─►  TF-IDF vector  ─┼─►  Logistic Regression ─┤
     │                                              ├─►  Random Forest       ─┼─►  Majority vote  ─►  REAL / FAKE
     │                                              └─►  SVM (linear)        ─┘
     │
     └────────────────────────────►  FactChecker (raw text)  ─►  advisory warnings
```

---

## 2. The datasets

There are **three raw source datasets** plus **two derived working files**.

### Raw sources (input data)

| File | Rows | Columns | Origin |
|---|---|---|---|
| `data/News-_dataset/Fake.csv` | 23,481 | `title, text, subject, date` | **ISOT Fake News Dataset** (University of Victoria) — fake political articles, 2016–2017 |
| `data/News-_dataset/True.csv` | 21,417 | `title, text, subject, date` | **ISOT Fake News Dataset** — real Reuters articles, 2016–2017 |
| `data/WELFake_Dataset.csv` | 72,134 | `(index), title, text, label` | **WELFake** dataset (Kaggle, IEEE benchmark) — broader fake news corpus across many topics, includes the COVID-19 era; label `1` = fake, `0` = real |

ISOT supplies the 2016–2017 political content the project's README mentions; WELFake supplies the 2020–2023 health/COVID/modern misinformation content.

### Derived working files

| File | Rows | Purpose |
|---|---|---|
| `data/training_data.csv` | **11,632** (5,816 fake + 5,816 real) | The actual file `train_models.py` reads. Two columns only: `text, label` (1 = fake, 0 = real). Perfectly class-balanced. |
| `data/sample_data.csv` | 15 (8 real + 7 fake) | Tiny hand-picked demo set illustrating the format. **It is not the training set** — it's only a small example file you can inspect or use as a smoke-test fallback. |

### How `training_data.csv` is assembled

Reproducible recipe, implemented in [combine_datasets.py](combine_datasets.py):

1. **Load each raw source** and attach a `label` column where it doesn't already exist:
   - `Fake.csv` → label `1`
   - `True.csv` → label `0`
   - `WELFake_Dataset.csv` → already has a `label` column, keep it
2. **Keep only `text` and `label`**. Drop `title`, `subject`, `date`, `(index)`.
3. **Concatenate** all three into one DataFrame.
4. **Drop rows with empty/NaN text** and de-duplicate.
5. **Class-balance**: take the smaller class count, then sample that many rows from each class so the final file is exactly 50/50.
6. **Downsample to a manageable size** — 11,632 rows total (5,816 per class), small enough to retrain on a laptop in a few minutes.
7. **Shuffle** with a fixed random seed (`42`) and **write** to `data/training_data.csv` with columns `text,label`.

Run it any time to regenerate the file:

```bash
python combine_datasets.py
```

---

## 3. Preprocessing pipeline

Every article (training and prediction) goes through `TextPreprocessor.preprocess()` in [src/data_processing.py](src/data_processing.py). The exact steps:

1. **Lowercase** the text.
2. **Strip URLs**: `http\S+`, `www\S+`, `https\S+`.
3. **Strip emails** (`\S+@\S+`), **mentions** (`@user`), **hashtags** (`#tag`).
4. **Strip digits** and any non-alphabetic character (only `[a-z\s]` survives).
5. **Tokenize** with NLTK's `word_tokenize`.
6. **Remove English stopwords** (NLTK list).
7. **Lemmatize** each token with WordNet's `WordNetLemmatizer`.
8. **Re-join** the tokens into a single space-separated string.

**Important:** digits are gone after preprocessing. The fact checker (which looks at numerical claims like "200% tax") therefore runs on the **raw, untouched text** — see §6.

---

## 4. Vectorization (TF-IDF)

Driven by [config.py](config.py), applied in [train_models.py](train_models.py):

```python
TfidfVectorizer(
    max_features=config.MAX_FEATURES,   # 5000
    ngram_range=config.NGRAM_RANGE,     # (1, 2)
    min_df=config.MIN_DF,               # 1
    max_df=config.MAX_DF,               # 1.0
)
```

| Parameter | Value | Why |
|---|---|---|
| `max_features` | `5000` | Cap vocabulary at the 5,000 most informative terms — keeps models fast and avoids overfitting on rare words. |
| `ngram_range` | `(1, 2)` | Use both unigrams and bigrams. Bigrams capture phrases like *"breaking news"*, *"fake media"*, *"shocking discovery"* that are strong fake-news signals. |
| `min_df` | `1` | Keep terms even if they appear in only one document — needed because the corpus is moderately small. |
| `max_df` | `1.0` | Don't filter very common terms — let TF-IDF weighting handle them. |

Edit any of these in `config.py` and rerun `python train_models.py` — the new values will be used.

---

## 5. The four models and their hyperparameters

All hyperparameters live in [config.py](config.py); the trainer reads them from there.

| Model | sklearn class | Hyperparameters | Why |
|---|---|---|---|
| **Naive Bayes** | `MultinomialNB` | `alpha=0.1` (`config.NB_ALPHA`) | Cheap, strong baseline for bag-of-words. Low alpha = less Laplace smoothing, lets distinctive terms have stronger effect. |
| **Logistic Regression** | `LogisticRegression` | `max_iter=1000` (`config.LR_MAX_ITER`), `random_state=42` | Linear decision boundary on TF-IDF. Robust, calibrated probabilities. |
| **Random Forest** | `RandomForestClassifier` | `n_estimators=100` (`config.RF_N_ESTIMATORS`), `max_depth=20` (`config.RF_MAX_DEPTH`), `random_state=42` | Captures non-linear feature interactions. `max_depth=20` controls overfitting on a sparse 5,000-dim input. |
| **SVM** | `SVC` | `kernel='linear'` (`config.SVM_KERNEL`), `probability=True`, `random_state=42` | Linear-kernel SVM is the gold standard for high-dimensional sparse text. `probability=True` enables `predict_proba` for the ensemble's tie-breaker (slower, but required). |

**Train/test split**: 80% / 20%, `stratify=y`, `random_state=42`. `stratify` keeps the 50/50 class balance in both splits.

After training, every model is serialized with `joblib.dump` into `models/` as:

```
models/
├── vectorizer.joblib
├── naive_bayes_model.joblib
├── logistic_regression_model.joblib
├── random_forest_model.joblib
├── svm_model.joblib
└── training_results.txt
```

The Flask app loads these directly at startup ([app.py](app.py)).

---

## 6. Ensemble decision logic + fact checker

Implemented in [app.py](app.py), in the `/predict` route:

1. **Vectorize** the preprocessed input once.
2. Run `predict()` and `predict_proba()` on **all four models**.
3. Count **fake votes** vs **real votes**.
4. **Majority wins** — 3-1 or 4-0 is decisive.
5. **On a 2-2 tie**, fall back to the higher *average probability* across the four models.
6. **Run the fact checker on the raw (un-preprocessed) text**. This is advisory and never overrides the ensemble verdict — it's returned alongside it so the frontend can flag suspicious numbers, scam phrasing, or unverifiable entities.
7. Return the verdict, the four individual model results, the averaged probabilities, and the fact-check report.

The "safety-first" voting reduces the risk of a single overconfident model being wrong; the fact checker adds a second, lightweight layer of skepticism.

### Fact checker details

The fact checker ([src/fact_checker.py](src/fact_checker.py)) does three things on the raw text:

- **Named-entity extraction** with spaCy (`en_core_web_sm`) → organizations, locations, dates, infrastructure.
- **Wikipedia verification** of the first organization mentioned (existence check only).
- **Pattern checks** on the raw text: unrealistic numerical claims (e.g. `200%`, distances over 500 km, speeds over 350 km/h) and scam-style phrasing (`"forward this"`, `"share urgently"`, `"WhatsApp will charge"`, `"breaking…!"`, `"shocking…!"`, …).

If spaCy / `en_core_web_sm` is not installed, the app prints a warning at startup and skips fact-checking — predictions still work.

---

## 7. Reported performance

Trained on the 11,632-article corpus, evaluated on the 20% held-out split:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **SVM** | **90.4%** | 90% | 90% | 91% |
| **Logistic Regression** | **90.0%** | 90% | 90% | 90% |
| **Random Forest** | **88.0%** | 88% | 88% | 88% |
| **Naive Bayes** | **83.1%** | 84% | 83% | 83% |

The ensemble's effective accuracy is at least as high as the best individual model (typically a touch higher, because errors uncorrelate across model families).

---

## 8. Project structure

```
Fake-News-Prediction/
├── app.py                     # Flask app — loads .joblib files, runs ensemble + fact checker
├── train_models.py          # Trainer — reads training_data.csv, writes .joblib files
├── combine_datasets.py        # Builds training_data.csv from the 3 raw datasets
├── config.py                  # ALL hyperparameters live here — single source of truth
├── requirements.txt           # Python dependencies
├── fix_nltk.py / debug_nltk.py# NLTK setup helpers
│
├── data/
│   ├── News-_dataset/
│   │   ├── Fake.csv           # ISOT fake (raw)
│   │   └── True.csv           # ISOT real (raw)
│   ├── WELFake_Dataset.csv    # WELFake (raw)
│   ├── training_data.csv      # Combined balanced corpus (text, label) — used by retrainer
│   └── sample_data.csv        # 15-row demo file
│
├── models/
│   ├── vectorizer.joblib
│   ├── naive_bayes_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── svm_model.joblib
│   └── training_results.txt
│
├── src/
│   ├── data_processing.py     # TextPreprocessor (used by app + retrainer)
│   ├── model.py               # OO classifier wrapper (NB / LR / RF / SVM) — used by tests
│   ├── fact_checker.py        # spaCy + Wikipedia fact checker (now wired into /predict)
│   └── utils.py
│
├── templates/
│   ├── index_professional.html  # Home page
│   └── about.html               # About page (rendered at /about)
│
└── tests/
    └── test_model.py
```

---

## 9. How to run the project

### Prerequisites

- Python 3.8+
- pip
- ~500 MB free disk (mostly for raw datasets)

### Step 1 — Create a virtualenv and install dependencies

```bash
git clone <this-repo>
cd Fake-News-Prediction

python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Download NLTK data

The `TextPreprocessor` needs tokenizers, stopwords, and the WordNet lemmatizer.

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

If that fails (corporate proxy, etc.), use the helper that downloads into `.venv/nltk_data`:

```bash
python fix_nltk.py
python debug_nltk.py   # smoke test — should print "NLTK is working"
```

### Step 3 — (Recommended) Install spaCy for the fact checker

```bash
pip install spacy wikipedia-api
python -m spacy download en_core_web_sm
```

If you skip this, the app prints a warning and runs without fact checking. Predictions still work.

### Step 4 — Run the app

The repo ships with pre-trained `.joblib` files in `models/`, so you can run immediately:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000), paste a news article, click Predict.

The `/predict` JSON response now includes a `fact_check` field with extracted entities, Wikipedia verification, and any pattern-based warnings — render or ignore it as you like in the frontend.

---

## 10. Retraining from scratch

You only need this if you want to (a) verify reproducibility, (b) try different hyperparameters, or (c) train on a different dataset.

### Step A — Build `training_data.csv` from the raw sources

Make sure the three raw datasets are in place:

```
data/News-_dataset/Fake.csv
data/News-_dataset/True.csv
data/WELFake_Dataset.csv
```

Then run:

```bash
python combine_datasets.py
```

This loads ISOT + WELFake, drops empties, balances classes 50/50, samples down to 11,632 rows, shuffles with `seed=42`, and writes `data/training_data.csv`.

### Step B — Retrain the four models

```bash
python train_models.py
```

If `data/training_data.csv` doesn't exist, this script automatically calls `combine_datasets.py` first.

It will:

1. Load `data/training_data.csv` (11,632 rows).
2. Preprocess every article (TextPreprocessor).
3. 80/20 stratified split (`random_state=42`).
4. Fit `TfidfVectorizer` with the parameters from `config.py`.
5. Train all four models with the hyperparameters from `config.py`.
6. Print per-model train/test accuracy and a classification report.
7. Save fresh `.joblib` artifacts into `models/` — overwriting whatever was there.
8. Write a summary to `models/training_results.txt`.

Expect ~3–10 minutes total on a laptop, dominated by SVM training.

### Step C — Restart the Flask app

```bash
python app.py
```

The new models are now live.

### Tuning hyperparameters

Edit `config.py` and rerun the trainer. The values that actually drive training:

```python
MAX_FEATURES = 5000          # TF-IDF vocabulary cap
NGRAM_RANGE  = (1, 2)        # unigrams + bigrams
MIN_DF       = 1             # min document frequency
MAX_DF       = 1.0           # max document frequency

NB_ALPHA          = 0.1      # Naive Bayes Laplace smoothing
LR_MAX_ITER       = 1000     # Logistic Regression max iterations
RF_N_ESTIMATORS   = 100      # Random Forest tree count
RF_MAX_DEPTH      = 20       # Random Forest tree depth
SVM_KERNEL        = "linear" # SVM kernel

TEST_SIZE    = 0.2           # train/test split
RANDOM_STATE = 42            # reproducibility seed
```

---

## 11. Tests

```bash
python -m unittest tests.test_model
# or
pytest tests/
```

Tests cover the OO `FakeNewsClassifier` / `ModelComparison` path in `src/model.py`, which now supports all four models including SVM (so `ModelComparison.train_all_models` no longer raises). The Flask runtime uses the flat path in `app.py` + `train_models.py`, so end-to-end verification of the live system is best done by running the app and predicting on the bundled `SAMPLE_NEWS` examples.

---

## 12. Tech stack

- **Backend**: Python 3.8+, Flask
- **ML**: scikit-learn (`MultinomialNB`, `LogisticRegression`, `RandomForestClassifier`, `SVC`)
- **NLP**: NLTK (tokenization, stopwords, WordNet lemmatization)
- **Vectorization**: scikit-learn `TfidfVectorizer` (unigrams + bigrams, top 5,000 features)
- **Fact checking**: spaCy (`en_core_web_sm`) + `wikipedia-api`
- **Frontend**: vanilla JavaScript + custom CSS (glassmorphism), Chart.js
- **Persistence**: joblib

---

## 13. Data credits

- **ISOT Fake News Dataset** — Information Security and Object Technology research lab, University of Victoria.
  https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
- **WELFake Dataset** — Verma, P. K. et al., IEEE Transactions on Computational Social Systems, 2021. Available on Kaggle:
  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Both datasets are publicly released for academic / research use. This project uses them only for model training; no article content is redistributed beyond the trained model weights.
