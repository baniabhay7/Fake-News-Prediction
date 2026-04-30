# Presentation Roles — 4-Person Split

Project: **Fake News Detection (ML Ensemble + Flask)**

The project is split into four logically separate areas. Each person owns ~2–3 files, ~3–5 minutes of speaking time, and one slide section. The flow below is the recommended presentation order — every person hands off cleanly to the next.

| # | Role | Owner | Files owned | Slide section | Time |
|---|---|---|---|---|---|
| 1 | **Data Engineer** | _(name)_ | `data/`, `combine_datasets.py` | Datasets & data pipeline | 3–4 min |
| 2 | **ML Engineer** | _(name)_ | `train_models.py`, `config.py`, `src/model.py` | Models & training | 4–5 min |
| 3 | **Backend Engineer** | _(name)_ | `app.py`, `src/data_processing.py`, `src/fact_checker.py` | Application & ensemble logic | 3–4 min |
| 4 | **Frontend / Demo / Docs** | _(name)_ | `templates/`, `tests/`, `PROJECT_GUIDE.md`, live demo | UI walkthrough + live demo + Q&A lead | 4–5 min |

Total presentation time: **15–18 minutes**, leaving ~5 min for Q&A.

---

## Person 1 — Data Engineer

### Owns

- `data/News-_dataset/Fake.csv` (ISOT, 23,481 rows)
- `data/News-_dataset/True.csv` (ISOT, 21,417 rows)
- `data/WELFake_Dataset.csv` (WELFake, 72,134 rows)
- `data/training_data.csv` (combined, 11,632 rows, balanced)
- `data/sample_data.csv` (15-row demo set)
- [combine_datasets.py](combine_datasets.py)

### What to study

- §2 of [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — the dataset section
- The combine script, line by line
- How the three sources differ (columns, label conventions, time period covered)

### Slide section: "Where the data comes from"

1. **Slide A — The two source datasets**
   - **ISOT Fake News Dataset** (University of Victoria) — political articles, 2016–2017. Two files: `Fake.csv` (label = 1) and `True.csv` (real Reuters articles, label = 0).
   - **WELFake Dataset** (IEEE benchmark, Kaggle mirror) — 72k articles spanning broader topics including the COVID-19 era. Already labeled.
2. **Slide B — Combining + balancing**
   - Show the funnel: 117k raw rows → drop NaN/empties → de-duplicate → class-balance → sample down to 11,632 → shuffle with seed 42 → `training_data.csv`.
   - 5,816 fake + 5,816 real → perfectly 50/50 → no class-imbalance bias.
3. **Slide C — `training_data.csv` vs `sample_data.csv`**
   - `training_data.csv` is what the trainer reads.
   - `sample_data.csv` is a 15-row hand-picked demo file, NOT the training set.

### Speaking points (3–4 min)

> "We use two publicly released datasets — ISOT from the University of Victoria and WELFake from IEEE — together giving us articles from 2016 right through the COVID-19 era. The raw datasets total about 117 thousand rows, but we don't use all of them. We balance the classes 50/50 and sample down to 11,632 articles — small enough to retrain on a laptop in under 10 minutes, large enough for stable accuracy. Everything is reproducible: one command, `python combine_datasets.py`, rebuilds the file from the raw sources with a fixed random seed."

### Q&A this person should handle

- "Why didn't you use all 117k rows?" → 5,816 per class is the lower bound after balancing; downsampling speeds up SVM training dramatically.
- "Why two datasets and not one?" → ISOT alone is 2016–17 only; WELFake adds COVID-era misinformation.
- "What about data leakage between train and test?" → de-duplication removes exact duplicates; the 80/20 split is stratified and uses `random_state=42`.

---

## Person 2 — ML Engineer

### Owns

- [train_models.py](train_models.py)
- [config.py](config.py)
- [src/model.py](src/model.py) — the OO classifier (used by tests)
- All `.joblib` artifacts in `models/`
- [models/training_results.txt](models/training_results.txt)

### What to study

- §4 (TF-IDF), §5 (the four models), §7 (performance) of `PROJECT_GUIDE.md`
- The training loop in `train_models.py`
- Why each model was chosen and what its hyperparameters mean

### Slide section: "How the models are trained"

1. **Slide A — Feature extraction (TF-IDF)**
   - Why TF-IDF and not raw counts: down-weights very common words.
   - `max_features=5000`, `ngram_range=(1,2)` — unigrams + bigrams.
   - Bigrams catch phrases like *"breaking news"*, *"shocking discovery"* — strong fake signals.
2. **Slide B — The four classifiers**
   - **Naive Bayes** (`MultinomialNB`, alpha=0.1) — fast bag-of-words baseline.
   - **Logistic Regression** (max_iter=1000) — linear, well-calibrated probabilities.
   - **Random Forest** (100 trees, max_depth=20) — non-linear feature interactions.
   - **SVM** (linear kernel, probability=True) — gold standard for sparse text.
   - Diagram: each takes the same TF-IDF vector → independent vote.
3. **Slide C — Train/test split + results table**
   - 80/20 stratified split, `random_state=42`.
   - Performance: SVM 90.4%, LR 90.0%, RF 88.0%, NB 83.1%.
   - Highlight: ensemble accuracy ≥ best single model because errors uncorrelate.

### Speaking points (4–5 min)

> "Once the data is preprocessed, we convert each article into a numerical fingerprint using TF-IDF. We cap the vocabulary at the 5,000 most informative terms and include both single words and two-word phrases — those bigrams are crucial because phrases like 'breaking news' or 'shocking discovery' are themselves fake-news signals. We then train four very different classifiers — Naive Bayes for speed, Logistic Regression for calibrated probabilities, Random Forest for non-linear interactions, and a linear SVM for high-dimensional text. SVM gets us 90.4% accuracy on the held-out test set, but rather than just pick the best model, we keep all four — and we'll see why in the next section."

### Q&A this person should handle

- "Why these four models specifically?" → covers four different model families (probabilistic, linear, tree-based, margin-based) so their errors are uncorrelated.
- "Why not deep learning / transformers?" → these classical models hit 90%+ accuracy with no GPU and a few minutes of training; for the corpus size, the ROI on transformers is small.
- "How do you avoid overfitting?" → max_features=5000, RF max_depth=20, stratified holdout, fixed seed.

---

## Person 3 — Backend Engineer

### Owns

- [app.py](app.py) — Flask app
- [src/data_processing.py](src/data_processing.py) — `TextPreprocessor`
- [src/fact_checker.py](src/fact_checker.py) — spaCy + Wikipedia fact checker
- The `/predict`, `/`, `/about`, `/api/health` routes

### What to study

- §3 (preprocessing pipeline), §6 (ensemble logic + fact checker) of `PROJECT_GUIDE.md`
- The `/predict` route in `app.py` end-to-end
- How `FactChecker.analyze()` works on raw text

### Slide section: "How a request becomes a verdict"

1. **Slide A — Preprocessing pipeline**
   - 8 steps: lowercase → strip URLs → strip emails/mentions/hashtags → strip digits → tokenize → stopwords → lemmatize → rejoin.
   - Crucially: **digits are stripped**, so anything that needs numbers (the fact checker's "200% tax" detector) must run on the **raw** text.
2. **Slide B — Ensemble decision logic (majority vote)**
   - All four models predict in parallel.
   - Count fake votes vs real votes → majority wins.
   - On a 2–2 tie → fall back to the higher *average* probability.
   - Why majority vote: one overconfident wrong model can't dominate.
3. **Slide C — The fact checker (advisory layer)**
   - Three checks on the raw text: spaCy NER → Wikipedia verification of organizations; regex check for unrealistic numbers (e.g. 200%, 1000 km/h); regex check for scam phrases ("forward this", "WhatsApp will charge").
   - It **never overrides** the ensemble — it's surfaced alongside as warnings.

### Speaking points (3–4 min)

> "When a user pastes an article, the Flask app does two things in parallel. First, it preprocesses the text — lowercase, strips URLs, removes stopwords, lemmatizes — then feeds the resulting TF-IDF vector to all four models simultaneously. Each model votes, and the majority wins; on a 2-2 tie we use the higher average probability as a tie-breaker. In parallel, the fact checker runs on the **raw** text, because we still need to see the digits and punctuation. It uses spaCy for entity recognition, Wikipedia to verify organizations, and pattern-matching for scam-style phrases like 'forward this urgently' or unrealistic numbers like a 200% tax. The fact checker is advisory — it can raise warnings, but it never overrides the model verdict."

### Q&A this person should handle

- "What if the fact checker disagrees with the ensemble?" → the ensemble verdict still wins; the warnings are surfaced as advisory data the user can read.
- "Why strip digits in preprocessing if you want the fact checker to see them?" → exactly why we run the fact checker on the **raw** text in parallel, before preprocessing.
- "What if spaCy isn't installed?" → app prints a warning at startup, runs without fact checking; predictions still work.
- "Is this real-time?" → yes, prediction takes <1 second; SVM probability calculation is the bottleneck.

---

## Person 4 — Frontend / Demo / Documentation

### Owns

- [templates/index_professional.html](templates/index_professional.html) — home page
- [templates/about.html](templates/about.html) — about page
- [tests/test_model.py](tests/test_model.py)
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) and [README.md](README.md)
- **The live demo during the presentation**
- Q&A facilitation

### What to study

- The home page UI: input box, "Predict" button, the result card showing per-model votes and probabilities
- The bundled `SAMPLE_NEWS` examples in [app.py:25](app.py#L25) — five articles (3 real, 2 fake) ready to demo
- How to start the app (`python app.py`) and what to do if something fails

### Slide section: "Demo + tech stack"

1. **Slide A — Tech stack overview**
   - Backend: Python 3.8+, Flask
   - ML: scikit-learn, NLTK, joblib
   - Optional: spaCy + Wikipedia API
   - Frontend: vanilla JS, custom CSS (glassmorphism), Chart.js
2. **Slide B — Live demo (skip the slide, show the app)**
   - Show one **real** article being correctly classified.
   - Show one **fake** article being correctly classified.
   - Highlight the per-model vote breakdown and the fact-check warnings panel.
3. **Slide C — Tests + reproducibility**
   - `pytest tests/` runs the test suite.
   - Everything reproducible: `combine_datasets.py` → `train_models.py` → `app.py`. One command per stage.
   - All hyperparameters live in [config.py](config.py) — one file to tune.

### Live demo script (~2 min)

1. **Before presenting**, on your laptop:
   ```bash
   source .venv/bin/activate
   python app.py
   ```
   Confirm the page loads at `http://localhost:5000`.
2. **Demo step 1** — pick "India Launches Gaganyaan Mission" from `SAMPLE_NEWS`. Click Predict. Show: 4-0 vote → REAL, ~95% confidence.
3. **Demo step 2** — pick "200% Tax on Bank Deposits". Click Predict. Show: 4-0 vote → FAKE. Point at the fact-check warnings panel: "scam phrasing detected", "unrealistic percentage detected".
4. **Demo step 3** — paste a borderline real article and show how the per-model breakdown changes. This is where you talk about the ensemble's value.

### Speaking points (4–5 min)

> "Let me show you what this looks like end-to-end. _\[Open localhost:5000\]_ The user pastes any article into this box, hits Predict, and gets back a verdict in under a second. Look at the result card: it shows the **ensemble verdict**, the **confidence**, the **per-model votes**, and — if spaCy is installed — the fact-check warnings panel. _\[Demo a real article.\]_ Notice all four models agreed. _\[Demo a fake article.\]_ Same here — and the fact checker independently flags the suspicious '200%' number and the scam-style phrasing. The whole project is reproducible: one command builds the dataset, another retrains the models, a third runs the app. Every hyperparameter sits in `config.py` so you can tune and rerun in minutes."

### Q&A this person should handle

- "Can I try a custom article?" → yes, hand the laptop to the questioner.
- "How do I run this myself?" → point to §9 of `PROJECT_GUIDE.md`: 4 steps, ~5 minutes.
- "What's the limitation?" → corpus is English-only, biased toward US/India political news (ISOT is Reuters-heavy), and doesn't catch sophisticated AI-generated misinformation.
- "Future work?" → integrate transformer embeddings, add a confidence-based "uncertain" verdict, expand the fact-check pattern list, multi-language support.

---

## Recommended slide order

```
1. Cover / team intro          (Person 4)        ~30 sec
2. Problem statement           (Person 1)        ~30 sec
3. Datasets & data pipeline    (Person 1)        3–4 min
4. Models & training           (Person 2)        4–5 min
5. Application & ensemble      (Person 3)        3–4 min
6. Live demo                   (Person 4)        2–3 min
7. Tech stack & reproducibility(Person 4)        1–2 min
8. Q&A                         (all, P4 leads)   ~5 min
```

## Practice tips

- **Each person should be able to answer questions about their own files.** Read your file end-to-end at least once.
- **The demo must be rehearsed at least twice** with the laptop you're presenting on. Cache failures will look bad.
- **If asked about another person's area, defer:** "That's [name]'s part — let me hand off." This shows the team operates as a unit.
- **Have a backup plan if the live demo fails** — a short screen recording of a successful prediction, kept on the same laptop.

## Safety net before presenting

Run this checklist 30 minutes before:

```bash
# 1. App boots cleanly
python app.py

# 2. Test endpoint responds
curl http://localhost:5000/api/health
# expect: {"status":"healthy","models_loaded":true,...}

# 3. Tests pass
pytest tests/

# 4. Sample predictions work — open browser, click each SAMPLE_NEWS button
```

If any step fails, fix it before you present. The most common failure is missing NLTK data — run `python fix_nltk.py` to fix.
