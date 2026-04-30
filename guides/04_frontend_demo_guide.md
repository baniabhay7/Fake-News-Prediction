# Person 4 — Frontend / Demo / Documentation Guide

**Your role**: own the presentation experience. You're the face of the demo — you walk the audience through the UI, run the live demo, handle reproducibility and tech stack, and lead Q&A. You don't go deepest on technical details (the others do that), but you tie it all together and make the project feel real.

**Files you own**:
- [templates/index_professional.html](../templates/index_professional.html) — the home page
- [templates/about.html](../templates/about.html) — about page
- [tests/test_model.py](../tests/test_model.py) — test suite
- [PROJECT_GUIDE.md](../PROJECT_GUIDE.md), [README.md](../README.md), [PRESENTATION_ROLES.md](../PRESENTATION_ROLES.md) — documentation
- **The live demo during the presentation**
- **Q&A facilitation**

**What Person 3 hands you**: a working `/predict` HTTP endpoint that returns JSON.
**What you hand off to the audience**: a demo they can try themselves.

---

## 1. Context (1 paragraph)

The previous three people explained the data, the models, and the backend. Your job is to (a) show what the user actually sees, (b) run a live demo with sample articles, (c) explain the tech stack and how someone could reproduce the whole project, and (d) lead the Q&A. You're closing the loop.

---

## 2. The UI — what the user sees

### Home page (`/` → `templates/index_professional.html`)

- **Header** with project title and team info.
- **Input box** — a large textarea where the user pastes an article.
- **Sample buttons** — pre-loaded examples from `SAMPLE_NEWS` in `app.py` (3 real, 2 fake). Click one and the textarea fills.
- **Predict button** — POSTs the text to `/predict`.
- **Result card** (appears after prediction):
  - Final verdict: **REAL NEWS** / **FAKE NEWS** with confidence percentage.
  - Vote breakdown: "4 votes FAKE / 0 votes REAL" or similar.
  - Per-model results — each of the 4 models with its individual prediction and probability.
  - A bar chart (Chart.js) visualizing the per-model probabilities.
  - **Fact-check panel** (if available) — extracted entities, Wikipedia verification, and any warnings.

### Visual design

- **Glassmorphism** — frosted glass cards with translucent backgrounds and blur effects.
- **Poppins font** from Google Fonts.
- **Color coding**: green for real, red for fake, with subtle gradients.
- Fully responsive (mobile-friendly).

### About page (`/about` → `templates/about.html`)

A static information page describing the project, the team, and the tech stack.

---

## 3. The live demo — the most important 2 minutes

### Pre-flight checklist (do this 30 minutes before presenting)

```bash
# 1. Activate the virtualenv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows

# 2. Boot the app
python app.py
```

Check the terminal output. You should see:

```
Vectorizer loaded successfully!
Model Naive Bayes loaded successfully!
Model Logistic Regression loaded successfully!
Model Random Forest loaded successfully!
Model SVM loaded successfully!
Successfully loaded 4 models!
Fact checker initialized.
 * Running on http://0.0.0.0:5000
```

If any line is missing or you see an error, **fix it now** — see "Common failures" below.

```bash
# 3. Smoke-test the API
curl http://localhost:5000/api/health
# Expect: {"status":"healthy","models_loaded":true,"available_models":["Naive Bayes","Logistic Regression","Random Forest","SVM"]}

# 4. Open the home page in a browser
# http://localhost:5000
# Click each of the 5 sample buttons — make sure all return predictions
```

### Demo script (~2 minutes)

You have **5 sample articles** pre-loaded:

| Sample | Type | Expected verdict |
|---|---|---|
| Gaganyaan Mission Successfully Launched | Real | REAL (4-0) |
| Mumbai Metro Line 3 Opens | Real | REAL (4-0) |
| Women's Cricket World Cup Victory | Real | REAL (4-0) |
| 200% Tax on All Bank Deposits | Fake | FAKE (4-0) |
| Hot Water Cures All Diseases | Fake | FAKE (4-0) |

**Demo flow**:

1. **(15s)** "Here's the home page. The user pastes an article into this box, hits Predict, and gets a verdict."
2. **(30s)** Click "Gaganyaan Mission". Hit Predict. As the result appears:
   > "All four models agreed: REAL, with high confidence. Notice the per-model breakdown — every classifier independently saw this as legitimate news. The fact checker also confirms: it found organizations like ISRO, no suspicious patterns, no warnings."
3. **(30s)** Click "200% Tax on Bank Deposits". Hit Predict.
   > "Now look at this one — also a unanimous vote, but FAKE. Critically, the fact-check panel here flags it independently: 'unrealistic percentage: 200', 'scam phrasing detected'. Two systems caught it from completely different angles."
4. **(30s)** **Optional power move**: paste a borderline article (a real article about a contentious topic, or a fake article that mimics real news style).
   > "This is more interesting — the vote is 3-1 instead of 4-0. The ensemble's value really shows on borderline cases like this, where individual models disagree."

### If the demo fails

**Always have a backup**: a screen recording of a successful prediction, kept as a `.mp4` on the same laptop. If the live app crashes, smoothly switch to "Let me show you what the demo *should* look like" without breaking pace.

### Common failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pandas` | Wrong Python | Activate the virtualenv (`source .venv/bin/activate`). |
| `LookupError: Resource punkt not found` | NLTK data missing | `python fix_nltk.py` |
| `Fact checker not available: ...` | spaCy missing | `python -m spacy download en_core_web_sm` (or skip it; predictions still work). |
| `Model X not found` | Missing `.joblib` files | `python train_models.py` (takes 5-10 min). |
| Port 5000 in use | Something else is running on 5000 | `lsof -ti:5000 | xargs kill` (macOS/Linux), or change `PORT` env var. |

---

## 4. Tech stack (a slide-worth of content)

| Layer | Technology |
|---|---|
| **Backend** | Python 3.8+, Flask |
| **ML / Classification** | scikit-learn — `MultinomialNB`, `LogisticRegression`, `RandomForestClassifier`, `SVC` |
| **NLP / Preprocessing** | NLTK — tokenization, stopwords, WordNet lemmatization |
| **Feature extraction** | scikit-learn `TfidfVectorizer` (5,000 features, unigrams + bigrams) |
| **Fact checking** | spaCy `en_core_web_sm` + `wikipedia-api` |
| **Frontend** | Vanilla JavaScript, custom CSS (glassmorphism), Chart.js, Poppins (Google Fonts) |
| **Persistence** | joblib (model serialization) |
| **Testing** | unittest / pytest |

### Why this stack

- **Flask** instead of FastAPI/Django — minimum ceremony, perfect for a single-page app.
- **scikit-learn** instead of PyTorch/TensorFlow — classical models hit 90%+ on this corpus with no GPU and 5 min of training.
- **Vanilla JS** instead of React/Vue — the UI is one page; no need for a build step.
- **joblib** instead of pickle — built for sklearn, faster, safer for large numpy arrays.

---

## 5. Reproducibility — three commands, one project

This is a great closing slide:

```bash
python combine_datasets.py    # 1. Build training_data.csv from raw datasets
python train_models.py      # 2. Train and save all 4 models
python app.py                 # 3. Run the web app
```

Plus tests:

```bash
pytest tests/                 # Run the test suite
```

Every hyperparameter is in [config.py](../config.py) — one file to tune. Edit there, rerun the trainer, and you have new models.

---

## 6. Q&A facilitation

You're the Q&A lead. Your job is to:

1. **Repeat the question** so the audience can hear it.
2. **Decide who answers** — your team, by area:
   - Data sources / class balance / dataset bias → **Person 1**
   - Model choice / hyperparameters / accuracy / training time → **Person 2**
   - Ensemble logic / fact checker / latency / API → **Person 3**
   - UI / demo / tech stack / future work → **You (Person 4)**
3. **Defer cleanly**: "Great question — that's actually [Person 1's name]'s area, let me hand it over."
4. **Cap long answers**: if a teammate goes too deep, gently redirect — "Let's take that offline / move to the next question."

### Questions YOU should answer

- **"How do I run this myself?"**
  Three commands: `combine_datasets.py`, `train_models.py`, `app.py`. Full instructions in `PROJECT_GUIDE.md` §9.

- **"Can I try a custom article?"**
  Yes — paste anything into the textarea. Hand the laptop to the questioner if there's time.

- **"What are the limitations?"**
  - English-only (NLTK + spaCy English models).
  - Biased toward US/India political and Reuters-style content (ISOT-heavy).
  - Doesn't catch sophisticated AI-generated misinformation that mimics legitimate writing styles.
  - The fact checker's pattern list is hand-coded — limited recall on novel scam patterns.

- **"What's the future work?"**
  - Add transformer embeddings (BERT, DistilBERT) for a 5th model.
  - Confidence-based "uncertain" verdict for borderline cases (e.g. all probabilities in 40-60% range).
  - Multi-language support.
  - Continuously expand the fact-check pattern library.
  - Live retraining on user-flagged articles (active learning).

- **"How big is the trained model?"**
  All four models + vectorizer: roughly 50-150 MB total. SVM with `probability=True` is the largest.

- **"Could this run on a phone?"**
  Naive Bayes + Logistic Regression — yes. Random Forest and SVM — borderline; you'd want quantization or model distillation.

---

## 7. Tests

Mention briefly that tests exist:

```bash
pytest tests/
```

Tests live in `tests/test_model.py` and exercise `TextPreprocessor` and the `FakeNewsClassifier` OO wrapper in `src/model.py`. They're unit tests — the live system is best verified by running the app and clicking sample buttons.

---

## 8. Quick reference — what to put on slides

You get **3-4 slides** (~4-5 minutes including the demo):

1. **Tech stack overview** — the table from §4.
2. **Live demo** — skip the slide, show the app instead. Have the URL `http://localhost:5000` already open in a browser tab.
3. **Reproducibility** — the three commands + config.py story.
4. **Future work + Q&A intro** — limitations, what we'd add next, "any questions?"

Detailed slide content is in [`slides/04_frontend_demo_slides.md`](../slides/04_frontend_demo_slides.md).

---

## 9. Closing the presentation

After the demo and tech stack, close with:

> "That's our project — we've shown you the dataset pipeline, the four-model ensemble, the backend that brings it together, and a working demo you just saw. Everything is reproducible from three commands, every parameter is tunable from one file. We're happy to take any questions."

Then open the floor.
