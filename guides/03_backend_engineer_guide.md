# Person 3 — Backend Engineer Guide

**Your role**: own the runtime — the Flask application that takes user requests, runs the four models in parallel, combines their votes, runs the optional fact checker, and returns a verdict. You're the bridge between Person 2's trained models and Person 4's UI.

**Files you own**:
- [app.py](../app.py) — the Flask application
- [src/data_processing.py](../src/data_processing.py) — `TextPreprocessor` (called at request time)
- [src/fact_checker.py](../src/fact_checker.py) — spaCy + Wikipedia advisory layer
- The HTTP routes: `/`, `/predict`, `/about`, `/api/health`

**What Person 2 hands you**: 5 `.joblib` files in `models/` — the vectorizer + 4 trained classifiers.
**What you hand off to Person 4**: a JSON-returning `/predict` endpoint that the frontend calls.

---

## 1. Context (1 paragraph)

Person 2 trained four classifiers and saved them to disk. Your job is to load those models once at startup, expose a `/predict` HTTP endpoint, and on every request: preprocess the input text, vectorize it, ask all four models to vote, optionally run the fact checker, and return a structured JSON response. Everything is real-time — sub-second latency per prediction.

---

## 2. Application architecture — what happens when the app starts

```
python app.py
    │
    ├─► load_models()  ─── reads models/*.joblib into memory
    │       ├─ vectorizer.joblib      (TfidfVectorizer)
    │       ├─ naive_bayes_model.joblib
    │       ├─ logistic_regression_model.joblib
    │       ├─ random_forest_model.joblib
    │       └─ svm_model.joblib
    │
    ├─► initialize FactChecker()  ─── loads spaCy en_core_web_sm
    │                                  (gracefully skipped if spaCy missing)
    │
    └─► Flask serves routes on port 5000
            ├─ GET  /              → home page (input form)
            ├─ POST /predict       → ensemble prediction
            ├─ GET  /about         → about page
            └─ GET  /api/health    → health check JSON
```

**Key design choice**: models are loaded **once** at startup, not per-request. Loading a 50 MB SVM from disk takes 100ms+; doing that on every request would tank latency. Instead, models live as global variables in `app.py`, ready to be called instantly.

---

## 3. The `/predict` route — step by step

This is the heart of your section. Walk through what happens when a user clicks "Predict":

```
USER PASTES TEXT
    │
    ▼
1. Receive JSON {"text": "..."}
    │
    ▼
2. TextPreprocessor.preprocess(text)
    └─► lowercase, strip URLs, strip digits, tokenize, stopwords, lemmatize
    │
    ▼
3. vectorizer.transform([processed_text])
    └─► sparse 5000-dim TF-IDF vector
    │
    ▼
4. For each of the 4 models:                          ┐
        prediction = model.predict(vector)            │   PARALLEL
        probabilities = model.predict_proba(vector)   │   (loop, but logically independent)
        record vote (fake or real) + probabilities    ┘
    │
    ▼
5. Count fake_votes vs real_votes
    │
    ▼
6. Decide final verdict:
        if fake_votes > real_votes  → FAKE
        if real_votes > fake_votes  → REAL
        if 2 vs 2 (tie)             → fall back to higher AVG probability
    │
    ▼
7. Run FactChecker.analyze(RAW text)   ◄── note: RAW, not preprocessed
    └─► returns entities + warnings (advisory only)
    │
    ▼
8. Return JSON {
       prediction, confidence, fake_probability, real_probability,
       fake_votes, real_votes, individual_results, fact_check
   }
```

---

## 4. The ensemble — majority voting with a tie-breaker

```python
final_verdict = 'FAKE NEWS' if fake_votes > real_votes else 'REAL NEWS'
if fake_votes == real_votes:
    # tie-break with average probabilities
    final_verdict = 'FAKE NEWS' if avg_fake_prob > avg_real_prob else 'REAL NEWS'
```

### Why majority vote and not weighted average / single best model?

- **Single best model** (just SVM) — wastes the other three. SVM has 90.4% accuracy; one wrong call in ten is high-stakes for fake news.
- **Weighted average of probabilities** — sensitive to one overconfident model. If Random Forest screams 99% fake while three others are weakly real, the average gets dragged toward fake.
- **Majority vote** — democratic. One overconfident model can be outvoted 3-1. Safety-first.

### The tie-break

A 2-2 tie is the only case where votes don't decide. We then average the probabilities — this is exactly the case where probabilities should matter, because two models disagree confidently and we need a tiebreaker.

### Vote distributions in practice

- **4-0 votes** → all four models agree → very high confidence, almost always correct.
- **3-1 votes** → one outlier — typically Naive Bayes (the weakest model). Still confident.
- **2-2 votes** → genuine borderline case. The fact checker's warnings become especially useful here.

---

## 5. The fact checker — an advisory second opinion

The fact checker (`src/fact_checker.py`) runs in parallel with the ensemble. It's not a classifier — it doesn't output FAKE/REAL. Instead it surfaces *warnings* that the user can read alongside the verdict.

### Why it runs on RAW text

Person 2's preprocessing strips digits and special characters. But the fact checker looks for things like:

- *"200% tax"* (number + %)
- *"1000 km/h"* (number + unit)
- *"Forward this!"* (punctuation matters)

If we ran it on preprocessed text, those signals would be gone. So the fact checker gets the **raw, untouched** input — it's a parallel pipeline, not a downstream consumer.

### Three checks

1. **Named-entity extraction** with spaCy `en_core_web_sm`:
   - Extracts organizations, locations, dates, infrastructure (e.g. *"Mumbai Metro Rail Corporation"*, *"Sriharikota"*).
   - Useful context for the user even if no warning fires.

2. **Wikipedia verification**:
   - Takes the first organization found.
   - Calls the Wikipedia API: does this entity have a Wikipedia page?
   - If not, that's a (weak) signal the article might be fabricating sources.

3. **Pattern checks** (regex on raw text):
   - **Unrealistic numbers**: percentages > 100, distances > 500 km, speeds > 350 km/h.
   - **Scam phrases**: *"forward this"*, *"share urgently"*, *"WhatsApp will charge"*, *"breaking…!"*, *"shocking…!"*, *"click here before"*, *"send to N people"*.

### Advisory only — never overrides

The fact checker's output is added to the JSON response under `fact_check`, but the `prediction` field is always determined by the ensemble vote. The frontend can choose to display the warnings prominently, but they don't change the verdict. This is deliberate — pattern matching is brittle, and we don't want regex rules vetoing the model's decision.

### Graceful degradation

If spaCy or `en_core_web_sm` is not installed, the app prints a warning at startup and `FACT_CHECKER_AVAILABLE` is set to `False`. The `/predict` route checks this flag and skips the fact-check step. Predictions still work — the response just doesn't have a `fact_check` field.

---

## 6. The full JSON response

```json
{
  "success": true,
  "prediction": "FAKE NEWS",
  "confidence": 87.5,
  "fake_probability": 87.5,
  "real_probability": 12.5,
  "fake_votes": 4,
  "real_votes": 0,
  "individual_results": {
    "Naive Bayes":         { "prediction": "FAKE NEWS", "fake_probability": 92.1, "real_probability": 7.9 },
    "Logistic Regression": { "prediction": "FAKE NEWS", "fake_probability": 88.0, "real_probability": 12.0 },
    "Random Forest":       { "prediction": "FAKE NEWS", "fake_probability": 81.5, "real_probability": 18.5 },
    "SVM":                 { "prediction": "FAKE NEWS", "fake_probability": 88.4, "real_probability": 11.6 }
  },
  "model_used": "Ensemble (Majority Vote)",
  "decision_type": "Multi-Model Consensus",
  "fact_check": {
    "entities": { "organizations": ["Finance Ministry"], "locations": [], "dates": [], "infrastructure": [] },
    "numerical_issues": [{"reason": "Unrealistic percentage: 200"}],
    "verification": [],
    "warnings": ["Unrealistic percentage: 200"]
  }
}
```

The frontend (Person 4) renders all of this — the verdict, the per-model votes (often as a chart), and the fact-check warnings panel.

---

## 7. Other routes

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Renders `templates/index_professional.html` (Person 4's UI). Passes the model-loaded flag and the `SAMPLE_NEWS` examples for one-click demos. |
| `/about` | GET | Renders `templates/about.html` — project info page. |
| `/api/health` | GET | Returns `{status, models_loaded, available_models}`. Good for monitoring or smoke-testing. |
| `/predict` | POST | The endpoint described above. |

---

## 8. What you should know cold

- **"What if the fact checker disagrees with the ensemble?"**
  The ensemble verdict still wins. The fact checker is advisory — it surfaces warnings the user can read, but it doesn't override the model's decision.

- **"Why strip digits in preprocessing if you want the fact checker to see them?"**
  Exactly why we run the fact checker on the **raw** text in parallel, before the preprocessing pipeline touches it.

- **"What if spaCy isn't installed?"**
  The app prints a warning at startup, `FACT_CHECKER_AVAILABLE` becomes `False`, and `/predict` skips the fact-check step. Predictions still work.

- **"Is this real-time?"**
  Yes. Models load once at startup (~2 seconds). Per-request latency is dominated by SVM's `predict_proba` (~100-300ms). Total response is well under a second.

- **"What's the bottleneck if you scaled this to 1000 requests/sec?"**
  SVM probability calibration. You'd either drop SVM, switch to a faster kernel, or precompute probabilities differently. For a class project, the current setup is fine.

- **"Why majority vote and not weighted ensemble?"**
  Weighted ensembles are sensitive to one overconfident wrong model. Majority vote is more robust — one outlier can be outvoted 3-1.

- **"What happens on a 2-2 tie?"**
  Fall back to the higher *average* probability across the four models. This is the only case where probabilities decide.

- **"Could the fact checker fire false positives?"**
  Absolutely. Regex rules are brittle. That's why we explicitly designed it as advisory — the user sees the warning but isn't forced to act on it.

- **"Why use Flask and not FastAPI / Django?"**
  Flask is the simplest possible choice for a single-page app. Zero ceremony, fast to write, easy to deploy. FastAPI would give us auto-generated docs but the project doesn't need them.

---

## 9. Quick reference — what to put on slides

You get **3 slides** (~3-4 minutes):

1. **Application architecture** — startup flow + the four routes.
2. **The `/predict` flow** — the 8-step pipeline diagram from §3.
3. **Ensemble + fact checker** — majority vote rule and the advisory layer.

Detailed slide content is in [`slides/03_backend_engineer_slides.md`](../slides/03_backend_engineer_slides.md).

---

## 10. Hand-off to Person 4

After your section, say:

> "So that's how a request becomes a verdict on the backend. Now let me hand it over to [Person 4's name] — they'll show you what this actually looks like to a user, and we'll do a live demo."
