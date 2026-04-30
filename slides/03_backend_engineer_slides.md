# Person 3 — Backend Engineer Slides

**Speaker**: _(name)_
**Section length**: ~3-4 minutes
**Number of slides**: 3

Each slide has:
- **Slide title** — the heading.
- **Content** — what to put on the slide.
- **Speaker notes** — what to actually say.
- **Visual suggestion** — diagram / layout idea.

---

## Slide 1 — Application architecture

### Title
> **Flask app: load once, serve fast**

### Content (put on slide)

**At startup** (`python app.py`):

1. Load `vectorizer.joblib` into memory
2. Load all 4 model `.joblib` files
3. Initialize spaCy fact checker (graceful skip if missing)
4. Start Flask on port 5000

**HTTP routes:**

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home page (input form) |
| `/predict` | POST | The ensemble prediction endpoint |
| `/about` | GET | Project info page |
| `/api/health` | GET | Health check (JSON) |

**Key design choice**: models load **once at startup**, not per-request. Per-request loading would add 100+ ms of disk I/O on every prediction.

### Speaker notes

> "When the Flask app starts up, the very first thing it does is load all the trained models from disk into memory — the vectorizer, plus the four classifiers. It also initializes the spaCy fact checker, with a graceful fallback if spaCy isn't installed. This is a deliberate design choice: we load once at startup, not on every request. Loading a 50-megabyte SVM from disk takes more than 100 milliseconds, and doing that on every prediction would tank our response time. The app exposes four routes, but the interesting one is `/predict` — that's where the magic happens. Let me walk you through what happens when a user hits Predict."

### Visual suggestion

- Top half of slide: startup flow diagram (vertical, with arrows).
- Bottom half: the four routes as a clean table.

---

## Slide 2 — The `/predict` flow

### Title
> **From user input to verdict — 8 steps**

### Content (put on slide)

```
       USER PASTES TEXT
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 1. Receive JSON {"text": "..."}         │
    └────────────────────────────────────────┘
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 2. TextPreprocessor.preprocess(text)    │
    │    (lowercase, strip URLs, lemmatize)   │
    └────────────────────────────────────────┘
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 3. vectorizer.transform([text])         │
    │    → 5000-dim TF-IDF vector             │
    └────────────────────────────────────────┘
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 4. Each of 4 models predicts            │
    │    → vote + probabilities               │
    └────────────────────────────────────────┘
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 5. Count votes → majority verdict       │
    │    (tie? avg probability decides)       │
    └────────────────────────────────────────┘
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 6. FactChecker.analyze(RAW text)        │  ◄── advisory, parallel
    └────────────────────────────────────────┘
              │
              ▼
    ┌────────────────────────────────────────┐
    │ 7. Return JSON to frontend              │
    │    verdict + per-model + fact-check     │
    └────────────────────────────────────────┘
```

**End-to-end latency: < 1 second.**

### Speaker notes

> "When a user hits Predict, here's what happens in under one second. We receive the text as JSON. We preprocess it through the same pipeline Person 2 walked you through — lowercase, strip URLs, tokenize, lemmatize. We vectorize it into a 5,000-dimensional TF-IDF vector. We then ask all four models to predict, and we record each model's vote plus its probability. We count the votes, decide the verdict by majority — with a probability-based tie-breaker if it's 2-2. In parallel, we run the fact checker on the **raw, untouched text** — and I'll explain why on the next slide. Finally, we return everything as JSON to the frontend. Total time: well under a second on a laptop."

### Visual suggestion

- Vertical pipeline diagram (the one in the content above), each box clearly labeled.
- Time annotations on the side: "preprocessing: ~10ms", "all 4 models: ~150ms", etc.

---

## Slide 3 — Ensemble + fact checker

### Title
> **Two layers of skepticism — voting + advisory checks**

### Content (put on slide)

**Layer 1: Ensemble voting**

```
4 models predict independently
        │
        ▼
   Count fake/real votes
        │
        ▼
  Majority wins (3-1 or 4-0)
  Tie (2-2) → higher avg probability wins
```

**Why majority vote, not weighted average?**
- Robust to one overconfident wrong model
- One outlier can be outvoted 3-1

**Layer 2: Fact checker (advisory only)**

Runs on the **raw text** (digits + punctuation intact):

| Check | What it looks for |
|---|---|
| **NER** (spaCy) | Organizations, locations, dates |
| **Wikipedia** | Does the first organization exist on Wikipedia? |
| **Numerical patterns** | Percentages > 100, distances > 500km, speeds > 350km/h |
| **Scam patterns** | "forward this", "share urgently", "shocking…!", "click here before" |

**Critical**: fact-check warnings **never override** the ensemble verdict — they're surfaced alongside it as advisory data.

### Speaker notes

> "We have two independent layers of skepticism. The first is the ensemble vote — four models predict in parallel, majority wins. We deliberately chose majority voting over a weighted probability average because it's more robust: one overconfident wrong model can be outvoted three-to-one. On a 2-2 tie, that's the only case where we fall back to averaged probabilities as a tie-breaker. The second layer is the fact checker, and it runs on the **raw, untouched text** — because Person 2's preprocessing strips digits and punctuation, but we still need to see things like '200%' or '1000 km/h' to catch them. The fact checker does named-entity extraction with spaCy, verifies the first organization on Wikipedia, and uses regex patterns for unrealistic numbers and scam-style phrasing. Critically, the fact checker is **advisory** — it can raise warnings, but it never overrides the ensemble verdict. Pattern matching is brittle, and we don't want regex rules vetoing the model's decision."

### Visual suggestion

- Two parallel "lanes" running top to bottom:
  - Left lane: "Ensemble Voting" with the 4 models → vote → verdict.
  - Right lane: "Fact Checker (raw text)" with NER, Wikipedia, regex.
- Both converge at the bottom into a final JSON response.
- Use distinct colors: one color for ensemble, another for fact-check.

---

## Closing line (transition to Person 4)

> "So that's how a request becomes a verdict on the backend. Now let me hand it over to [Person 4's name] — they'll show you what this actually looks like to a user, and we'll do a live demo."

---

## Pre-presentation checklist

- [ ] You can explain *why* models load at startup (avoid 100ms+ disk I/O per request)
- [ ] You can explain *why* fact checker runs on raw text (digits get stripped in preprocessing)
- [ ] You can explain *why* majority vote not weighted average (robust to one outlier)
- [ ] You know what happens on a 2-2 tie (averaged probability decides)
- [ ] You know what happens if spaCy is missing (graceful skip, predictions still work)
- [ ] Memorized: end-to-end latency < 1 second
- [ ] Ready for "what about scaling to 1000 req/sec" question (SVM probability calibration is the bottleneck)
