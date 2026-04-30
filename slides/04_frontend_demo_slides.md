# Person 4 — Frontend / Demo / Documentation Slides

**Speaker**: _(name)_
**Section length**: ~4-5 minutes (including the live demo)
**Number of slides**: 4 — but Slide 2 is the live demo (no actual slide content; switch to the browser)

Each slide has:
- **Slide title** — the heading.
- **Content** — what to put on the slide.
- **Speaker notes** — what to actually say.
- **Visual suggestion** — diagram / layout idea.

---

## Slide 1 — Tech stack overview

### Title
> **What it's built with**

### Content (put on slide)

| Layer | Technology |
|---|---|
| **Backend** | Python 3.8+ · Flask |
| **Machine Learning** | scikit-learn (Naive Bayes, Logistic Regression, Random Forest, SVM) |
| **NLP / Preprocessing** | NLTK (tokenize, stopwords, WordNet lemmatizer) |
| **Feature extraction** | TF-IDF (5,000 features, unigrams + bigrams) |
| **Fact checking** | spaCy `en_core_web_sm` + Wikipedia API |
| **Frontend** | Vanilla JavaScript · Custom CSS (glassmorphism) · Chart.js |
| **Persistence** | joblib (model serialization) |
| **Testing** | pytest / unittest |

**Why this stack:**
- **Flask** over FastAPI/Django — minimum ceremony, perfect for a single-page app.
- **scikit-learn** over PyTorch/TensorFlow — 90%+ accuracy with no GPU.
- **Vanilla JS** over React/Vue — one page, no build step needed.

### Speaker notes

> "Quick tour of the stack. Backend is Python with Flask — we picked Flask over FastAPI or Django because the project is a single-page app and Flask is the minimum ceremony you can get away with. The ML side is all scikit-learn — we use four classical algorithms instead of deep learning because we hit 90% accuracy with no GPU and 5 minutes of training. NLTK handles preprocessing, spaCy handles fact-checking entity extraction, and joblib handles serializing the trained models. The frontend is intentionally simple — vanilla JavaScript and custom CSS with a glassmorphism design, plus Chart.js for the per-model probability bars. No React, no build step. One page, one API."

### Visual suggestion

- A clean two-column table.
- Optionally: small library logos (Flask, scikit-learn, NLTK, spaCy) along the top.

---

## Slide 2 — Live demo (NO SLIDE CONTENT — switch to browser)

### Title (if you have a placeholder slide)
> **Demo time**

### What to do
**Don't put bullets on this slide.** Instead, switch to your browser tab on `http://localhost:5000` and run the demo live. Have the URL pre-loaded.

### Demo script (~2 minutes)

**Setup before presenting:**
1. App is running (`python app.py`)
2. Browser tab open on `http://localhost:5000`
3. Backup screen recording ready in case anything fails

**Run order:**

**(15s) Show the home page:**
> "Here's what the user sees. A clean input box, sample articles to try, and a Predict button."

**(30s) Click "Gaganyaan Mission" → hit Predict:**
> "All four models agreed: REAL, with high confidence. Look at the per-model breakdown — Naive Bayes, Logistic Regression, Random Forest, and SVM all independently classified this as real. The fact checker also picks up legitimate organizations like ISRO. No warnings."

**(30s) Click "200% Tax on Bank Deposits" → hit Predict:**
> "Now this one — also unanimous, but FAKE. And critically, look at the fact-check panel: it independently flags 'Unrealistic percentage: 200' and the scam-style language. Two systems caught this from completely different angles."

**(30s) Optional power move — paste a custom article or pick a borderline sample:**
> "Let me try something a little more ambiguous … notice the vote here is 3-1 instead of 4-0. This is exactly where the ensemble's value shows — individual models disagree, but the majority decides."

### Speaker notes

> "This is the part where the slides take a back seat — let me just show you the working app." [Switch to browser, run the demo above.]

### Visual suggestion

- If you must have something on the slide, put: a screenshot of the app + the URL `http://localhost:5000` in case the live demo fails.

### Backup plan

If the live app crashes:
- Have a **30-second screen recording** ready as a `.mp4` in the same folder as your slides.
- Smoothly say: "Let me show you what it looks like" and play the recording.
- **Don't** try to debug live. Move on, recover the energy.

---

## Slide 3 — Reproducibility

### Title
> **Three commands, one project**

### Content (put on slide)

**Reproduce the entire project:**

```bash
# 1. Build the balanced training corpus from raw datasets
python combine_datasets.py

# 2. Train and save all 4 models + vectorizer
python train_models.py

# 3. Run the web app
python app.py
```

**Tune anything**: edit [config.py](../config.py).

| Knob | Where it is | What it changes |
|---|---|---|
| `MAX_FEATURES = 5000` | TF-IDF | Vocabulary size |
| `NGRAM_RANGE = (1, 2)` | TF-IDF | Phrase capture |
| `RF_N_ESTIMATORS = 100` | Random Forest | Tree count |
| `SVM_KERNEL = 'linear'` | SVM | Decision boundary |
| `RANDOM_STATE = 42` | All splits/sampling | Reproducibility seed |

**Tests**: `pytest tests/`

### Speaker notes

> "The whole project reproduces from three commands. First, `combine_datasets.py` builds the training file from the raw datasets — Person 1 walked you through that. Second, `train_models.py` trains all four models with the hyperparameters from `config.py` and saves them as `.joblib` files. Third, `python app.py` boots the Flask app. That's it. Want to change a hyperparameter? Open `config.py`, edit one line, rerun the trainer, and you have new models in 5 minutes. We deliberately consolidated all tunable values into one file so iterating is fast."

### Visual suggestion

- Three large numbered command boxes top to bottom.
- Below: the `config.py` table with arrows pointing to "→ retrain → new models".

---

## Slide 4 — Limitations + future work + closing

### Title
> **Where we go from here**

### Content (put on slide)

**Current limitations:**
- English-only (NLTK + spaCy English models)
- Biased toward US/India political content (ISOT-heavy training data)
- Doesn't catch sophisticated AI-generated misinformation
- Fact-check pattern list is hand-coded — limited coverage of novel scams

**Future work:**
- Transformer embeddings (BERT, DistilBERT) as a 5th model
- "Uncertain" verdict for borderline cases (all probabilities in 40-60% range)
- Multi-language support (Hindi, Spanish, Mandarin priority)
- Active learning — retrain on user-flagged articles
- Continuously expand the fact-check pattern library

**Final note:**
- 90%+ accuracy
- < 1 second prediction latency
- Reproducible from 3 commands
- All hyperparameters in 1 file

### Speaker notes

> "Honest limitations: this is English-only because of the NLTK and spaCy models we use. Our training data leans heavily on US and Indian political content, so the model can pick up writing-style bias rather than purely factual bias. And it doesn't catch sophisticated AI-generated misinformation that mimics legitimate writing. For future work, the obvious next step is integrating transformer embeddings as a fifth model — that could push accuracy past 92-93%. We'd also like to add an 'uncertain' verdict for genuinely borderline cases, multi-language support, and active learning where the system retrains itself on articles users flag. To wrap up: we hit 90% accuracy across four models, sub-second prediction latency, full reproducibility from three commands, and all hyperparameters in a single tunable file. Happy to take any questions."

### Visual suggestion

- Two columns: "Current Limitations" (left) and "Future Work" (right).
- Bottom strip: the four key stats as big numbers — "90%+", "<1s", "3 commands", "1 config file".

---

## Closing line (open Q&A)

> "That's our project. Happy to take any questions."

Then look around the room and wait. **Don't fill the silence yourself** — if it takes 5 seconds for the first hand to go up, that's fine.

---

## Q&A — your job is to facilitate

When a question comes in:

1. **Repeat the question** so the back of the room can hear it.
2. **Hand it to the right person**:
   - Datasets / class balance / sources → **Person 1**
   - Model choice / hyperparameters / accuracy → **Person 2**
   - Ensemble logic / fact checker / API / latency → **Person 3**
   - UI / demo / future work / tech stack → **You**
3. **If you don't know who owns it**, take a beat: "Good question — [Person X's name], do you want to take that one?"
4. **Cap long answers** if a teammate goes too deep — "Let's take that offline."

---

## Pre-presentation checklist (most critical of any role)

**30 minutes before:**
- [ ] App is running (`python app.py`) — verify in terminal
- [ ] `curl http://localhost:5000/api/health` returns 200 with all 4 models loaded
- [ ] Browser tab is open on `http://localhost:5000`
- [ ] All 5 sample buttons work end-to-end (click each, verify a verdict appears)
- [ ] Backup screen recording is on the desktop, ready to play
- [ ] `config.py` is open in another tab in case of "show me a hyperparameter" questions
- [ ] You've practiced the demo flow at least twice from a cold start

**During the demo:**
- [ ] Maximize the browser window — projector audiences need big text
- [ ] Don't apologize for things that don't matter ("sorry the spinner is slow")
- [ ] If something fails, smoothly switch to the recording — don't debug live

**During Q&A:**
- [ ] Repeat each question for the back of the room
- [ ] Defer cleanly to teammates — don't try to answer everything yourself
