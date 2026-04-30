# Person 2 — ML Engineer Slides

**Speaker**: _(name)_
**Section length**: ~4-5 minutes (you have the densest technical content)
**Number of slides**: 4

Each slide has:
- **Slide title** — the heading.
- **Content** — what to put on the slide.
- **Speaker notes** — what to actually say.
- **Visual suggestion** — diagram / layout idea.

---

## Slide 1 — Preprocessing pipeline

### Title
> **From raw text to clean tokens**

### Content (put on slide)

Every article goes through 8 steps before any ML touches it:

1. **Lowercase** the text
2. **Strip URLs** (`http`, `www`)
3. **Strip emails, @mentions, #hashtags**
4. **Strip digits and punctuation** → only `[a-z\s]` survives
5. **Tokenize** — split into words (NLTK)
6. **Remove English stopwords** — *the, a, is, on, …*
7. **Lemmatize** — *running → run, better → good* (WordNet)
8. **Re-join** into a clean string

**Why**: smaller vocabulary, same words map to same tokens regardless of case or grammatical form, no irrelevant noise.

**Trade-off**: digits get stripped — so the fact checker (Person 3) has to run on the raw text, before this pipeline.

### Speaker notes

> "Before any model sees the text, we clean it up. Lowercase, strip URLs, strip digits and punctuation, tokenize into individual words, remove English stopwords like 'the' and 'is', and finally lemmatize — that turns 'running' into 'run' and 'better' into 'good' so different forms of the same word collapse together. After this pipeline, the vocabulary is much smaller and much more informative. Worth flagging: digits get stripped here, which is why our fact checker — Person 3's part — has to run on the raw text in parallel, before this preprocessing happens."

### Visual suggestion

- A horizontal flow diagram with the 8 steps as labelled boxes.
- A "before / after" example pair below it:
  - Before: "BREAKING!!! @user Trump tweeted about https://example.com on January 5th, 2024 #fakenews"
  - After: "breaking trump tweet january"

---

## Slide 2 — TF-IDF feature extraction

### Title
> **Turning text into numbers — TF-IDF with bigrams**

### Content (put on slide)

**TF-IDF** (Term Frequency – Inverse Document Frequency):

```
TF-IDF(term, doc) = TF(term, doc) × log(N / DF(term))
```

A word that appears in *every* document → near-zero weight (uninformative).
A word that appears in only a few documents → high weight (signal).

**Our config:**

| Parameter | Value | Why |
|---|---|---|
| `max_features` | 5,000 | Top 5,000 most informative terms |
| `ngram_range` | (1, 2) | Single words **+ two-word phrases** |
| `min_df` | 1 | Keep rare terms |
| `max_df` | 1.0 | Don't filter common terms |

**Why bigrams matter**: phrases like *"breaking news"*, *"shocking discovery"*, *"fake media"* are themselves fake-news signals — single words miss them.

**Output**: each article → sparse 5,000-dimensional vector.

### Speaker notes

> "Once the text is clean, we convert it into numbers using TF-IDF. The intuition is simple: words that appear everywhere — like 'news' in a news corpus — are useless for classification, so they get near-zero weight. Words that appear in only a few documents but appear strongly there get high weight. We use the top 5,000 most informative terms, and crucially, we include both single words AND two-word phrases. Bigrams catch things like 'breaking news', 'shocking discovery', 'fake media' — those phrases themselves are strong signals of fake content, and you'd miss them if you only looked at single words."

### Visual suggestion

- The TF-IDF formula prominently displayed.
- A small table comparing weights: low for "the", "news"; high for "shocking", "scandal".
- A simple visual: text → 5,000-dim vector with mostly zeros.

---

## Slide 3 — The four classifiers, on purpose

### Title
> **Four classifiers from four different families**

### Content (put on slide)

| Model | Family | Key hyperparameters | Why this one |
|---|---|---|---|
| **Naive Bayes** | Probabilistic | `alpha=0.1` | Fast baseline; great for bag-of-words |
| **Logistic Regression** | Linear | `max_iter=1000` | Calibrated probabilities; needed for tie-break |
| **Random Forest** | Tree ensemble | `n_estimators=100`, `max_depth=20` | Captures non-linear feature interactions |
| **SVM** | Margin-based | `kernel='linear'`, `probability=True` | Gold standard for high-dim sparse text |

**Why four different families?**
Their *errors are uncorrelated*. When one model is wrong, the others are usually right. That's what makes the ensemble work — voting four identical models would gain nothing.

**Train/test split**: 80% / 20%, `stratify=y`, `random_state=42`.

**All hyperparameters live in `config.py`** — single source of truth for tuning.

### Speaker notes

> "We don't just pick one model — we train four, and we pick them from four completely different families on purpose. Naive Bayes gives us a probabilistic baseline. Logistic Regression gives us a linear classifier with well-calibrated probabilities. Random Forest is a tree ensemble — it catches non-linear interactions the linear models miss. And linear-kernel SVM is the classical gold standard for high-dimensional sparse text. The reason we pick four *different* families is that their errors don't overlap — when one model is wrong, the others are typically right. Voting four identical models would gain nothing; voting four diverse models gives you the safety net of an ensemble."

### Visual suggestion

- Four icons side by side, each in its own coloured card.
- Below each: 1-line description.
- Bottom: an "ensemble" arrow tying them together.

---

## Slide 4 — Performance results

### Title
> **90% accuracy, with the ensemble on top**

### Content (put on slide)

**Performance on the held-out 20% test set:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **SVM** | **90.4%** | 90% | 90% | **91%** |
| **Logistic Regression** | 90.0% | 90% | 90% | 90% |
| **Random Forest** | 88.0% | 88% | 88% | 88% |
| **Naive Bayes** | 83.1% | 84% | 83% | 83% |

**The ensemble adds robustness**:
- 4-0 votes → very high confidence (most cases)
- 3-1 votes → solid majority, typically Naive Bayes is the outlier
- 2-2 votes → the rare borderline case (broken by averaged probability)

→ Person 3 explains the voting mechanics next.

### Speaker notes

> "Here are our results on the held-out 20% test set. SVM is our best single model at 90.4% accuracy, with Logistic Regression a hair behind. Random Forest is solid at 88%, and Naive Bayes is the weakest at 83% — but the gap is the price of how fast and simple it is, and we keep it for ensemble diversity. The ensemble's effective accuracy is at least as high as the best single model, typically a touch higher because the four models' errors don't correlate. In the next section, [Person 3's name] is going to show you exactly how we combine these four votes into one final verdict."

### Visual suggestion

- The performance table styled as a leaderboard.
- A small visual at the bottom: "4 models → 1 ensemble → 1 verdict" arrow leading into Person 3's section.

---

## Closing line (transition to Person 3)

> "We've got four trained models, each with its own perspective. The next question is: how do we combine them at prediction time, and how do we serve this to a user? Over to [Person 3's name]."

---

## Pre-presentation checklist

- [ ] You can explain TF-IDF in one sentence ("up-weights distinctive words, down-weights common ones")
- [ ] You can explain *why* bigrams matter (catch phrases like "breaking news")
- [ ] You can explain *why* four different model families (uncorrelated errors)
- [ ] Memorized: 5000 features, ngrams (1,2), 80/20 split, seed 42
- [ ] Memorized: SVM 90.4% / LR 90% / RF 88% / NB 83%
- [ ] Ready for "why not deep learning / BERT" question (ROI vs. compute)
- [ ] Ready for "how do you avoid overfitting" question (max_features cap, max_depth, holdout)
