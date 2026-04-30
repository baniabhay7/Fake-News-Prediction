# Person 1 — Data Engineer Guide

**Your role**: own the data layer. You explain *where the data came from, how it's combined, and what's special about the final training file*. The rest of the team builds on top of what you produce.

**Files you own**:
- `data/News-_dataset/Fake.csv` — ISOT fake news (raw)
- `data/News-_dataset/True.csv` — ISOT real news (raw)
- `data/WELFake_Dataset.csv` — WELFake dataset (raw)
- `data/training_data.csv` — combined balanced corpus (your output)
- `data/sample_data.csv` — small demo file
- [combine_datasets.py](../combine_datasets.py) — your script

**What you hand off to Person 2**: the file `data/training_data.csv` with two columns, `text` and `label`, perfectly class-balanced.

---

## 1. The problem (1 paragraph context)

The team is building a fake news classifier. Given a news article, the system labels it `REAL` or `FAKE` using four ML models in an ensemble. Before any of that machine-learning work can happen, **someone has to assemble a clean, balanced training corpus** from public datasets. That's your job.

---

## 2. The two raw datasets

### ISOT Fake News Dataset

- **Source**: Information Security and Object Technology research lab, University of Victoria.
- **Files**: two CSVs split by class.
  - `Fake.csv` — 23,481 fake political news articles (label = 1 after we add it).
  - `True.csv` — 21,417 real Reuters news articles (label = 0 after we add it).
- **Columns**: `title, text, subject, date`
- **Time period**: 2016–2017
- **Topic**: heavily political (US elections era)
- **Why we use it**: the gold-standard academic dataset for fake news detection research.

### WELFake Dataset

- **Source**: Verma et al., IEEE Transactions on Computational Social Systems (2021). Available on Kaggle.
- **File**: a single CSV.
  - `WELFake_Dataset.csv` — 72,134 articles, already labeled.
- **Columns**: `(unnamed index), title, text, label` (1 = fake, 0 = real)
- **Time period**: broader, including 2020–2023
- **Topic**: covers COVID-19 misinformation, health hoaxes, modern fake news patterns.
- **Why we use it**: ISOT is too narrow on its own — WELFake adds modern, post-COVID misinformation our model needs to recognize.

---

## 3. The two derived working files

| File | Rows | Purpose |
|---|---|---|
| `data/training_data.csv` | **11,632** (5,816 fake + 5,816 real) | The actual file `retrain_models.py` reads. Two columns only: `text, label`. Perfectly class-balanced. **This is what you produce.** |
| `data/sample_data.csv` | 15 (8 real + 7 fake) | Tiny hand-picked demo set. **NOT the training set** — it's a small example file used only as a fallback if `training_data.csv` is missing. |

It's important to be clear about this distinction in the presentation: `sample_data.csv` is just a few rows for show; the **real** training data is the 11,632-row balanced file you build.

---

## 4. How `training_data.csv` is built

Implemented in [combine_datasets.py](../combine_datasets.py). Walk through these steps:

### Step-by-step recipe

1. **Load each raw source** and attach a `label` column where it doesn't already exist.
   - `Fake.csv` → label `1`
   - `True.csv` → label `0`
   - `WELFake_Dataset.csv` → already has `label`, keep it.

2. **Keep only `text` and `label`**. Drop `title`, `subject`, `date`, `(index)` — they're not features the model uses.

3. **Concatenate** all three sources into one DataFrame.

4. **Clean**: drop rows with empty/NaN text, then de-duplicate on the `text` column. Removes garbage rows and prevents the same article appearing in both train and test splits.

5. **Class-balance**: separate the combined data into two pools (fake and real). The smaller pool determines how many we can keep per class. Sample from each pool to a 50/50 split.

6. **Downsample to 11,632 rows total** (5,816 per class). Why this size?
   - SVM training time scales roughly **quadratically** with data size — 100k rows would take hours; 11k rows trains in 2-3 minutes.
   - 11k articles is large enough for stable test accuracy on a 4-model ensemble.
   - The original project author landed on this number; we keep it for reproducibility.

7. **Shuffle** with a fixed `random_state=42` so anyone running the script gets the exact same file.

8. **Write** to `data/training_data.csv` with columns `text,label` only.

### One-command reproducibility

```bash
python combine_datasets.py
```

The script prints a funnel showing how many rows survive each filtering step. End result is always the same 11,632-row balanced file.

---

## 5. Why class balancing matters

Without balancing, an algorithm trained on, say, 70% real / 30% fake learns to default to "real" — it gets 70% accuracy without learning anything. By forcing exactly 50/50:

- The model can't cheat by guessing the majority class.
- Accuracy becomes a meaningful metric (a coin flip would only get 50%).
- Precision and recall stay symmetric across both classes.

The `stratify=y` flag in the train/test split (Person 2's part) preserves this 50/50 ratio inside both the training set and the held-out test set.

---

## 6. What you should know cold

For Q&A, be ready to answer:

- **"Why didn't you use all 117k rows from the raw datasets?"**
  After balancing, the smaller class (real news after de-duplication) is the bottleneck. We then downsampled further to 5,816/class to keep SVM training time manageable on a laptop.

- **"Why two datasets and not one?"**
  ISOT alone covers only 2016–2017 political content. WELFake adds 2020–2023 articles including COVID-era health misinformation, which is essential for modern fake-news detection.

- **"Could you have used real-time scraped data?"**
  Yes, but labeling is the hard part. Pre-labeled academic datasets give us trustworthy ground truth. Live scraping would need a separate fact-checking labeling pipeline.

- **"How do you avoid data leakage between train and test?"**
  De-duplication on the `text` column removes exact duplicates. The 80/20 split happens *after* de-duplication, with `stratify=y` and `random_state=42`.

- **"What if WELFake and ISOT contain the same article?"**
  De-duplication catches them — exact text matches are dropped.

- **"Are these datasets biased?"**
  Honestly, yes — ISOT is heavily Reuters-sourced for the "real" class and political-blog-sourced for the "fake" class. The model can pick up writing-style bias instead of factual bias. That's a real limitation we mention in future work.

---

## 7. Quick reference — what to put on slides

You get **3 slides** (~3-4 minutes of speaking time):

1. **Where the data comes from** — ISOT + WELFake, with row counts and why we chose them.
2. **How the data is combined and balanced** — the funnel diagram (117k → cleaned → 11,632 balanced).
3. **`training_data.csv` vs `sample_data.csv`** — what each is, and the handoff to Person 2.

Detailed slide content is in [`slides/01_data_engineer_slides.md`](../slides/01_data_engineer_slides.md).

---

## 8. Hand-off to Person 2

After your section, say:

> "So we have 11,632 perfectly balanced articles, ready to be turned into something a machine learning model can actually understand. Over to [Person 2's name] to walk you through how we extract features and train the models."

That sets up the ML Engineer's section cleanly.
