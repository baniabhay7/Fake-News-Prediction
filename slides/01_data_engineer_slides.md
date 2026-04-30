# Person 1 — Data Engineer Slides

**Speaker**: _(name)_
**Section length**: ~3-4 minutes
**Number of slides**: 3

Each slide below has:
- **Slide title** — what to put at the top.
- **Content** — bullets / table / diagram to put on the slide.
- **Speaker notes** — what to actually *say* (not read off the slide).
- **Visual suggestion** — optional design ideas.

Copy the content sections directly into your slide tool.

---

## Slide 1 — Where the data comes from

### Title
> **Two public datasets, one combined corpus**

### Content (put on slide)

**ISOT Fake News Dataset** — University of Victoria
- `Fake.csv` — 23,481 fake political articles → label = 1
- `True.csv` — 21,417 real Reuters articles → label = 0
- Time period: 2016–2017
- Topic: US political news

**WELFake Dataset** — IEEE benchmark, Kaggle
- 72,134 articles, already labeled
- Time period: 2020–2023
- Topic: COVID-era misinformation, broader

**Combined raw input: ~117,000 labeled articles**

### Speaker notes (what to say)

> "We use two publicly released datasets. The first is ISOT from the University of Victoria — that's the gold-standard academic dataset, with about 45,000 political articles from 2016 to 2017, split across two CSV files for fake and real. But ISOT alone is too narrow. So we add WELFake, a Kaggle benchmark with 72,000 articles spanning the COVID era — that gives us modern misinformation patterns, health hoaxes, the kind of fake news that's circulating today. Together, that's roughly 117,000 labeled articles to start from."

### Visual suggestion

- Two coloured cards side by side (one per dataset), with row counts in big text.
- An arrow merging them into a "Combined raw" pile.

---

## Slide 2 — Cleaning and balancing the data

### Title
> **From 117,000 raw rows to 11,632 balanced articles**

### Content (put on slide)

**The pipeline** (top to bottom funnel):

```
117,000 raw rows
       │
       ▼
Drop NaN / empty text
       │
       ▼
De-duplicate by text
       │
       ▼
Class-balance (50/50)
       │
       ▼
Sample down to 11,632
       │
       ▼
Shuffle (seed = 42)
       │
       ▼
training_data.csv
   5,816 fake + 5,816 real
   Perfectly balanced
```

**Why balanced?** Without balancing, the model would learn to default to the majority class instead of actually learning fake-vs-real patterns.

**Why 11,632 and not all 117k?** SVM training scales quadratically with data size — 11k articles trains in 2-3 minutes; 100k+ would take hours.

### Speaker notes

> "Before we hand this to the ML team, we have to clean it up. We drop empty rows, de-duplicate, and then — this is critical — we class-balance to exactly 50/50. Without balancing, the model would just learn to predict whichever class is more common instead of actually learning fake-vs-real patterns. We also downsample to 11,632 articles total. The reason is practical: SVM training time scales roughly with the square of dataset size, so 11,000 rows trains in under three minutes; 100,000 rows would take hours. The whole pipeline is one command, `python combine_datasets.py`, with a fixed random seed so anyone can reproduce the exact same file."

### Visual suggestion

- A vertical funnel diagram with each stage labelled.
- End cap: a green box highlighting the "5,816 + 5,816 = 11,632 perfectly balanced" stat.

---

## Slide 3 — `training_data.csv` vs `sample_data.csv`

### Title
> **What we hand off to the ML team**

### Content (put on slide)

| File | Rows | Purpose |
|---|---|---|
| `data/training_data.csv` | **11,632** (5,816 + 5,816) | The actual training file. Two columns: `text, label`. Used by `retrain_models.py`. |
| `data/sample_data.csv` | 15 | A tiny demo file — NOT the training set. Just a small example illustrating the format. |

**Hand-off**:
The ML team takes `training_data.csv` and turns it into trained models. We don't touch the data again after this.

**Reproducibility**:
- Source: ISOT (academic) + WELFake (Kaggle) — both publicly released.
- Combined with `combine_datasets.py`.
- Random seed `42` everywhere.

### Speaker notes

> "Important distinction: we have two CSV files in our data folder. `training_data.csv` is the real one — 11,632 rows, balanced, two columns: text and label. That's what feeds into the ML models. `sample_data.csv` is tiny — only 15 rows — and it's purely for demonstration. It's a small example file. Don't confuse the two. After this stage, we hand `training_data.csv` to [Person 2's name] and the data side is done. Over to them to walk you through the machine learning."

### Visual suggestion

- Two side-by-side cards: one prominent (training_data.csv), one small/grey (sample_data.csv).
- Arrow from `training_data.csv` to "Person 2 / ML Engineer" on the next section.

---

## Closing line (transition to Person 2)

> "So we have 11,632 perfectly balanced articles, ready to be turned into something a machine learning model can actually understand. Over to [Person 2's name] to walk you through how we extract features and train the models."

---

## Pre-presentation checklist

- [ ] Slide 1 has both datasets clearly distinguished
- [ ] Slide 2's funnel is readable from the back of the room (font ≥ 18pt)
- [ ] You can pronounce "Reuters" and "WELFake" comfortably
- [ ] Memorized: 117k → 11,632 → 5,816 per class
- [ ] Memorized: ISOT = 2016-17 political; WELFake = 2020-23 broader
- [ ] You know what to do if asked "why didn't you use all 117k rows" (SVM training time)
