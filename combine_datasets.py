"""
Build data/training_data.csv from the three raw source datasets.

Sources expected:
    data/News-_dataset/Fake.csv     (ISOT — fake political news, 2016-2017)
    data/News-_dataset/True.csv     (ISOT — real Reuters news, 2016-2017)
    data/WELFake_Dataset.csv        (WELFake — broader corpus including 2020-2023)

Output:
    data/training_data.csv  with two columns: text,label  (1=fake, 0=real)
    Class-balanced (50/50), 11,632 rows total by default, shuffled with seed=42.

Run:
    python combine_datasets.py
"""

import os
import sys
import pandas as pd

ISOT_FAKE = "data/News-_dataset/Fake.csv"
ISOT_TRUE = "data/News-_dataset/True.csv"
WELFAKE   = "data/WELFake_Dataset.csv"
OUTPUT    = "data/training_data.csv"

TARGET_TOTAL = 11632   # 5,816 per class — same size the original author shipped
RANDOM_SEED  = 42


def load_isot():
    """Load ISOT Fake.csv + True.csv, attach labels, return a (text, label) DataFrame."""
    if not (os.path.exists(ISOT_FAKE) and os.path.exists(ISOT_TRUE)):
        print(f"  Skipping ISOT — files not found at {ISOT_FAKE} / {ISOT_TRUE}")
        return pd.DataFrame(columns=["text", "label"])

    fake = pd.read_csv(ISOT_FAKE)[["text"]].copy()
    fake["label"] = 1
    true = pd.read_csv(ISOT_TRUE)[["text"]].copy()
    true["label"] = 0

    df = pd.concat([fake, true], ignore_index=True)
    print(f"  ISOT loaded: {len(fake):,} fake + {len(true):,} real = {len(df):,} rows")
    return df


def load_welfake():
    """Load WELFake, keep only text+label."""
    if not os.path.exists(WELFAKE):
        print(f"  Skipping WELFake — file not found at {WELFAKE}")
        return pd.DataFrame(columns=["text", "label"])

    df = pd.read_csv(WELFAKE, usecols=["text", "label"])
    fake_n = (df["label"] == 1).sum()
    real_n = (df["label"] == 0).sum()
    print(f"  WELFake loaded: {fake_n:,} fake + {real_n:,} real = {len(df):,} rows")
    return df


def main():
    print("=" * 60)
    print("BUILDING training_data.csv FROM RAW DATASETS")
    print("=" * 60)

    print("\nStep 1: Loading raw datasets...")
    isot = load_isot()
    welfake = load_welfake()

    if isot.empty and welfake.empty:
        print("\nERROR: No source datasets found. Place at least one of:")
        print(f"  - {ISOT_FAKE} + {ISOT_TRUE}")
        print(f"  - {WELFAKE}")
        sys.exit(1)

    print("\nStep 2: Concatenating sources...")
    combined = pd.concat([isot, welfake], ignore_index=True)
    print(f"  Combined: {len(combined):,} rows")

    print("\nStep 3: Cleaning (drop NaN/empty/duplicate text)...")
    before = len(combined)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.dropna(subset=["text", "label"])
    combined = combined.drop_duplicates(subset=["text"])
    combined["label"] = combined["label"].astype(int)
    print(f"  Removed {before - len(combined):,} empty/duplicate rows  →  {len(combined):,} remain")

    print("\nStep 4: Class-balancing 50/50...")
    fake_pool = combined[combined["label"] == 1]
    real_pool = combined[combined["label"] == 0]
    print(f"  Available: {len(fake_pool):,} fake / {len(real_pool):,} real")

    per_class = min(TARGET_TOTAL // 2, len(fake_pool), len(real_pool))
    fake_sample = fake_pool.sample(n=per_class, random_state=RANDOM_SEED)
    real_sample = real_pool.sample(n=per_class, random_state=RANDOM_SEED)
    print(f"  Sampled {per_class:,} fake + {per_class:,} real = {2 * per_class:,} rows")

    print("\nStep 5: Shuffling and writing output...")
    final = pd.concat([fake_sample, real_sample], ignore_index=True)
    final = final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    final = final[["text", "label"]]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    final.to_csv(OUTPUT, index=False)
    print(f"  Wrote {len(final):,} rows to {OUTPUT}")

    print("\nFinal label distribution:")
    print(final["label"].value_counts().to_string())
    print("\nDone. Next: python retrain_models.py")


if __name__ == "__main__":
    main()
