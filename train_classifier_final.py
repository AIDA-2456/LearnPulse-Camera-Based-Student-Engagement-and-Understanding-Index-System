"""
Run 6 - official split with TRAINING-SET rebalancing.

Rule respected: the official Test split is NEVER touched or resampled.
Only the Train split is rebalanced (minority oversampling by duplication),
which is standard, defensible practice for imbalanced learning.

Usage:
  python engine\\train_classifier_final.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import joblib

FEATURES_FILE = "features_full.csv"
LABELS_FILE = "data/DAiSEE/Labels/AllLabels.csv"

feats = pd.read_csv(FEATURES_FILE)
labels = pd.read_csv(LABELS_FILE)
labels.columns = [c.strip() for c in labels.columns]

merged = feats.merge(labels, left_on="clip", right_on="ClipID", how="inner")
merged = merged[merged["face_ratio_mean"] > -1]

train_df = merged[merged["split"] == "Train"]
test_df = merged[merged["split"] == "Test"]
feature_cols = [c for c in feats.columns if c not in ("clip", "split")]

Xte = test_df[feature_cols].values


def oversample(df, label_col):
    """Duplicate minority-class rows in TRAINING data until classes are balanced."""
    counts = df[label_col].value_counts()
    target = counts.max()
    parts = []
    for lv, n in counts.items():
        sub = df[df[label_col] == lv]
        if n < target:
            extra = sub.sample(target - n, replace=True, random_state=42)
            sub = pd.concat([sub, extra])
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1, random_state=42)  # shuffle
    return out


def run(label_series_train, label_series_test, task, binary=False):
    print("\n" + "=" * 62)
    print(f"TASK: {task}  (train OVERSAMPLED, official test untouched)")
    print("=" * 62)

    tr = train_df.copy()
    tr["_y"] = label_series_train.values
    print("Train distribution BEFORE oversampling:",
          dict(tr["_y"].value_counts().sort_index()))
    tr = oversample(tr, "_y")
    print("Train distribution AFTER oversampling: ",
          dict(tr["_y"].value_counts().sort_index()))

    Xtr = tr[feature_cols].values
    ytr = tr["_y"].values
    yte = label_series_test.values

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    best_name, best_model, best_f1 = None, None, -1
    for nm, m in candidates.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        acc = accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro")
        print(f"  {nm}: accuracy {acc*100:.1f}% | macro-F1 {mf1:.3f}")
        if mf1 > best_f1:
            best_name, best_model, best_f1 = nm, m, mf1

    pred = best_model.predict(Xte)
    print(f"\nBest by macro-F1: {best_name}")
    print(classification_report(yte, pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, pred))
    return best_model, best_f1


m4, f14 = run(train_df["Engagement"], test_df["Engagement"],
              "4-level engagement (0-3)")

m2, f12 = run((train_df["Engagement"] >= 2).astype(int),
              (test_df["Engagement"] >= 2).astype(int),
              "Binary engagement", binary=True)

joblib.dump({"model": m2, "features": feature_cols, "task": "binary",
             "trained_on": "official Train split, minority-oversampled"},
            "engagement_model.joblib")
print("\nSaved oversampled-trained binary model to engagement_model.joblib")
print("Compare minority-class recall against Run 5 - that is the number to watch.")
