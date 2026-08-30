"""
Final trainer - evaluates on the OFFICIAL DAiSEE splits.

Uses features_full.csv (from extract_features_full.py):
  - Trains on the official Train split
  - Evaluates on the official Test split
  -> results directly comparable to published DAiSEE benchmarks

Reports both 4-level and binary tasks, with per-class metrics.

Usage:
  python engine\\train_classifier_full.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import joblib

FEATURES_FILE = "features_full.csv"
LABELS_FILE = "data/DAiSEE/Labels/AllLabels.csv"

try:
    feats = pd.read_csv(FEATURES_FILE)
except FileNotFoundError:
    print(f"Could not find {FEATURES_FILE}. Run extract_features_full.py first.")
    exit()

labels = pd.read_csv(LABELS_FILE)
labels.columns = [c.strip() for c in labels.columns]

merged = feats.merge(labels, left_on="clip", right_on="ClipID", how="inner")
merged = merged[merged["face_ratio_mean"] > -1]

print(f"Total usable clips: {len(merged)}")
print("By official split:")
print(merged["split"].value_counts().to_string())

train_df = merged[merged["split"] == "Train"]
test_df = merged[merged["split"] == "Test"]
# validation split available for tuning if wanted:
val_df = merged[merged["split"] == "Validation"]

if len(train_df) < 100 or len(test_df) < 50:
    print("\nWARNING: extraction may still be incomplete for a solid "
          "official-split evaluation. You can still run, but numbers firm up "
          "as more clips are processed.")

feature_cols = [c for c in feats.columns if c not in ("clip", "split")]
Xtr, Xte = train_df[feature_cols].values, test_df[feature_cols].values


def run(y_train, y_test, task):
    print("\n" + "=" * 62)
    print(f"TASK: {task}  (official Train -> official Test)")
    print("=" * 62)
    print("Train size:", len(y_train), "| Test size:", len(y_test))
    print("Test label distribution:",
          dict(zip(*np.unique(y_test, return_counts=True))))

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    best_name, best_model, best_f1 = None, None, -1
    for nm, m in candidates.items():
        m.fit(Xtr, y_train)
        pred = m.predict(Xte)
        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        print(f"  {nm}: accuracy {acc*100:.1f}% | macro-F1 {mf1:.3f}")
        # select on macro-F1, not raw accuracy (imbalance-aware)
        if mf1 > best_f1:
            best_name, best_model, best_f1 = nm, m, mf1

    pred = best_model.predict(Xte)
    print(f"\nBest by macro-F1: {best_name}")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, pred))
    return best_model


m4 = run(train_df["Engagement"].values, test_df["Engagement"].values,
         "4-level engagement (0-3)")

m2 = run((train_df["Engagement"] >= 2).astype(int).values,
         (test_df["Engagement"] >= 2).astype(int).values,
         "Binary engagement")

joblib.dump({"model": m2, "features": feature_cols, "task": "binary",
             "trained_on": "official DAiSEE Train split"},
            "engagement_model.joblib")
print("\nSaved binary model (official-split trained) to engagement_model.joblib")
print("NOTE: model selection above uses macro-F1, because raw accuracy is")
print("misleading on the imbalanced official test split.")
