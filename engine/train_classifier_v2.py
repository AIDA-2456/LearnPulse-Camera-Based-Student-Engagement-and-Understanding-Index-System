"""
Trainer v2 - evaluates BOTH the 4-level and binary engagement tasks.

Upgrades over v1:
  - Uses the richer features_v2.csv (~20 features)
  - Reports 4-level accuracy AND binary (engaged vs not-engaged) accuracy
  - Compares Random Forest vs Gradient Boosting, keeps the better one
  - Saves the best model for the analyzer

Usage:
  python engine\\train_classifier_v2.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

FEATURES_FILE = "features_v2.csv"
LABELS_FILE = "data/DAiSEE/Labels/AllLabels.csv"

try:
    feats = pd.read_csv(FEATURES_FILE)
except FileNotFoundError:
    print(f"Could not find {FEATURES_FILE}. Run extract_features_v2.py first.")
    exit()

try:
    labels = pd.read_csv(LABELS_FILE)
except FileNotFoundError:
    print(f"Could not find {LABELS_FILE}.")
    exit()

labels.columns = [c.strip() for c in labels.columns]
merged = feats.merge(labels, left_on="clip", right_on="ClipID", how="inner")
print(f"Feature rows: {len(feats)} | matched with labels: {len(merged)}")

merged = merged[merged["face_ratio_mean"] > -1]  # drop failed extractions
print(f"Usable rows: {len(merged)}")
if len(merged) < 30:
    print("WARNING: small sample; results will be noisy.")

feature_cols = [c for c in feats.columns if c != "clip"]
X = merged[feature_cols].values
y4 = merged["Engagement"].values                     # 4-level task
y2 = (merged["Engagement"] >= 2).astype(int).values  # binary: engaged or not

print("\n4-level label distribution:")
for lv in sorted(set(y4)):
    print(f"  level {lv}: {list(y4).count(lv)}")
print("Binary distribution: not-engaged(0):",
      list(y2).count(0), "| engaged(1):", list(y2).count(1))


def run_task(X, y, task_name):
    print("\n" + "=" * 60)
    print(f"TASK: {task_name}")
    print("=" * 60)
    strat = y if np.min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    best_name, best_model, best_acc = None, None, -1
    for name, m in candidates.items():
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        print(f"  {name}: {acc * 100:.1f}% accuracy")
        if acc > best_acc:
            best_name, best_model, best_acc = name, m, acc

    y_pred = best_model.predict(X_te)
    print(f"\nBest: {best_name} ({best_acc * 100:.1f}%)")
    print(classification_report(y_te, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))

    if hasattr(best_model, "feature_importances_"):
        print("Top 8 features:")
        pairs = sorted(zip(feature_cols, best_model.feature_importances_),
                       key=lambda t: -t[1])[:8]
        for n, v in pairs:
            print(f"  {n:24s} {v:.3f}")
    return best_model, best_acc


model4, acc4 = run_task(X, y4, "4-level engagement (0-3)")
model2, acc2 = run_task(X, y2, "Binary engagement (engaged vs not)")

joblib.dump({"model": model2, "features": feature_cols, "task": "binary"},
            "engagement_model.joblib")
print("\n" + "=" * 60)
print(f"SUMMARY:  4-level: {acc4*100:.1f}%   |   binary: {acc2*100:.1f}%")
print("Saved the binary model to engagement_model.joblib (for the analyzer).")
print("=" * 60)
