"""
Train and evaluate an engagement classifier on DAiSEE features.

What it does:
  1. Loads features.csv (from extract_features.py)
  2. Joins each clip to its true Engagement label (AllLabels.csv)
  3. Trains a Random Forest classifier
  4. Evaluates on held-out clips: accuracy, precision, recall, F1
  5. Saves the trained model (engagement_model.joblib) for use in the analyzer

Usage:
  python engine\\train_classifier.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import joblib

FEATURES_FILE = "features.csv"
LABELS_FILE = "data/DAiSEE/Labels/AllLabels.csv"

# ---- 1. Load features and labels ----
try:
    feats = pd.read_csv(FEATURES_FILE)
except FileNotFoundError:
    print(f"Could not find {FEATURES_FILE}. Run extract_features.py first.")
    exit()

try:
    labels = pd.read_csv(LABELS_FILE)
except FileNotFoundError:
    print(f"Could not find {LABELS_FILE}. Check the path to AllLabels.csv.")
    exit()

# tidy column names (the labels file has a trailing space in the header)
labels.columns = [c.strip() for c in labels.columns]

# ---- 2. Join on clip filename ----
merged = feats.merge(labels, left_on="clip", right_on="ClipID", how="inner")
print(f"Feature rows: {len(feats)} | matched with labels: {len(merged)}")

if len(merged) < 20:
    print("WARNING: very few matched clips. Results will be unreliable.")
    print("(Copy more clips into data\\daisee_clips and re-run extract_features.py)")
if len(merged) < 5:
    exit()

# drop clips where pose extraction found nothing usable
merged = merged[merged["face_ratio_mean"] >= 0]
print(f"Usable rows after removing failed extractions: {len(merged)}")

# ---- 3. Prepare training data ----
feature_cols = ["face_ratio_mean", "face_ratio_std", "facing_front_frac",
                "head_down_frac", "nose_conf_mean", "nose_x_std", "nose_y_std"]
X = merged[feature_cols].values
y = merged["Engagement"].values

print("\nEngagement label distribution in your sample:")
for level in sorted(set(y)):
    print(f"  level {level}: {list(y).count(level)} clips")

# hold out 25% of clips for testing (the model never sees them in training)
strat = y if min(np.bincount(y)) >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=strat)

# ---- 4. Train ----
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # compensate for DAiSEE's label imbalance
    random_state=42)
model.fit(X_train, y_train)

# ---- 5. Evaluate on the unseen test clips ----
y_pred = model.predict(X_test)

print("\n" + "=" * 55)
print("EVALUATION ON UNSEEN CLIPS")
print("=" * 55)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print("\nPer-class precision / recall / F1:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

# which features mattered most?
print("\nFeature importance (what the model relied on):")
for name, imp in sorted(zip(feature_cols, model.feature_importances_),
                        key=lambda t: -t[1]):
    print(f"  {name:20s} {imp:.3f}")

# ---- 6. Save the trained model ----
joblib.dump({"model": model, "features": feature_cols}, "engagement_model.joblib")
print("\nSaved trained model to engagement_model.joblib")
print("This model can now replace the hand-written rules in the analyzer.")
