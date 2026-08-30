"""
Targeted sampler - hunts down LOW-engagement clips to balance your training data.

What it does:
  1. Reads DAiSEE's labels and finds every clip with Engagement 0 or 1
  2. Searches your DAiSEE folder for those exact files
  3. Copies ALL of them into data\daisee_clips
  4. Also tops up engagement 2/3 clips to a target balance

Usage:
  python engine\\sample_balanced_clips.py
"""

import os
import shutil
import pandas as pd

DAISEE_ROOT = "data/DAiSEE"
LABELS_FILE = os.path.join(DAISEE_ROOT, "Labels", "AllLabels.csv")
DEST = "data/daisee_clips"

TARGET_HIGH = 150   # how many level-2 and level-3 clips to keep (each)

labels = pd.read_csv(LABELS_FILE)
labels.columns = [c.strip() for c in labels.columns]

low = labels[labels["Engagement"] <= 1]["ClipID"].tolist()
lvl2 = labels[labels["Engagement"] == 2]["ClipID"].tolist()[:TARGET_HIGH]
lvl3 = labels[labels["Engagement"] == 3]["ClipID"].tolist()[:TARGET_HIGH]

wanted = set(low + lvl2 + lvl3)
print(f"Low-engagement clips in labels (level 0/1): {len(low)}")
print(f"Target total wanted: {len(wanted)}")

os.makedirs(DEST, exist_ok=True)
already = set(os.listdir(DEST))

# index every avi in the dataset once (fast lookup)
print("Indexing DAiSEE files (this takes a minute)...")
index = {}
for root, _, files in os.walk(os.path.join(DAISEE_ROOT, "DataSet")):
    for f in files:
        if f.lower().endswith(".avi"):
            index[f] = os.path.join(root, f)

copied = skipped = missing = 0
for clip in wanted:
    if clip in already:
        skipped += 1
        continue
    if clip in index:
        shutil.copy2(index[clip], os.path.join(DEST, clip))
        copied += 1
    else:
        missing += 1

print(f"\nCopied: {copied} | already had: {skipped} | not found on disk: {missing}")

# show the resulting balance of everything now in the folder
have = set(os.listdir(DEST))
subset = labels[labels["ClipID"].isin(have)]
print("\nYour training folder now contains:")
for lv in sorted(subset["Engagement"].unique()):
    print(f"  Engagement level {lv}: {len(subset[subset['Engagement'] == lv])} clips")
print(f"  TOTAL: {len(subset)} clips")
print("\nNext: python engine\\extract_features_v2.py data\\daisee_clips")
