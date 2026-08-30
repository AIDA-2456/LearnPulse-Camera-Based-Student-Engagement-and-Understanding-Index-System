"""
Bias / fairness audit for the engagement classifier.

Two analyses:
  A) PER-SUBJECT fairness (always runs) - does the model perform equally well
     across different individuals? Uses the subject ID encoded in DAiSEE clip names.
  B) DEMOGRAPHIC fairness (runs if a demographics file is found) - performance
     broken down by gender or other recorded attributes.

Method: trains on the official Train split, evaluates on the official Test split
(identical protocol to Run 5), then disaggregates the results by group.

Usage:
  python engine\\bias_audit.py

Outputs:
  bias_audit_by_subject.csv
  bias_audit_subject_chart.png
  (+ bias_audit_by_demographic.csv if demographics available)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_FILE = "features_full.csv"
LABELS_FILE = "data/DAiSEE/Labels/AllLabels.csv"
MIN_CLIPS_PER_GROUP = 10   # ignore groups too small for a meaningful rate

# ---------- load ----------
feats = pd.read_csv(FEATURES_FILE)
labels = pd.read_csv(LABELS_FILE)
labels.columns = [c.strip() for c in labels.columns]

df = feats.merge(labels, left_on="clip", right_on="ClipID", how="inner")
df = df[df["face_ratio_mean"] > -1].copy()

# subject id: DAiSEE clip names are <userID><clipNo>; the trailing 4 digits are
# the clip number, so the prefix identifies the person. Approximate but consistent.
df["subject"] = df["clip"].str.replace(".avi", "", regex=False).str[:-4]

feature_cols = [c for c in feats.columns if c not in ("clip", "split")]
train = df[df["split"] == "Train"]
test = df[df["split"] == "Test"].copy()

print(f"Train clips: {len(train)} | Test clips: {len(test)} | "
      f"Subjects in test: {test['subject'].nunique()}")

# ---------- train (binary task, same protocol as Run 5) ----------
ytr = (train["Engagement"] >= 2).astype(int).values
yte = (test["Engagement"] >= 2).astype(int).values

model = GradientBoostingClassifier(random_state=42)
model.fit(train[feature_cols].values, ytr)
test["pred"] = model.predict(test[feature_cols].values)
test["true"] = yte

overall_acc = accuracy_score(yte, test["pred"])
overall_f1 = f1_score(yte, test["pred"], average="macro")
print(f"\nOverall on official test split: accuracy {overall_acc*100:.1f}% | "
      f"macro-F1 {overall_f1:.3f}")


def group_report(frame, group_col, title, outfile):
    rows = []
    for g, sub in frame.groupby(group_col):
        if len(sub) < MIN_CLIPS_PER_GROUP:
            continue
        acc = accuracy_score(sub["true"], sub["pred"])
        # equal-opportunity view: recall on the DISENGAGED class where present
        dis = sub[sub["true"] == 0]
        rec_dis = (recall_score(dis["true"], dis["pred"], pos_label=0,
                                zero_division=0) if len(dis) else np.nan)
        rows.append({
            group_col: g,
            "n_clips": len(sub),
            "accuracy": round(acc * 100, 1),
            "disengaged_clips": len(dis),
            "disengaged_recall": (round(rec_dis, 3) if not np.isnan(rec_dis) else ""),
            "predicted_engaged_rate": round(float(sub["pred"].mean()), 3),
        })

    if not rows:
        print(f"\n[{title}] no groups with >= {MIN_CLIPS_PER_GROUP} clips.")
        return None

    rep = pd.DataFrame(rows).sort_values("accuracy")
    rep.to_csv(outfile, index=False)

    accs = rep["accuracy"].values
    print("\n" + "=" * 62)
    print(f"{title}  ({len(rep)} groups with >= {MIN_CLIPS_PER_GROUP} clips)")
    print("=" * 62)
    print(f"Mean accuracy across groups : {accs.mean():.1f}%")
    print(f"Std deviation               : {accs.std():.1f} points")
    print(f"Best group                  : {accs.max():.1f}%")
    print(f"Worst group                 : {accs.min():.1f}%")
    print(f"DISPARITY (best - worst)    : {accs.max() - accs.min():.1f} points")
    par = rep["predicted_engaged_rate"].values
    print(f"Demographic-parity spread   : {(par.max()-par.min()):.3f} "
          f"(rate of predicting 'engaged')")
    print(f"\nFive worst-performing groups:")
    print(rep.head(5).to_string(index=False))
    print(f"\nSaved: {outfile}")
    return rep


# ---------- A) per-subject ----------
rep_subj = group_report(test, "subject", "PER-SUBJECT FAIRNESS",
                        "bias_audit_by_subject.csv")

if rep_subj is not None:
    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(rep_subj)), rep_subj["accuracy"], color="#3D6BD8")
    plt.axhline(overall_acc * 100, color="#B07A1E", linestyle="--",
                label=f"overall {overall_acc*100:.1f}%")
    plt.xlabel("Subject (sorted worst to best)")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-subject accuracy: fairness across individuals")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bias_audit_subject_chart.png", dpi=120)
    print("Saved: bias_audit_subject_chart.png")

# ---------- B) demographics, if available ----------
demo_file = None
for pattern in ("data/DAiSEE/Labels/*emograph*.csv", "data/DAiSEE/Labels/*ender*.csv",
                "data/DAiSEE/*emograph*.csv", "data/DAiSEE/**/*emograph*.csv"):
    hits = glob.glob(pattern, recursive=True)
    if hits:
        demo_file = hits[0]
        break

if demo_file:
    print(f"\nFound demographics file: {demo_file}")
    demo = pd.read_csv(demo_file)
    demo.columns = [c.strip() for c in demo.columns]
    key = next((c for c in demo.columns if "clip" in c.lower() or "id" in c.lower()), None)
    attr = next((c for c in demo.columns
                 if c.lower() in ("gender", "sex", "ethnicity", "race")), None)
    if key and attr:
        test2 = test.merge(demo, left_on="clip", right_on=key, how="left")
        if test2[attr].notna().sum() > 0:
            group_report(test2.dropna(subset=[attr]), attr,
                         f"DEMOGRAPHIC FAIRNESS ({attr})",
                         "bias_audit_by_demographic.csv")
        else:
            print("Demographics did not join to the test clips.")
    else:
        print(f"Could not identify id/attribute columns in {demo_file}: "
              f"{list(demo.columns)}")
else:
    print("\nNo demographics file found in the DAiSEE folder.")
    print("The per-subject audit above still provides a valid fairness analysis:")
    print("it measures whether the model performs consistently across individuals.")

print("\nInterpretation guide:")
print("  Disparity under ~10 points  -> reasonably consistent across groups")
print("  Disparity 10-25 points      -> notable variation, report and discuss")
print("  Disparity over ~25 points   -> substantial fairness concern")
