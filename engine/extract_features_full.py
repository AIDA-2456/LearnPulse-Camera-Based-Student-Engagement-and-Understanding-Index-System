"""
Resumable FULL-dataset feature extractor for DAiSEE.

Designed for the ~9,000-clip run (25-50 hours on CPU):
  - Walks the entire DAiSEE DataSet folder directly (no copying needed)
  - RESUMABLE: skips clips already in the output CSV - stop and restart freely
  - Appends each clip's features immediately (a crash loses nothing)
  - Records which official split (Train/Validation/Test) each clip belongs to
  - Prints progress with a time estimate

Usage:
  python engine\\extract_features_full.py
  (Ctrl+C anytime to stop; run again later to resume)

Output:
  features_full.csv
"""

import os
import csv
import time
import numpy as np
from ultralytics import YOLO
import cv2

DAISEE_ROOT = "data/DAiSEE/DataSet"
OUT_FILE = "features_full.csv"
SAMPLE_EVERY = 5

FIELDNAMES = [
    "clip", "split",
    "face_ratio_mean", "face_ratio_std", "facing_front_frac",
    "head_down_frac", "nose_conf_mean", "nose_x_std", "nose_y_std",
    "eye_conf_mean", "eye_nose_dist_mean", "eye_nose_dist_std",
    "nose_to_shoulder_mean", "nose_to_shoulder_std",
    "shoulder_width_mean", "shoulder_width_std",
    "move_mean", "move_max", "big_moves",
    "longest_still", "still_frac", "movement_drift",
]

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6

if not os.path.isdir(DAISEE_ROOT):
    print(f"ERROR: {DAISEE_ROOT} not found.")
    exit()

# ---- 1. Index every clip in the dataset with its official split ----
print("Indexing DAiSEE clips...")
all_clips = []   # (filename, full_path, split)
for split in ("Train", "Validation", "Test"):
    split_dir = os.path.join(DAISEE_ROOT, split)
    if not os.path.isdir(split_dir):
        continue
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(".avi"):
                all_clips.append((f, os.path.join(root, f), split))

print(f"Found {len(all_clips)} clips in the dataset.")

# ---- 2. Resume support: load names already processed ----
done = set()
if os.path.exists(OUT_FILE):
    with open(OUT_FILE, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["clip"])
    print(f"Resuming: {len(done)} clips already processed, "
          f"{len(all_clips) - len(done)} remaining.")
else:
    with open(OUT_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    print("Starting fresh.")

todo = [c for c in all_clips if c[0] not in done]
if not todo:
    print("All clips already processed. Nothing to do.")
    exit()

model = YOLO("yolov8n-pose.pt")


def extract_one(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    face_ratios, head_down_flags = [], []
    nose_confs, eye_confs = [], []
    nose_xs, nose_ys = [], []
    shoulder_widths, nose_to_shoulder = [], []
    eye_nose_dists = []

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % SAMPLE_EVERY != 0:
            frame_i += 1
            continue
        frame_i += 1

        results = model(frame, conf=0.4, verbose=False)
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            kpts = result.keypoints.data
            boxes = result.boxes.xyxy
            areas = [(float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
                     for b in boxes]
            p = int(np.argmax(areas))
            person = kpts[p]

            nose = person[NOSE]
            l_eye, r_eye = person[L_EYE], person[R_EYE]
            l_ear, r_ear = person[L_EAR], person[R_EAR]
            l_sh, r_sh = person[L_SHOULDER], person[R_SHOULDER]

            x1, y1, x2, y2 = (float(v) for v in boxes[p])
            box_h = max(y2 - y1, 1.0)

            nose_confs.append(float(nose[2]))
            eye_confs.append((float(l_eye[2]) + float(r_eye[2])) / 2)

            if nose[2] > 0.5:
                nose_xs.append(float(nose[0]) / box_h)
                nose_ys.append(float(nose[1]) / box_h)

            if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
                lx = float(min(l_ear[0], r_ear[0]))
                rx = float(max(l_ear[0], r_ear[0]))
                if rx > lx:
                    face_ratios.append((float(nose[0]) - lx) / (rx - lx))

            if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
                sh_y = (float(l_sh[1]) + float(r_sh[1])) / 2
                head_down_flags.append(1.0 if float(nose[1]) >= sh_y else 0.0)
                nose_to_shoulder.append((sh_y - float(nose[1])) / box_h)
                shoulder_widths.append(abs(float(r_sh[0]) - float(l_sh[0])) / box_h)

            if nose[2] > 0.5 and l_eye[2] > 0.3 and r_eye[2] > 0.3:
                emx = (float(l_eye[0]) + float(r_eye[0])) / 2
                emy = (float(l_eye[1]) + float(r_eye[1])) / 2
                eye_nose_dists.append(
                    float(np.hypot(float(nose[0]) - emx,
                                   float(nose[1]) - emy)) / box_h)

    cap.release()

    def m(x): return float(np.mean(x)) if x else -1.0
    def sd(x): return float(np.std(x)) if len(x) > 1 else -1.0

    if len(nose_xs) > 2:
        step = np.hypot(np.diff(nose_xs), np.diff(nose_ys))
        move_mean, move_max = float(np.mean(step)), float(np.max(step))
        big = int(np.sum(step > np.mean(step) + 2 * np.std(step))) if len(step) > 2 else 0
        th = np.median(step)
        longest = cur = 0
        for v in step:
            cur = cur + 1 if v <= th else 0
            longest = max(longest, cur)
        still_frac = float(np.mean(step <= th))
        half = len(step) // 2
        drift = float(np.mean(step[half:]) - np.mean(step[:half])) if half else 0.0
    else:
        move_mean = move_max = -1.0
        big, longest, still_frac, drift = -1, -1, -1.0, 0.0

    return {
        "face_ratio_mean": m(face_ratios), "face_ratio_std": sd(face_ratios),
        "facing_front_frac": (float(np.mean([1 if 0.3 < r < 0.7 else 0
                              for r in face_ratios])) if face_ratios else -1.0),
        "head_down_frac": m(head_down_flags),
        "nose_conf_mean": m(nose_confs),
        "nose_x_std": sd(nose_xs), "nose_y_std": sd(nose_ys),
        "eye_conf_mean": m(eye_confs),
        "eye_nose_dist_mean": m(eye_nose_dists),
        "eye_nose_dist_std": sd(eye_nose_dists),
        "nose_to_shoulder_mean": m(nose_to_shoulder),
        "nose_to_shoulder_std": sd(nose_to_shoulder),
        "shoulder_width_mean": m(shoulder_widths),
        "shoulder_width_std": sd(shoulder_widths),
        "move_mean": move_mean, "move_max": move_max, "big_moves": big,
        "longest_still": longest, "still_frac": still_frac,
        "movement_drift": drift,
    }


# ---- 3. Process, appending after every clip ----
print(f"Processing {len(todo)} clips. Press Ctrl+C anytime; progress is saved.\n")
start_time = time.time()
processed = 0

try:
    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for fname, path, split in todo:
            row = extract_one(path)
            processed += 1
            if row is None:
                print(f"  [skip] could not open {fname}")
                continue
            row["clip"] = fname
            row["split"] = split
            writer.writerow(row)
            f.flush()   # write to disk immediately - crash-safe

            if processed % 10 == 0:
                elapsed = time.time() - start_time
                per_clip = elapsed / processed
                remaining = (len(todo) - processed) * per_clip
                hrs, mins = int(remaining // 3600), int((remaining % 3600) // 60)
                print(f"  [{processed}/{len(todo)}] {fname}  "
                      f"({per_clip:.1f}s/clip, ~{hrs}h {mins}m remaining)")
except KeyboardInterrupt:
    print(f"\nStopped by user after {processed} clips this session.")
    print("Run the same command again later to resume where you left off.")
    exit()

total = len(done) + processed
print(f"\nDONE. {total} clips in {OUT_FILE}.")
print("Next: train on the official split with train_classifier_full.py")
