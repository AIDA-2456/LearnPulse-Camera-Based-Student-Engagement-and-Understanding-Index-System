"""
Feature extractor for DAiSEE clips.

What it does:
  - Walks through a folder of video clips (DAiSEE .avi/.mp4 files)
  - Runs each clip through the YOLO pose pipeline (same signals as analyze_video.py)
  - Averages the signals over the clip into ONE feature row per video
  - Saves everything to features.csv, ready for classifier training

Usage:
  python engine\\extract_features.py data\\daisee_clips
  (point it at any folder that contains video files)

Note: DAiSEE clips show ONE student each, so we take the single largest person.
"""

import sys
import os
import csv
import numpy as np
from ultralytics import YOLO
import cv2

# ---- where are the clips? ----
if len(sys.argv) >= 2:
    clips_folder = sys.argv[1]
else:
    clips_folder = "data/daisee_clips"
    print("No folder given, using default:", clips_folder)
    print("Tip: python engine\\extract_features.py data\\your_clip_folder")

if not os.path.isdir(clips_folder):
    print(f"ERROR: folder not found -> {clips_folder}")
    exit()

model = YOLO("yolov8n-pose.pt")

NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6

SAMPLE_EVERY = 5   # analyse every 5th frame (DAiSEE clips are 10s @ 30fps; this is plenty)

video_files = [f for f in os.listdir(clips_folder)
               if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))]

if not video_files:
    print("No video files found in that folder.")
    exit()

print(f"Found {len(video_files)} clips. Extracting features...")

rows = []
for i, fname in enumerate(video_files, 1):
    path = os.path.join(clips_folder, fname)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [skip] could not open {fname}")
        continue

    # per-frame signal collectors
    face_ratios = []      # nose position between ears (0.5 = facing front)
    head_down_flags = []  # 1 if nose at/below shoulders
    nose_confs = []       # how visible the face was
    nose_xs, nose_ys = [], []  # for movement features

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

            # take the LARGEST person (DAiSEE has one subject per clip)
            areas = [(float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
                     for b in boxes]
            p = int(np.argmax(areas))
            person = kpts[p]

            nose = person[NOSE]
            l_ear, r_ear = person[L_EAR], person[R_EAR]
            l_sh, r_sh = person[L_SHOULDER], person[R_SHOULDER]

            nose_confs.append(float(nose[2]))

            if nose[2] > 0.5:
                nose_xs.append(float(nose[0]))
                nose_ys.append(float(nose[1]))

            if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
                lx = float(min(l_ear[0], r_ear[0]))
                rx = float(max(l_ear[0], r_ear[0]))
                if rx > lx:
                    face_ratios.append((float(nose[0]) - lx) / (rx - lx))

            if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
                sh_y = (float(l_sh[1]) + float(r_sh[1])) / 2
                head_down_flags.append(1.0 if float(nose[1]) >= sh_y else 0.0)

    cap.release()

    # ---- summarise the whole clip into ONE feature row ----
    def safe_mean(x): return float(np.mean(x)) if x else -1.0
    def safe_std(x):  return float(np.std(x)) if len(x) > 1 else -1.0

    row = {
        "clip": fname,
        # head direction: mean + how much it wandered
        "face_ratio_mean": safe_mean(face_ratios),
        "face_ratio_std": safe_std(face_ratios),
        # fraction of time roughly facing front
        "facing_front_frac": (float(np.mean([1 if 0.3 < r < 0.7 else 0
                              for r in face_ratios])) if face_ratios else -1.0),
        # head-down time fraction
        "head_down_frac": safe_mean(head_down_flags),
        # face visibility
        "nose_conf_mean": safe_mean(nose_confs),
        # head movement (fidgeting vs still)
        "nose_x_std": safe_std(nose_xs),
        "nose_y_std": safe_std(nose_ys),
    }
    rows.append(row)
    print(f"  [{i}/{len(video_files)}] {fname} done")

# ---- save ----
if not rows:
    print("No features extracted (all clips failed to open?)")
    exit()

out = "features.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} feature rows to {out}")
print("Next: join these with the DAiSEE labels file, then train the classifier.")
