"""
Feature extractor v2 - richer features for higher accuracy.

Upgrades over v1:
  - Eye keypoints (visibility, eye-nose geometry)
  - Shoulder posture (width, nose-to-shoulder distance = slump/upright)
  - Temporal features (movement over time, stillness periods, big-move counts)
  - ~20 features per clip instead of 7

Usage:
  python engine\\extract_features_v2.py data\\daisee_clips
Output:
  features_v2.csv
"""

import sys
import os
import csv
import numpy as np
from ultralytics import YOLO
import cv2

if len(sys.argv) >= 2:
    clips_folder = sys.argv[1]
else:
    clips_folder = "data/daisee_clips"
    print("No folder given, using default:", clips_folder)

if not os.path.isdir(clips_folder):
    print(f"ERROR: folder not found -> {clips_folder}")
    exit()

model = YOLO("yolov8n-pose.pt")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6

SAMPLE_EVERY = 5

video_files = [f for f in os.listdir(clips_folder)
               if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))]
if not video_files:
    print("No video files found in that folder.")
    exit()

print(f"Found {len(video_files)} clips. Extracting v2 features...")

rows = []
for i, fname in enumerate(video_files, 1):
    path = os.path.join(clips_folder, fname)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [skip] could not open {fname}")
        continue

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

            # box height to normalise distances (person size invariance)
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
                # posture: vertical distance nose -> shoulders, normalised
                nose_to_shoulder.append((sh_y - float(nose[1])) / box_h)
                shoulder_widths.append(abs(float(r_sh[0]) - float(l_sh[0])) / box_h)

            if nose[2] > 0.5 and l_eye[2] > 0.3 and r_eye[2] > 0.3:
                eye_mid_x = (float(l_eye[0]) + float(r_eye[0])) / 2
                eye_mid_y = (float(l_eye[1]) + float(r_eye[1])) / 2
                d = np.hypot(float(nose[0]) - eye_mid_x,
                             float(nose[1]) - eye_mid_y) / box_h
                eye_nose_dists.append(d)

    cap.release()

    def s_mean(x): return float(np.mean(x)) if x else -1.0
    def s_std(x):  return float(np.std(x)) if len(x) > 1 else -1.0

    # ---- temporal features from the nose trajectory ----
    if len(nose_xs) > 2:
        dx = np.diff(nose_xs)
        dy = np.diff(nose_ys)
        step = np.hypot(dx, dy)                  # movement per sampled frame
        move_mean = float(np.mean(step))
        move_max = float(np.max(step))
        big_moves = int(np.sum(step > (np.mean(step) + 2 * np.std(step)))) \
            if len(step) > 2 else 0
        # longest run of near-stillness (below median movement)
        thresh = np.median(step)
        longest_still, cur = 0, 0
        for v in step:
            cur = cur + 1 if v <= thresh else 0
            longest_still = max(longest_still, cur)
        still_frac = float(np.mean(step <= thresh))
        # first half vs second half movement (drift over the clip)
        half = len(step) // 2
        drift = (float(np.mean(step[half:]) - np.mean(step[:half]))
                 if half > 0 else 0.0)
    else:
        move_mean = move_max = -1.0
        big_moves = -1
        longest_still = -1
        still_frac = -1.0
        drift = 0.0

    row = {
        "clip": fname,
        # v1 features (kept for comparison)
        "face_ratio_mean": s_mean(face_ratios),
        "face_ratio_std": s_std(face_ratios),
        "facing_front_frac": (float(np.mean([1 if 0.3 < r < 0.7 else 0
                              for r in face_ratios])) if face_ratios else -1.0),
        "head_down_frac": s_mean(head_down_flags),
        "nose_conf_mean": s_mean(nose_confs),
        "nose_x_std": s_std(nose_xs),
        "nose_y_std": s_std(nose_ys),
        # NEW: eyes
        "eye_conf_mean": s_mean(eye_confs),
        "eye_nose_dist_mean": s_mean(eye_nose_dists),
        "eye_nose_dist_std": s_std(eye_nose_dists),
        # NEW: posture
        "nose_to_shoulder_mean": s_mean(nose_to_shoulder),
        "nose_to_shoulder_std": s_std(nose_to_shoulder),
        "shoulder_width_mean": s_mean(shoulder_widths),
        "shoulder_width_std": s_std(shoulder_widths),
        # NEW: temporal movement
        "move_mean": move_mean,
        "move_max": move_max,
        "big_moves": big_moves,
        "longest_still": longest_still,
        "still_frac": still_frac,
        "movement_drift": drift,
    }
    rows.append(row)
    print(f"  [{i}/{len(video_files)}] {fname} done")

if not rows:
    print("No features extracted.")
    exit()

out = "features_v2.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} feature rows ({len(rows[0]) - 1} features each) to {out}")
print("Next: python engine\\train_classifier_v2.py")
