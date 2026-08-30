"""
Hybrid classroom analyzer - trained model + context-aware rules.

The problem it solves (observed in testing):
  The DAiSEE-trained model misreads distant, head-down desk-working students
  as "disengaged" because it learned close-up webcam engagement cues.

The hybrid strategy:
  - If a student is CLOSE/CLEAR enough (box big, face keypoints confident):
      -> use the TRAINED MODEL (its domain applies)
  - Otherwise (small/distant/head-down students):
      -> use CLASSROOM RULES where head-down = "Working" (engaged),
         facing front = engaged, only turned-away = disengaged.

Labels show which brain decided: [M] = model, [R] = rules.

Usage:
  python engine\\analyze_video_hybrid.py data\\classroom_session.mp4
"""

import sys
import os
import csv
from collections import defaultdict, deque
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- video path ----
if len(sys.argv) >= 2:
    video_path = sys.argv[1]
else:
    video_path = "data/classroom_session.mp4"
    print("No video given, using default:", video_path)

if not os.path.exists(video_path):
    print(f"ERROR: file not found -> {video_path}")
    exit()

name = os.path.splitext(os.path.basename(video_path))[0]

# ---- when is a student "close/clear enough" for the model? ----
MIN_BOX_FRAC = 0.18   # box height must be at least this fraction of frame height
MIN_FACE_CONF = 0.55  # average nose+eye confidence must be at least this

# ---- load trained model ----
USE_MODEL = False
try:
    import joblib
    bundle = joblib.load("engagement_model.joblib")
    clf = bundle["model"]
    FEATURE_ORDER = bundle["features"]
    USE_MODEL = True
    print("Loaded trained model (engagement_model.joblib)")
except Exception as e:
    print("No trained model - rules only.", e)

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Could not open the video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
LOWER_BODY = [11, 12, 13, 14, 15, 16]

WINDOW = int(fps * 5)
history = defaultdict(lambda: deque(maxlen=WINDOW))

times, engagement_log = [], []
model_used_count = rules_used_count = 0
last_second, frame_idx = -1, 0


def signals_from_pose(person, box):
    x1, y1, x2, y2 = (float(v) for v in box)
    box_h = max(y2 - y1, 1.0)
    nose = person[NOSE]
    l_eye, r_eye = person[L_EYE], person[R_EYE]
    l_ear, r_ear = person[L_EAR], person[R_EAR]
    l_sh, r_sh = person[L_SHOULDER], person[R_SHOULDER]

    s = {"nose_conf": float(nose[2]),
         "eye_conf": (float(l_eye[2]) + float(r_eye[2])) / 2,
         "nose_x": float(nose[0]) / box_h if nose[2] > 0.5 else None,
         "nose_y": float(nose[1]) / box_h if nose[2] > 0.5 else None,
         "face_ratio": None, "head_down": None,
         "nose_to_shoulder": None, "shoulder_width": None,
         "eye_nose_dist": None}

    if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
        lx, rx = float(min(l_ear[0], r_ear[0])), float(max(l_ear[0], r_ear[0]))
        if rx > lx:
            s["face_ratio"] = (float(nose[0]) - lx) / (rx - lx)

    if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
        sh_y = (float(l_sh[1]) + float(r_sh[1])) / 2
        s["head_down"] = 1.0 if float(nose[1]) >= sh_y else 0.0
        s["nose_to_shoulder"] = (sh_y - float(nose[1])) / box_h
        s["shoulder_width"] = abs(float(r_sh[0]) - float(l_sh[0])) / box_h

    if nose[2] > 0.5 and l_eye[2] > 0.3 and r_eye[2] > 0.3:
        emx = (float(l_eye[0]) + float(r_eye[0])) / 2
        emy = (float(l_eye[1]) + float(r_eye[1])) / 2
        s["eye_nose_dist"] = float(np.hypot(float(nose[0]) - emx,
                                            float(nose[1]) - emy)) / box_h
    return s


def features_from_history(hist):
    def col(key):
        return [f[key] for f in hist if f[key] is not None]

    def m(x): return float(np.mean(x)) if x else -1.0
    def sd(x): return float(np.std(x)) if len(x) > 1 else -1.0

    ratios = col("face_ratio")
    xs, ys = col("nose_x"), col("nose_y")

    if len(xs) > 2:
        step = np.hypot(np.diff(xs), np.diff(ys))
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
        "face_ratio_mean": m(ratios), "face_ratio_std": sd(ratios),
        "facing_front_frac": (float(np.mean([1 if 0.3 < r < 0.7 else 0
                              for r in ratios])) if ratios else -1.0),
        "head_down_frac": m(col("head_down")),
        "nose_conf_mean": m(col("nose_conf")),
        "nose_x_std": sd(xs), "nose_y_std": sd(ys),
        "eye_conf_mean": m(col("eye_conf")),
        "eye_nose_dist_mean": m(col("eye_nose_dist")),
        "eye_nose_dist_std": sd(col("eye_nose_dist")),
        "nose_to_shoulder_mean": m(col("nose_to_shoulder")),
        "nose_to_shoulder_std": sd(col("nose_to_shoulder")),
        "shoulder_width_mean": m(col("shoulder_width")),
        "shoulder_width_std": sd(col("shoulder_width")),
        "move_mean": move_mean, "move_max": move_max, "big_moves": big,
        "longest_still": longest, "still_frac": still_frac,
        "movement_drift": drift,
    }


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(frame, persist=True, conf=0.4, verbose=False)

    for result in results:
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue
        kpts = result.keypoints.data
        boxes = result.boxes.xyxy
        ids = result.boxes.id
        n = len(kpts)

        teacher_idx, best = -1, -1
        for p in range(n):
            x1, y1, x2, y2 = (float(v) for v in boxes[p])
            area = (x2 - x1) * (y2 - y1)
            standing = sum(1 for k in LOWER_BODY if float(kpts[p][k][2]) > 0.3)
            score = standing * 1_000_000 + area
            if score > best:
                best, teacher_idx = score, p

        engaged_n = disengaged_n = 0

        for p in range(n):
            x1, y1, x2, y2 = (int(v) for v in boxes[p])
            tid = int(ids[p]) if ids is not None else -1

            if p == teacher_idx:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"TEACHER #{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                continue

            sig = signals_from_pose(kpts[p], boxes[p])
            history[tid].append(sig)

            # ---- HYBRID DECISION ----
            box_frac = (y2 - y1) / frame_h
            face_clear = (sig["nose_conf"] + sig["eye_conf"]) / 2 >= MIN_FACE_CONF
            model_applicable = (USE_MODEL and box_frac >= MIN_BOX_FRAC
                                and face_clear
                                and len(history[tid]) >= WINDOW // 2)

            if model_applicable:
                feats = features_from_history(history[tid])
                vec = np.array([[feats[k] for k in FEATURE_ORDER]])
                is_engaged = bool(clf.predict(vec)[0] == 1)
                src = "M"
                model_used_count += 1
                label = "Engaged" if is_engaged else "Disengaged"
            else:
                # CLASSROOM RULES: head-down at a desk = working = engaged
                src = "R"
                rules_used_count += 1
                if sig["head_down"] == 1.0:
                    is_engaged = True
                    label = "Working"
                elif sig["face_ratio"] is not None and 0.3 < sig["face_ratio"] < 0.7:
                    is_engaged = True
                    label = "Engaged"
                elif sig["face_ratio"] is not None:
                    is_engaged = False
                    label = "Distracted"
                else:
                    # can't see the face at all: most desk setups mean working
                    is_engaged = True
                    label = "Working?"

            if is_engaged:
                color = (0, 255, 0)
                engaged_n += 1
            else:
                color = (0, 0, 255)
                disengaged_n += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid}[{src}] {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        students = engaged_n + disengaged_n
        pct = (engaged_n / students) * 100 if students else 0
        second = int(frame_idx / fps)
        if second != last_second:
            times.append(second)
            engagement_log.append(pct)
            last_second = second

        cv2.putText(frame, f"[HYBRID] Engaged: {int(pct)}%  ({engaged_n}/{students})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(f"Hybrid Analyzer: {name} - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

total_dec = model_used_count + rules_used_count
if total_dec:
    print(f"\nDecision sources: model {model_used_count} "
          f"({100*model_used_count/total_dec:.0f}%) | rules {rules_used_count} "
          f"({100*rules_used_count/total_dec:.0f}%)")

if times:
    with open(f"engagement_log_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_seconds", "engaged_percent"])
        for t, e in zip(times, engagement_log):
            w.writerow([t, round(e, 1)])

    plt.figure(figsize=(10, 4))
    plt.plot([t / 60 for t in times], engagement_log, color="#1D9E75", linewidth=2)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Class engagement (%)")
    plt.title(f"Engagement over time (hybrid) - {name}")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"engagement_{name}.png", dpi=120)
    print(f"Saved: engagement_{name}.png and engagement_log_{name}.csv")
