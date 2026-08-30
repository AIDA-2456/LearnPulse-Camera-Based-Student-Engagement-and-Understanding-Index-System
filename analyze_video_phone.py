"""
Hybrid classroom analyzer + PHONE DETECTION.

Adds to the hybrid analyzer:
  - A second YOLO pass detecting 'cell phone' (COCO class 67)
  - Each detected phone is associated with the nearest student
  - A student holding a phone is marked "Phone" and counted as disengaged
  - Class-level phone statistics are logged over time and saved

Ethical design: phone counts are reported at CLASS level in the outputs
(e.g. "3 phones visible"), never as a per-student record.

Usage:
  python engine\\analyze_video_phone.py data\\classroom_session.mp4
Outputs:
  engagement_<name>.png / engagement_log_<name>.csv  (now with phone_count column)
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

video_path = sys.argv[1] if len(sys.argv) >= 2 else "data/classroom_session.mp4"
if not os.path.exists(video_path):
    print(f"ERROR: file not found -> {video_path}")
    exit()
name = os.path.splitext(os.path.basename(video_path))[0]

MIN_BOX_FRAC = 0.18
MIN_FACE_CONF = 0.55
PHONE_CONF = 0.25          # phones are small; a lower threshold detects more
PHONE_CLASS = 67           # 'cell phone' in the COCO classes YOLO was trained on

USE_MODEL = False
try:
    import joblib
    bundle = joblib.load("engagement_model.joblib")
    clf = bundle["model"]
    FEATURE_ORDER = bundle["features"]
    USE_MODEL = True
    print("Loaded trained model.")
except Exception as e:
    print("No trained model - rules only.", e)

pose_model = YOLO("yolov8n-pose.pt")     # people + keypoints
object_model = YOLO("yolov8n.pt")        # general objects, incl. cell phone

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
LOWER_BODY = [11, 12, 13, 14, 15, 16]

WINDOW = int(fps * 5)
history = defaultdict(lambda: deque(maxlen=WINDOW))
phone_streak = defaultdict(int)   # consecutive frames a phone is seen with a student

times, engagement_log, phone_log = [], [], []
last_second, frame_idx = -1, 0
total_phone_frames = 0


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
         "nose_to_shoulder": None, "shoulder_width": None, "eye_nose_dist": None}
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
    def col(k): return [f[k] for f in hist if f[k] is not None]
    def m(x): return float(np.mean(x)) if x else -1.0
    def sd(x): return float(np.std(x)) if len(x) > 1 else -1.0
    ratios, xs, ys = col("face_ratio"), col("nose_x"), col("nose_y")
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
    return {"face_ratio_mean": m(ratios), "face_ratio_std": sd(ratios),
            "facing_front_frac": (float(np.mean([1 if 0.3 < r < 0.7 else 0 for r in ratios]))
                                  if ratios else -1.0),
            "head_down_frac": m(col("head_down")), "nose_conf_mean": m(col("nose_conf")),
            "nose_x_std": sd(xs), "nose_y_std": sd(ys), "eye_conf_mean": m(col("eye_conf")),
            "eye_nose_dist_mean": m(col("eye_nose_dist")),
            "eye_nose_dist_std": sd(col("eye_nose_dist")),
            "nose_to_shoulder_mean": m(col("nose_to_shoulder")),
            "nose_to_shoulder_std": sd(col("nose_to_shoulder")),
            "shoulder_width_mean": m(col("shoulder_width")),
            "shoulder_width_std": sd(col("shoulder_width")),
            "move_mean": move_mean, "move_max": move_max, "big_moves": big,
            "longest_still": longest, "still_frac": still_frac, "movement_drift": drift}


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # --- PHONE DETECTION PASS ---
    phone_boxes = []
    obj = object_model(frame, classes=[PHONE_CLASS], conf=PHONE_CONF, verbose=False)
    for r in obj:
        for b in r.boxes.xyxy:
            x1, y1, x2, y2 = (float(v) for v in b)
            phone_boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2))
    if phone_boxes:
        total_phone_frames += 1
    for (_, _, x1, y1, x2, y2) in phone_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

    # --- POSE / TRACKING PASS ---
    results = pose_model.track(frame, persist=True, conf=0.4, verbose=False)

    for result in results:
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue
        kpts, boxes = result.keypoints.data, result.boxes.xyxy
        ids = result.boxes.id
        n = len(kpts)

        teacher_idx, best = -1, -1
        for p in range(n):
            x1, y1, x2, y2 = (float(v) for v in boxes[p])
            area = (x2 - x1) * (y2 - y1)
            standing = sum(1 for k in LOWER_BODY if float(kpts[p][k][2]) > 0.3)
            sc = standing * 1_000_000 + area
            if sc > best:
                best, teacher_idx = sc, p

        engaged_n = disengaged_n = phones_with_students = 0

        for p in range(n):
            x1, y1, x2, y2 = (int(v) for v in boxes[p])
            tid = int(ids[p]) if ids is not None else -1

            if p == teacher_idx:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, "TEACHER", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
                continue

            # is a phone inside/near this student's box?
            has_phone = any(x1 - 20 <= cx <= x2 + 20 and y1 - 20 <= cy <= y2 + 20
                            for (cx, cy, *_ ) in phone_boxes)
            # require the phone to persist briefly - avoids single-frame false alarms
            phone_streak[tid] = phone_streak[tid] + 1 if has_phone else 0
            phone_confirmed = phone_streak[tid] >= max(2, int(fps * 0.4))

            sig = signals_from_pose(kpts[p], boxes[p])
            history[tid].append(sig)

            if phone_confirmed:
                # phone use overrides other signals: a strong disengagement indicator
                label, color = "Phone", (255, 0, 255)
                is_engaged = False
                phones_with_students += 1
            else:
                recent = list(history[tid])[-int(fps):]
                hd = [f["head_down"] for f in recent if f["head_down"] is not None]
                head_down_now = (np.mean(hd) > 0.6) if hd else False
                box_frac = (y2 - y1) / frame_h
                face_clear = (sig["nose_conf"] + sig["eye_conf"]) / 2 >= MIN_FACE_CONF
                applicable = (USE_MODEL and box_frac >= MIN_BOX_FRAC and face_clear
                              and not head_down_now and len(history[tid]) >= WINDOW // 2)
                if applicable:
                    feats = features_from_history(history[tid])
                    vec = np.array([[feats[k] for k in FEATURE_ORDER]])
                    is_engaged = bool(clf.predict(vec)[0] == 1)
                    label = "Engaged" if is_engaged else "Disengaged"
                else:
                    if sig["head_down"] == 1.0:
                        is_engaged, label = True, "Working"
                    elif sig["face_ratio"] is not None:
                        is_engaged = 0.3 < sig["face_ratio"] < 0.7
                        label = "Engaged" if is_engaged else "Distracted"
                    else:
                        is_engaged, label = True, "Working?"
                color = (0, 255, 0) if is_engaged else (0, 0, 255)

            engaged_n += 1 if is_engaged else 0
            disengaged_n += 0 if is_engaged else 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid} {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        students = engaged_n + disengaged_n
        pct = (engaged_n / students) * 100 if students else 0
        second = int(frame_idx / fps)
        if second != last_second:
            times.append(second)
            engagement_log.append(pct)
            phone_log.append(phones_with_students)
            last_second = second

        cv2.putText(frame, f"Engaged: {int(pct)}%  ({engaged_n}/{students})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Phones visible: {phones_with_students}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow(f"Analyzer + phone detection: {name} - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ---------- outputs (class level only) ----------
if times:
    with open(f"engagement_log_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_seconds", "engaged_percent", "phones_visible"])
        for t, e, ph in zip(times, engagement_log, phone_log):
            w.writerow([t, round(e, 1), ph])

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot([t / 60 for t in times], engagement_log, color="#1D9E75",
             linewidth=2, label="Engagement %")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Class engagement (%)", color="#1D9E75")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between([t / 60 for t in times], phone_log, color="#C43BC4",
                     alpha=0.25, label="Phones visible")
    ax2.set_ylabel("Phones visible", color="#C43BC4")
    ax2.set_ylim(0, max(max(phone_log), 1) + 1)

    plt.title(f"Engagement and phone use over time - {name}")
    fig.tight_layout()
    plt.savefig(f"engagement_{name}.png", dpi=120)

    peak = max(phone_log) if phone_log else 0
    frac = float(np.mean([1 if p > 0 else 0 for p in phone_log])) if phone_log else 0
    print(f"\nPhone summary (class level):")
    print(f"  Peak phones visible at once : {peak}")
    print(f"  Share of session with a phone visible: {frac*100:.1f}%")
    print(f"Saved: engagement_{name}.png and engagement_log_{name}.csv")
