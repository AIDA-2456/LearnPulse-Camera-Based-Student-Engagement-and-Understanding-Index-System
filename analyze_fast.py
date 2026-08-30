"""
LearnPulse - fast analyser (optimised for producing the engagement log).

Speed improvements over analyze_video_phone.py:
  - Phone detection runs every 5th frame and is cached in between
  - Optional --headless mode skips the video window (much faster)
  - Prints live progress so it is obvious the program is working
  - Saves progress even if interrupted, so Ctrl+C never loses the run

Usage:
  python engine\\analyze_fast.py data\\classroom_session.mp4
  python engine\\analyze_fast.py data\\classroom_session.mp4 --headless
  python engine\\analyze_fast.py data\\classroom_session.mp4 --headless --skip 3

Output:
  engagement_log_<name>.csv   (feeds generate_report.py)
"""

import sys
import os
import csv
import time
from collections import defaultdict, deque

import numpy as np
import cv2
from ultralytics import YOLO

args = sys.argv[1:]
video_path = next((a for a in args if not a.startswith("--")), "data/classroom_session.mp4")
HEADLESS = "--headless" in args
SKIP = 2
if "--skip" in args:
    i = args.index("--skip")
    if i + 1 < len(args):
        SKIP = max(1, int(args[i + 1]))

if not os.path.exists(video_path):
    print(f"ERROR: file not found -> {video_path}")
    sys.exit()
name = os.path.splitext(os.path.basename(video_path))[0]

PHONE_EVERY = 5          # run the phone detector only every Nth processed frame
PHONE_CONF = 0.25
PHONE_CLASS = 67
MIN_BOX_FRAC = 0.18
MIN_FACE_CONF = 0.55

print("Loading models... (first run may take 20-30 seconds)")
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

USE_MODEL = False
try:
    import joblib
    bundle = joblib.load("engagement_model.joblib")
    clf, FEATURE_ORDER = bundle["model"], bundle["features"]
    USE_MODEL = True
    print("Trained engagement model loaded.")
except Exception as e:
    print("No trained model found - using rules only.", e)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: could not open the video.")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
print(f"Video: {name} | {fps:.0f} fps | {total_frames} frames "
      f"| analysing every {SKIP} frame(s)"
      + (" | HEADLESS" if HEADLESS else " | press Q in the window to stop"))
print("Working... do NOT press any keys in this terminal.\n")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
LOWER_BODY = [11, 12, 13, 14, 15, 16]

WINDOW = max(4, int(fps * 5 / SKIP))
history = defaultdict(lambda: deque(maxlen=WINDOW))
phone_streak = defaultdict(int)

times, engagement_log, phone_log = [], [], []
last_second, frame_idx, processed = -1, 0, 0
cached_phones = []
start_time = time.time()


def signals(person, box):
    x1, y1, x2, y2 = (float(v) for v in box)
    bh = max(y2 - y1, 1.0)
    nose, l_eye, r_eye = person[NOSE], person[L_EYE], person[R_EYE]
    l_ear, r_ear = person[L_EAR], person[R_EAR]
    l_sh, r_sh = person[L_SHOULDER], person[R_SHOULDER]
    s = {"nose_conf": float(nose[2]),
         "eye_conf": (float(l_eye[2]) + float(r_eye[2])) / 2,
         "nose_x": float(nose[0]) / bh if nose[2] > 0.5 else None,
         "nose_y": float(nose[1]) / bh if nose[2] > 0.5 else None,
         "face_ratio": None, "head_down": None,
         "nose_to_shoulder": None, "shoulder_width": None, "eye_nose_dist": None}
    if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
        lx, rx = float(min(l_ear[0], r_ear[0])), float(max(l_ear[0], r_ear[0]))
        if rx > lx:
            s["face_ratio"] = (float(nose[0]) - lx) / (rx - lx)
    if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
        shy = (float(l_sh[1]) + float(r_sh[1])) / 2
        s["head_down"] = 1.0 if float(nose[1]) >= shy else 0.0
        s["nose_to_shoulder"] = (shy - float(nose[1])) / bh
        s["shoulder_width"] = abs(float(r_sh[0]) - float(l_sh[0])) / bh
    if nose[2] > 0.5 and l_eye[2] > 0.3 and r_eye[2] > 0.3:
        emx = (float(l_eye[0]) + float(r_eye[0])) / 2
        emy = (float(l_eye[1]) + float(r_eye[1])) / 2
        s["eye_nose_dist"] = float(np.hypot(float(nose[0]) - emx,
                                            float(nose[1]) - emy)) / bh
    return s


def features(hist):
    def col(k): return [f[k] for f in hist if f[k] is not None]
    def m(x): return float(np.mean(x)) if x else -1.0
    def sd(x): return float(np.std(x)) if len(x) > 1 else -1.0
    ratios, xs, ys = col("face_ratio"), col("nose_x"), col("nose_y")
    if len(xs) > 2:
        step = np.hypot(np.diff(xs), np.diff(ys))
        mv, mx = float(np.mean(step)), float(np.max(step))
        big = int(np.sum(step > np.mean(step) + 2 * np.std(step))) if len(step) > 2 else 0
        th = np.median(step)
        longest = cur = 0
        for v in step:
            cur = cur + 1 if v <= th else 0
            longest = max(longest, cur)
        stf = float(np.mean(step <= th))
        half = len(step) // 2
        drift = float(np.mean(step[half:]) - np.mean(step[:half])) if half else 0.0
    else:
        mv = mx = -1.0
        big, longest, stf, drift = -1, -1, -1.0, 0.0
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
            "move_mean": mv, "move_max": mx, "big_moves": big,
            "longest_still": longest, "still_frac": stf, "movement_drift": drift}


def save():
    if not times:
        print("No data logged - no students detected.")
        return
    out = f"engagement_log_{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_seconds", "engaged_percent", "phones_visible"])
        for t, e, p in zip(times, engagement_log, phone_log):
            w.writerow([t, round(e, 1), p])
    print(f"\nSaved: {out}  ({len(times)} seconds logged)")
    print(f"Next:  python engine\\generate_report.py {out}")


try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % SKIP != 0:
            frame_idx += 1
            continue

        # --- phones: only every PHONE_EVERY processed frames, cached between ---
        if processed % PHONE_EVERY == 0:
            cached_phones = []
            for r in object_model(frame, classes=[PHONE_CLASS],
                                  conf=PHONE_CONF, verbose=False):
                for b in r.boxes.xyxy:
                    x1, y1, x2, y2 = (float(v) for v in b)
                    cached_phones.append(((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2))

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
                sc = (sum(1 for k in LOWER_BODY if float(kpts[p][k][2]) > 0.3) * 1_000_000
                      + (x2 - x1) * (y2 - y1))
                if sc > best:
                    best, teacher_idx = sc, p

            eng_n = dis_n = phones_now = 0
            for p in range(n):
                if p == teacher_idx:
                    if not HEADLESS:
                        x1, y1, x2, y2 = (int(v) for v in boxes[p])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                        cv2.putText(frame, "TEACHER", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
                    continue

                x1, y1, x2, y2 = (int(v) for v in boxes[p])
                tid = int(ids[p]) if ids is not None else -1
                sig = signals(kpts[p], boxes[p])
                history[tid].append(sig)

                has_phone = any(x1 - 20 <= cx <= x2 + 20 and y1 - 20 <= cy <= y2 + 20
                                for (cx, cy, *_) in cached_phones)
                phone_streak[tid] = phone_streak[tid] + 1 if has_phone else 0
                phone_confirmed = phone_streak[tid] >= 2

                if phone_confirmed:
                    is_eng, label, color = False, "Phone", (255, 0, 255)
                    phones_now += 1
                else:
                    recent = list(history[tid])[-3:]
                    hd = [f["head_down"] for f in recent if f["head_down"] is not None]
                    head_down_now = (np.mean(hd) > 0.6) if hd else False
                    box_frac = (y2 - y1) / frame_h
                    face_clear = (sig["nose_conf"] + sig["eye_conf"]) / 2 >= MIN_FACE_CONF
                    if (USE_MODEL and box_frac >= MIN_BOX_FRAC and face_clear
                            and not head_down_now and len(history[tid]) >= WINDOW // 2):
                        vec = np.array([[features(history[tid])[k] for k in FEATURE_ORDER]])
                        is_eng = bool(clf.predict(vec)[0] == 1)
                        label = "Engaged" if is_eng else "Disengaged"
                    elif sig["head_down"] == 1.0:
                        is_eng, label = True, "Working"
                    elif sig["face_ratio"] is not None:
                        is_eng = 0.3 < sig["face_ratio"] < 0.7
                        label = "Engaged" if is_eng else "Distracted"
                    else:
                        is_eng, label = True, "Working?"
                    color = (0, 255, 0) if is_eng else (0, 0, 255)

                eng_n += 1 if is_eng else 0
                dis_n += 0 if is_eng else 1

                if not HEADLESS:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"#{tid} {label}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            students = eng_n + dis_n
            pct = (eng_n / students) * 100 if students else 0
            second = int(frame_idx / fps)
            if second != last_second:
                times.append(second)
                engagement_log.append(pct)
                phone_log.append(phones_now)
                last_second = second

            if not HEADLESS:
                for (_, _, px1, py1, px2, py2) in cached_phones:
                    cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)),
                                  (255, 0, 255), 2)
                cv2.putText(frame, f"Engaged: {int(pct)}%  ({eng_n}/{students})   "
                                   f"Phones: {phones_now}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        processed += 1
        if processed % 25 == 0:
            el = time.time() - start_time
            pctdone = (frame_idx / total_frames * 100) if total_frames else 0
            eta = (el / frame_idx * (total_frames - frame_idx)) if frame_idx else 0
            print(f"  {pctdone:5.1f}%  |  {len(times)}s logged  |  "
                  f"{el:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

        if not HEADLESS:
            cv2.imshow(f"LearnPulse: {name} - press Q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

except KeyboardInterrupt:
    print("\nInterrupted - saving what was analysed so far...")

cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()
save()
