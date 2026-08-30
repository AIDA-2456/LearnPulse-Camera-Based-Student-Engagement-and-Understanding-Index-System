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

if len(sys.argv) >= 2:
    video_path = sys.argv[1]
else:
    video_path = "data/classroom_session.mp4"
    print("no video given, defaulting to", video_path)

if not os.path.exists(video_path):
    print("ERROR: couldn't find file:", video_path)
    exit()

vid_name = os.path.splitext(os.path.basename(video_path))[0]

use_model = False
try:
    import joblib
    bundle = joblib.load("engagement_model.joblib")
    clf = bundle["model"]
    feature_order = bundle["features"]
    use_model = True
    print("loaded engagement_model.joblib, using trained model")
except Exception as e:
    print("no model found, falling back to rules:", e)

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
LOWER_BODY_KPTS = [11, 12, 13, 14, 15, 16]  

WINDOW = int(fps * 5) 
track_history = defaultdict(lambda: deque(maxlen=WINDOW))

times = []
engagement_pct_log = []
last_logged_sec = -1
frame_idx = 0


def get_frame_signals(kpts_for_person, box):
    x1, y1, x2, y2 = (float(v) for v in box)
    box_h = max(y2 - y1, 1.0)  
    nose = kpts_for_person[NOSE]
    l_eye, r_eye = kpts_for_person[L_EYE], kpts_for_person[R_EYE]
    l_ear, r_ear = kpts_for_person[L_EAR], kpts_for_person[R_EAR]
    l_sh, r_sh = kpts_for_person[L_SHOULDER], kpts_for_person[R_SHOULDER]

    sig = {
        "nose_conf": float(nose[2]),
        "eye_conf": (float(l_eye[2]) + float(r_eye[2])) / 2,
        "nose_x": float(nose[0]) / box_h if nose[2] > 0.5 else None,
        "nose_y": float(nose[1]) / box_h if nose[2] > 0.5 else None,
        "face_ratio": None,
        "head_down": None,
        "nose_to_shoulder": None,
        "shoulder_width": None,
        "eye_nose_dist": None,
    }

    # where's the nose sitting between the two ears -> proxy for "facing camera"
    if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
        lx = float(min(l_ear[0], r_ear[0]))
        rx = float(max(l_ear[0], r_ear[0]))
        if rx > lx:
            sig["face_ratio"] = (float(nose[0]) - lx) / (rx - lx)

    if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
        shoulder_y = (float(l_sh[1]) + float(r_sh[1])) / 2
        sig["head_down"] = 1.0 if float(nose[1]) >= shoulder_y else 0.0
        sig["nose_to_shoulder"] = (shoulder_y - float(nose[1])) / box_h
        sig["shoulder_width"] = abs(float(r_sh[0]) - float(l_sh[0])) / box_h

    if nose[2] > 0.5 and l_eye[2] > 0.3 and r_eye[2] > 0.3:
        eye_mid_x = (float(l_eye[0]) + float(r_eye[0])) / 2
        eye_mid_y = (float(l_eye[1]) + float(r_eye[1])) / 2
        sig["eye_nose_dist"] = float(np.hypot(float(nose[0]) - eye_mid_x,
                                               float(nose[1]) - eye_mid_y)) / box_h

    return sig


def build_features(hist):

    def col(key):
        return [f[key] for f in hist if f[key] is not None]

    def avg(vals):
        return float(np.mean(vals)) if vals else -1.0

    def std(vals):
        return float(np.std(vals)) if len(vals) > 1 else -1.0

    face_ratios = col("face_ratio")
    nx, ny = col("nose_x"), col("nose_y")

    if len(nx) > 2:
        step_sizes = np.hypot(np.diff(nx), np.diff(ny))
        move_mean = float(np.mean(step_sizes))
        move_max = float(np.max(step_sizes))
        big_move_thresh = np.mean(step_sizes) + 2 * np.std(step_sizes)
        num_big_moves = int(np.sum(step_sizes > big_move_thresh)) if len(step_sizes) > 2 else 0

        med_step = np.median(step_sizes)
        longest_still_run = 0
        run = 0
        for s in step_sizes:
            if s <= med_step:
                run += 1
            else:
                run = 0
            longest_still_run = max(longest_still_run, run)

        still_frac = float(np.mean(step_sizes <= med_step))

        mid = len(step_sizes) // 2
        drift = float(np.mean(step_sizes[mid:]) - np.mean(step_sizes[:mid])) if mid else 0.0
    else:
        move_mean = move_max = -1.0
        num_big_moves = -1
        longest_still_run = -1
        still_frac = -1.0
        drift = 0.0

    return {
        "face_ratio_mean": avg(face_ratios),
        "face_ratio_std": std(face_ratios),
        "facing_front_frac": avg([1 if 0.3 < r < 0.7 else 0 for r in face_ratios]) if face_ratios else -1.0,
        "head_down_frac": avg(col("head_down")),
        "nose_conf_mean": avg(col("nose_conf")),
        "nose_x_std": std(nx),
        "nose_y_std": std(ny),
        "eye_conf_mean": avg(col("eye_conf")),
        "eye_nose_dist_mean": avg(col("eye_nose_dist")),
        "eye_nose_dist_std": std(col("eye_nose_dist")),
        "nose_to_shoulder_mean": avg(col("nose_to_shoulder")),
        "nose_to_shoulder_std": std(col("nose_to_shoulder")),
        "shoulder_width_mean": avg(col("shoulder_width")),
        "shoulder_width_std": std(col("shoulder_width")),
        "move_mean": move_mean,
        "move_max": move_max,
        "big_moves": num_big_moves,
        "longest_still": longest_still_run,
        "still_frac": still_frac,
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
        num_people = len(kpts)

        teacher_idx = -1
        best_score = -1
        for i in range(num_people):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            area = (x2 - x1) * (y2 - y1)
            standing_pts = sum(1 for k in LOWER_BODY_KPTS if float(kpts[i][k][2]) > 0.3)
            score = standing_pts * 1_000_000 + area
            if score > best_score:
                best_score = score
                teacher_idx = i

        engaged_count = 0
        disengaged_count = 0

        for i in range(num_people):
            x1, y1, x2, y2 = (int(v) for v in boxes[i])
            track_id = int(ids[i]) if ids is not None else -1

            if i == teacher_idx:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"TEACHER #{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                continue

            track_history[track_id].append(get_frame_signals(kpts[i], boxes[i]))

            if use_model and len(track_history[track_id]) >= WINDOW // 2:
                feats = build_features(track_history[track_id])
                feat_vec = np.array([[feats[k] for k in feature_order]])
                engaged = bool(clf.predict(feat_vec)[0] == 1)
            else:

                latest = track_history[track_id][-1]
                engaged = latest["face_ratio"] is not None and 0.3 < latest["face_ratio"] < 0.7

            if engaged:
                label, color = "Engaged", (0, 255, 0)
                engaged_count += 1
            else:
                label, color = "Disengaged", (0, 0, 255)
                disengaged_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id} {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        total_students = engaged_count + disengaged_count
        engaged_pct = (engaged_count / total_students * 100) if total_students else 0

        cur_sec = int(frame_idx / fps)
        if cur_sec != last_logged_sec:
            times.append(cur_sec)
            engagement_pct_log.append(engaged_pct)
            last_logged_sec = cur_sec

        mode_label = "MODEL" if use_model else "RULES"
        cv2.putText(frame, f"[{mode_label}] Engaged: {int(engaged_pct)}%  ({engaged_count}/{total_students})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(f"ML Analyzer: {vid_name} - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ---- dump results ----
if times:
    csv_path = f"engagement_log_{vid_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "engaged_percent"])
        for t, pct in zip(times, engagement_pct_log):
            writer.writerow([t, round(pct, 1)])

    plt.figure(figsize=(10, 4))
    plt.plot([t / 60 for t in times], engagement_pct_log, color="#1D9E75", linewidth=2)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Class engagement (%)")
    plt.title(f"Engagement over time (trained model) - {vid_name}")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = f"engagement_{vid_name}.png"
    plt.savefig(png_path, dpi=120)
    print(f"saved {png_path} and {csv_path}")
else:
    print("no engagement data logged - did the video actually have frames?")