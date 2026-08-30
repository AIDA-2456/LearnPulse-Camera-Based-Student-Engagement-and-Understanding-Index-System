import cv2
import csv
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")          # save graphs to file without needing a screen
import matplotlib.pyplot as plt

model = YOLO("yolov8n-pose.pt")

video_path = "data/classroom_session.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video. Check the filename/extension above.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0   # frames per second of the video

# COCO keypoint indices
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
LOWER_BODY = [11, 12, 13, 14, 15, 16]     # visible when STANDING (teacher)

# Logs for the engagement-over-time graph
times = []          # seconds
engagement = []     # class engagement %
last_second = -1
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Finished the video.")
        break

    # track() gives each person a consistent ID across frames
    results = model.track(frame, persist=True, conf=0.4, verbose=False)

    for result in results:
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue

        kpts = result.keypoints.data
        boxes = result.boxes.xyxy
        ids = result.boxes.id            # may be None on some frames
        n = len(kpts)

        # ---- find the teacher (standing + largest body) ----
        teacher_idx, best_score = -1, -1
        for p in range(n):
            x1, y1, x2, y2 = (float(v) for v in boxes[p])
            area = (x2 - x1) * (y2 - y1)
            standing = sum(1 for k in LOWER_BODY if float(kpts[p][k][2]) > 0.3)
            score = standing * 1_000_000 + area
            if score > best_score:
                best_score, teacher_idx = score, p

        # ---- analyse each person ----
        engaged = head_down = distracted = 0

        for p in range(n):
            person = kpts[p]
            x1, y1, x2, y2 = (int(v) for v in boxes[p])
            track_id = int(ids[p]) if ids is not None else -1

            if p == teacher_idx:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"TEACHER #{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                continue

            nose = person[NOSE]
            l_ear, r_ear = person[L_EAR], person[R_EAR]
            l_sh, r_sh = person[L_SHOULDER], person[R_SHOULDER]

            facing_front = is_head_down = False

            if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
                left_x = float(min(l_ear[0], r_ear[0]))
                right_x = float(max(l_ear[0], r_ear[0]))
                if right_x > left_x:
                    ratio = (float(nose[0]) - left_x) / (right_x - left_x)
                    facing_front = 0.3 < ratio < 0.7

            if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
                shoulder_y = (float(l_sh[1]) + float(r_sh[1])) / 2
                if float(nose[1]) >= shoulder_y:
                    is_head_down = True

            if is_head_down:
                label, color = "Head down", (0, 165, 255)
                head_down += 1
            elif facing_front:
                label, color = "Engaged", (0, 255, 0)
                engaged += 1
            else:
                label, color = "Distracted", (0, 0, 255)
                distracted += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id} {label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---- log engagement once per second ----
        students = engaged + head_down + distracted
        pct = (engaged / students) * 100 if students > 0 else 0
        second = int(frame_idx / fps)
        if second != last_second:
            times.append(second)
            engagement.append(pct)
            last_second = second

        cv2.putText(frame, f"Engaged: {int(pct)}%   "
                           f"[E:{engaged}  Head-down:{head_down}  Distracted:{distracted}]",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Classroom Analysis (tracking) - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ---- save the results: CSV + graph ----
if times:
    with open("engagement_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "engaged_percent"])
        for t, e in zip(times, engagement):
            writer.writerow([t, round(e, 1)])

    plt.figure(figsize=(10, 4))
    plt.plot([t / 60 for t in times], engagement, color="#1D9E75", linewidth=2)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Class engagement (%)")
    plt.title("Class engagement over time")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("engagement_over_time.png", dpi=120)
    print("Saved: engagement_over_time.png and engagement_log.csv")
else:
    print("No engagement data was logged (no students detected).")
