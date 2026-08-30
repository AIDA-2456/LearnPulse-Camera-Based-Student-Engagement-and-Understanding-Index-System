import cv2
from ultralytics import YOLO

# Pose model: finds body keypoints (nose, eyes, ears, shoulders, hips, knees...).
# First run downloads "yolov8n-pose.pt" automatically (needs internet).
model = YOLO("yolov8n-pose.pt")

video_path = "data/classroom_session.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    print("Check the filename and extension on the line above.")
    exit()

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
LOWER_BODY = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles -> visible when STANDING

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Finished the video (or could not read a frame).")
        break

    results = model(frame, conf=0.4, verbose=False)

    for result in results:
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue

        kpts = result.keypoints.data        # [people, 17, 3] -> (x, y, conf)
        boxes = result.boxes.xyxy
        n = len(kpts)

        # ---- STEP 1: find the teacher (standing + largest body) ----
        teacher_idx = -1
        best_score = -1
        for p in range(n):
            person = kpts[p]
            x1, y1, x2, y2 = (float(v) for v in boxes[p])
            area = (x2 - x1) * (y2 - y1)
            standing = sum(1 for k in LOWER_BODY if float(person[k][2]) > 0.3)
            score = standing * 1_000_000 + area  # standing dominates, area breaks ties
            if score > best_score:
                best_score = score
                teacher_idx = p

        # ---- STEP 2: analyse each person ----
        engaged = head_down = distracted = 0

        for p in range(n):
            person = kpts[p]
            x1, y1, x2, y2 = (int(v) for v in boxes[p])

            # The teacher: label separately, do not score as a student
            if p == teacher_idx:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, "TEACHER", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                continue

            nose = person[NOSE]
            l_ear, r_ear = person[L_EAR], person[R_EAR]
            l_sh, r_sh = person[L_SHOULDER], person[R_SHOULDER]

            facing_front = False
            is_head_down = False

            # Head direction: nose centred between the two ears = facing front
            if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
                left_x = float(min(l_ear[0], r_ear[0]))
                right_x = float(max(l_ear[0], r_ear[0]))
                if right_x > left_x:
                    ratio = (float(nose[0]) - left_x) / (right_x - left_x)
                    facing_front = 0.3 < ratio < 0.7

            # Head down: nose has dropped to/below shoulder level (likely writing)
            if nose[2] > 0.5 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
                shoulder_y = (float(l_sh[1]) + float(r_sh[1])) / 2
                if float(nose[1]) >= shoulder_y:
                    is_head_down = True

            # Decide the behaviour state
            if is_head_down:
                label, color = "Head down", (0, 165, 255)   # orange
                head_down += 1
            elif facing_front:
                label, color = "Engaged", (0, 255, 0)        # green
                engaged += 1
            else:
                label, color = "Distracted", (0, 0, 255)     # red
                distracted += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---- STEP 3: class summary (students only) ----
        students = engaged + head_down + distracted
        pct = int((engaged / students) * 100) if students > 0 else 0
        cv2.putText(frame, f"Engaged: {pct}%   "
                           f"[E:{engaged}  Head-down:{head_down}  Distracted:{distracted}]",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Classroom Behaviour Analysis - press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
