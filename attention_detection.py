import cv2
from ultralytics import YOLO

# Pose model: finds body keypoints (nose, eyes, ears, shoulders...).
# On the FIRST run this downloads "yolov8n-pose.pt" automatically (needs internet).
model = YOLO("yolov8n-pose.pt")

# Your video file (change name/extension if yours is different)
video_path = "data/classroom_session.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    print("Check the filename and extension on the line above.")
    exit()

# COCO keypoint indices (the order YOLO-pose returns them in)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Finished the video (or could not read a frame).")
        break

    results = model(frame, conf=0.4, verbose=False)

    total = 0
    attentive = 0

    for result in results:
        if result.keypoints is None:
            continue

        # kpts shape: [num_people, 17 keypoints, 3 values (x, y, confidence)]
        kpts = result.keypoints.data
        boxes = result.boxes.xyxy

        for p in range(len(kpts)):
            person = kpts[p]
            x1, y1, x2, y2 = map(int, boxes[p])

            nose = person[NOSE]
            l_ear = person[L_EAR]
            r_ear = person[R_EAR]

            label = "Looking away"
            color = (0, 0, 255)  # red

            # Only judge head direction if we can see the nose and both ears
            if nose[2] > 0.5 and l_ear[2] > 0.3 and r_ear[2] > 0.3:
                left_x = float(min(l_ear[0], r_ear[0]))
                right_x = float(max(l_ear[0], r_ear[0]))
                if right_x > left_x:
                    # Where the nose sits between the two ears (0.5 = centred)
                    ratio = (float(nose[0]) - left_x) / (right_x - left_x)
                    if 0.3 < ratio < 0.7:
                        label = "Facing front"
                        color = (0, 255, 0)  # green

                # Draw a small arrow from between the ears through the nose
                mid_x = int((left_x + right_x) / 2)
                mid_y = int((float(l_ear[1]) + float(r_ear[1])) / 2)
                cv2.arrowedLine(frame, (mid_x, mid_y),
                                (int(nose[0]), int(nose[1])), color, 2, tipLength=0.4)

            total += 1
            if label == "Facing front":
                attentive += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Live class attention summary
    pct = int((attentive / total) * 100) if total > 0 else 0
    cv2.putText(frame, f"Attention: {pct}%  ({attentive}/{total} facing front)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Attention Detection - press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
