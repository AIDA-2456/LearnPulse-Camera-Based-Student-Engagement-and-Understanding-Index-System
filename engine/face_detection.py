import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model.
# On the FIRST run this downloads "yolov8n.pt" automatically (needs internet).
model = YOLO("yolov8n.pt")

# Your video file (change the name/extension here if yours is different)
video_path = "data/classroom_session.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    print("Check the filename and extension on the line above.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Finished the video (or could not read a frame).")
        break

    # Detect only people.
    # class 0 = "person" in the COCO dataset YOLO was trained on.
    # conf=0.4 means: only keep detections the model is at least 40% sure about.
    results = model(frame, classes=[0], conf=0.4, verbose=False)

    person_count = 0
    for result in results:
        for box in result.boxes:
            # Get the corners of the box around each person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_count += 1

    # Show how many people were found, on the video itself
    cv2.putText(frame, f"People detected: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("People Detection - press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()