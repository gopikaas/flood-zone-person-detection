import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

input_video_path = r'C:\Users\Administrator\Downloads\WhatsApp Video 2025-06-30 at 10.34.34 AM(1).mp4'
output_video_path = r'C:\Users\Administrator\Downloads\flood_output.mp4'

model = YOLO('yolov8n.pt')

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def classify_gender(image):
    return np.random.choice(["Male", "Female", "Child"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    male_count = 0
    female_count = 0
    child_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            if label == "person" and conf > 0.4:
                bbox_height = y2 - y1
                water_level = y2 / height
                distance_factor = 1 - (bbox_height / height)

                if distance_factor > 0.7 and water_level > 0.7:
                    color, zone = (0, 0, 255), "Danger Zone"
                elif distance_factor > 0.4 and water_level > 0.5:
                    color, zone = (255, 0, 0), "Risky Zone"
                else:
                    color, zone = (0, 255, 0), "Safe Zone"

                person_crop = frame[y1:y2, x1:x2]
                gender = classify_gender(person_crop)

                if gender == "Male":
                    male_count += 1
                elif gender == "Female":
                    female_count += 1
                else:
                    child_count += 1

                detections.append([[x1, y1, x2, y2], conf, cls])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{gender} - {zone}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

    tracks = tracker.update_tracks(detections, frame=frame)

    cv2.putText(
        frame,
        f"Male: {male_count}  Female: {female_count}  Children: {child_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
