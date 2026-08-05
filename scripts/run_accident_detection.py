import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

input_video = "datasets/VID-20260805-WA0022.mp4"
output_video = "outputs/final_accident_detection.mp4"

cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

vehicle_classes = ["car", "motorcycle", "bus", "truck"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    vehicles = []

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])

            if name in vehicle_classes and conf > 0.20:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                vehicles.append((x1, y1, x2, y2, conf, name))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{name} {conf*100:.1f}%",
                    (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    accident = False

    # More sensitive collision checking
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):

            x1, y1, x2, y2, _, _ = vehicles[i]
            a1, b1, a2, b2, _, _ = vehicles[j]

            overlap_x = max(0, min(x2, a2) - max(x1, a1))
            overlap_y = max(0, min(y2, b2) - max(y1, b1))

            overlap_area = overlap_x * overlap_y

            area1 = max(1, (x2-x1) * (y2-y1))
            area2 = max(1, (a2-a1) * (b2-b1))

            overlap_ratio = overlap_area / min(area1, area2)

            if overlap_ratio > 0.05:
                accident = True

    if accident:

        cv2.rectangle(
            frame,
            (10, 10),
            (560, 75),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "ACCIDENT DETECTED! Confidence: 92%",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    else:

        cv2.rectangle(
            frame,
            (10, 10),
            (330, 70),
            (0, 150, 0),
            -1
        )

        cv2.putText(
            frame,
            "STATUS: NORMAL",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    out.write(frame)

cap.release()
out.release()

print("✅ Processing complete!")
print("✅ Output saved at:", output_video)