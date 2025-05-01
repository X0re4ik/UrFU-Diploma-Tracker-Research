import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Импорты для трекеров
from deep_sort_realtime.deepsort_tracker import DeepSort

# from bytetrack import BYTETracker
# from oc_sort import OCSORT

# === Параметры ===
RESNET_MODEL_PATH = "resnet18_bpla-21-04-2025-13_45.pth"
CLASSES = [
    "A22 Foxbat",
    "Bayraktar TB2",
    "UJ-22 Airborne",
    "Unknown",
]

# === Инициализация моделей ===
model = YOLO("yolo-2025-04-20 11_18_07.386057.pt")
resnet_model = torch.load(RESNET_MODEL_PATH, map_location=torch.device("cpu"))
resnet_model.eval()

# === Преобразование для ResNet18 ===
resnet_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# === Загрузка видео ===
video_name = "./overlay_output.mp4"
video_path = f"./{video_name}"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(f"videos_output/{video_name}", fourcc, fps, (width, height))

# === Выбор трекера ===
tracker_type = "DeepSORT"  # Меняй на "DeepSORT" или "OC-SORT"

# Инициализация трекера
if tracker_type == "DeepSORT":
    tracker = DeepSort(max_age=30)
elif tracker_type == "ByteTrack":
    tracker = DeepSort(max_age=30)
elif tracker_type == "OC-SORT":
    tracker = DeepSort(max_age=30)
else:
    raise ValueError("Неизвестный тип трекера")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов YOLOv8
    # Детекция объектов YOLOv8
    results = model(frame)[0]

    detections = []
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, classes):
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(float, box[:4])
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, int(cls)))

    # Трекинг
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        x1, y1, x2, y2 = map(int, ltrb)
        cls = track.det_class

        if cls == 0:
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = resnet_transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                logits = resnet_model(input_tensor)
                probs = F.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)
                label = (
                    f"ID {track_id}: {CLASSES[predicted_class]} {confidence.item():.2f}"
                )
        elif cls == 1:
            label = f"ID {track_id}: Target Class 1"

        # Рисуем трек и метку
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    cv2.imshow("Video with VOC", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
