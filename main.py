import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort  # Добавляем DeepSort
import cv2
import os
import xml.etree.ElementTree as ET


from scipy.optimize import linear_sum_assignment


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


def calculate_mota(gt_data, pred_data, iou_threshold=0.5):
    total_gt = 0
    total_fn = 0
    total_fp = 0
    total_idsw = 0  # Пока пропускаем ID-смену (IDSW)

    for frame_gt, frame_pred in zip(gt_data, pred_data):
        gt_boxes = [box for cls, box in frame_gt]
        pred_boxes = [
            track.to_ltrb() if hasattr(track, "to_ltrb") else track
            for track in frame_pred
        ]

        total_gt += len(gt_boxes)

        # Построение IoU-матрицы
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                cost_matrix[i, j] = 1 - calculate_iou(gt, pred)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_pred = set()

        for i, j in zip(row_ind, col_ind):
            iou_val = 1 - cost_matrix[i, j]
            if iou_val >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(j)

        fn = len(gt_boxes) - len(matched_gt)
        fp = len(pred_boxes) - len(matched_pred)

        total_fn += fn
        total_fp += fp

    mota = 1 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)
    return mota


from collections import defaultdict


def flatten_data(data):
    """Преобразует данные из формата [[[id, bbox]]] в [(id, bbox)]"""
    flattened = {}
    for frame_idx, frame_data in data.items():
        clean_objs = []
        for obj_list in frame_data:
            if obj_list:  # Пропускаем пустые списки
                for obj in obj_list:
                    if len(obj) == 2:  # Проверяем, что это (id, bbox)
                        clean_objs.append(
                            (obj[0], list(map(float, obj[1])))
                        )  # Конвертируем np.float в обычный float
        flattened[frame_idx] = clean_objs
    return flattened


def calculate_idf1(gt_data, pred_data, iou_threshold=0.5):
    """
    gt_data: dict[frame_idx] = list of (gt_id, [x1, y1, x2, y2])
    pred_data: dict[frame_idx] = list of (track_id, [x1, y1, x2, y2])
    """
    # Предварительная обработка данных
    gt_data = flatten_data(gt_data)
    pred_data = flatten_data(pred_data)

    # print(gt_data)

    idtp = 0  # True Positive (TP)
    idfp = 0  # False Positive (FP)
    idfn = 0  # False Negative (FN)

    for frame_idx in sorted(gt_data.keys()):
        gt_objs = gt_data.get(frame_idx, [])
        pred_objs = pred_data.get(frame_idx, [])

        matched_gt = set()
        matched_pred = set()

        for gt_id, gt_box in gt_objs:
            best_iou = 0
            best_match = None

            for track_id, pred_box in pred_objs:
                if (track_id, frame_idx) in matched_pred:
                    continue

                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = (track_id, pred_box)

            if best_match:
                track_id, _ = best_match
                matched_gt.add((gt_id, frame_idx))
                matched_pred.add((track_id, frame_idx))

                if gt_id == track_id:
                    idtp += 1
                else:
                    idfp += 1
                    idfn += 1
            else:
                idfn += 1

        # Учет несоответствий
        idfp += len(pred_objs) - len(matched_pred)
        idfn += len(gt_objs) - len(matched_gt)

    idf1 = (
        (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
    )
    return idf1


def calculate_idf1_from_metrics(total_gt, fp, fn, idsw):
    idtp = total_gt - fn - idsw  # ID True Positives
    idfp = fp                   # ID False Positives
    idfn = fn                   # ID False Negatives
    
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
    return idf1

def calculate_fp_fn_idsw(gt_data, pred_data, iou_threshold=0.5):
    """
    Вычисляет FP, FN и IDSW (Identity Switches) для трекинга.

    Параметры:
        gt_data: dict[frame_idx] = list of (gt_id, [x1, y1, x2, y2])
        pred_data: dict[frame_idx] = list of (track_id, [x1, y1, x2, y2])
        iou_threshold: порог IoU для сопоставления объектов

    Возвращает:
        dict: {"FP": int, "FN": int, "IDSW": int}
    """
    # Предварительная обработка данных
    gt_data = flatten_data(gt_data)
    pred_data = flatten_data(pred_data)

    FP = 0  # False Positives (неправильные детекции)
    FN = 0  # False Negatives (пропущенные объекты)
    IDSW = 0  # Identity Switches (смена ID при трекинге)

    # Словарь для отслеживания предыдущих сопоставлений (gt_id -> pred_id)
    prev_matches = {}

    for frame_idx in sorted(gt_data.keys()):
        gt_objs = gt_data.get(frame_idx, [])
        pred_objs = pred_data.get(frame_idx, [])

        current_matches = {}  # {gt_id: pred_id} для текущего кадра

        # Сопоставление объектов на текущем кадре
        matched_pred = set()
        for gt_id, gt_box in gt_objs:
            best_iou = 0
            best_pred_id = None

            for pred_id, pred_box in pred_objs:
                if pred_id in matched_pred:
                    continue  # Пропускаем уже сопоставленные pred_objs

                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id

            if best_pred_id is not None:
                matched_pred.add(best_pred_id)
                current_matches[gt_id] = best_pred_id

                # Проверка на IDSW (если pred_id изменился для этого gt_id)
                if gt_id in prev_matches and prev_matches[gt_id] != best_pred_id:
                    IDSW += 1

        # Обновляем предыдущие сопоставления
        prev_matches = current_matches

        # Подсчет FP и FN
        FP += len(pred_objs) - len(matched_pred)  # Несопоставленные pred_objs = FP
        FN += len(gt_objs) - len(current_matches)  # Несопоставленные gt_objs = FN

    return {"FP": FP, "FN": FN, "IDSW": IDSW}


# === Параметры ===
RESNET_MODEL_PATH = "resnet18_bpla-21-04-2025-13_45.pth"
CLASSES = ["A22 Foxbat", "Bayraktar TB2", "UJ-22 Airborne", "Unknown"]

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

# === Инициализация Deep SORT ===
tracker = DeepSort(max_age=10, n_init=1, max_cosine_distance=0.6)

# === Загрузка видео ===

import os

DEMOS = [
    "Demo5",
    "Demo6",
]

ANNOTAION_PATH = [os.path.join(f"./{demo}/Annotations/") for demo in DEMOS]

VIDEO_PATH = [os.path.join(f"./{demo}/Video.mp4") for demo in DEMOS]


for video_path, annotaion_path in zip(ANNOTAION_PATH, VIDEO_PATH):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if not os.path.exists(annotaion_path):
        raise FileNotFoundError(annotaion_path)

ORIGINAL_SIZE: list[tuple[int, int]] = []

for video_path in VIDEO_PATH:
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ORIGINAL_SIZE.append((orig_width, orig_height))


assert len(ORIGINAL_SIZE) == len(ANNOTAION_PATH) == len(DEMOS)

print(ORIGINAL_SIZE)

video_name = f"{'-'.join(DEMOS)}.mp4"
video_path = f"./{video_name}"
cap = cv2.VideoCapture(video_path)


FPS = cap.get(cv2.CAP_PROP_FPS)

NEW_FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
NEW_FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    f"videos_output/{video_name}", fourcc, FPS, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT)
)


def parse_voc(xml_path: str, scale_x: int, scale_y: int):
    boxes = []
    if not os.path.exists(xml_path):
        raise Exception(xml_path)

    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text) * scale_x)
        ymin = int(float(bndbox.find("ymin").text) * scale_y)
        xmax = int(float(bndbox.find("xmax").text) * scale_x)
        ymax = int(float(bndbox.find("ymax").text) * scale_y)
        boxes.append((class_name, [xmin, ymin, xmax, ymax]))
    return boxes


TRUE_COUNT = 2

frame_idx = 0

processed_pred_data = []
gt_data = []

gt_data_idf1 = dict()
pred_data_idf1 = dict()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gt_data_idf1[frame_idx] = []
    pred_data_idf1[frame_idx] = []

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

    # Расчет MOTA
    processed_pred_data.append([track.to_ltrb() for track in tracks])

    pred_data_idf1[frame_idx].append(
        [[int(track.track_id), list(track.to_ltrb())] for track in tracks]
    )

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # [left, top, right, bottom]

        # Координаты трекера
        x1, y1, x2, y2 = map(int, ltrb)
        cls = track.det_class

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

    data = []
    # Путь до XML-файла, соответствующего текущему фрейму
    for i, a_path in enumerate(ANNOTAION_PATH):
        annotation_path = os.path.join(
            a_path, f"frame_{frame_idx:06d}.xml"
        )  # 000001.xml
        scale_x = NEW_FRAME_WIDTH / ORIGINAL_SIZE[i][0]
        scale_y = NEW_FRAME_HEIGHT / ORIGINAL_SIZE[i][1]

        boxes = parse_voc(annotation_path, scale_x, scale_y)

        gt_data_idf1[frame_idx].append(
            [[i, [xmin, ymin, xmax, ymax]] for _, (xmin, ymin, xmax, ymax) in boxes]
        )

        for class_name, (xmin, ymin, xmax, ymax) in boxes:

            data.append((class_name, [xmin, ymin, xmax, ymax]))

            # Реальные координаты объекта
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "True - Drone",
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.1,
                (0, 255, 0),
                2,
            )

    gt_data.append(data)

    cv2.imshow("Video with VOC", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
    
    if frame_idx == 150:
        break
    
    frame_idx += 1


def calculate_total_gt(gt_data):
    """
    Вычисляет общее количество объектов в ground truth данных.
    
    Параметры:
        gt_data (dict): Данные GT в формате {frame_idx: [[[obj_id, bbox]], ...]}, 
                        где bbox = [x1, y1, x2, y2].
    
    Возвращает:
        int: Общее количество объектов в GT.
    """
    total_gt = 0
    for frame_idx, frame_objs in gt_data.items():
        for obj_list in frame_objs:
            if obj_list:  # Игнорируем пустые списки
                total_gt += len(obj_list)
    return total_gt

print(f"MOTA: {calculate_mota(gt_data, processed_pred_data):.4f}")
#print(f"IDF1: {calculate_idf1(gt_data_idf1, pred_data_idf1):.4f}")
result = calculate_fp_fn_idsw(gt_data_idf1, pred_data_idf1)
print(f"ЫВАЫОВА: {result}")
# {'FP': 4, 'FN': 0, 'IDSW': 1}
print(calculate_idf1_from_metrics(calculate_total_gt(gt_data_idf1), result["FP"], result["FN"], result["IDSW"]))

cap.release()
cv2.destroyAllWindows()
