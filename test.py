from collections import defaultdict

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def calculate_idf1(gt_data, pred_data, iou_threshold=0.5):
    """
    gt_data: dict[frame_idx] = list of (gt_id, [x1, y1, x2, y2])
    pred_data: dict[frame_idx] = list of (track_id, [x1, y1, x2, y2])
    """
    matches_by_gt = defaultdict(set)
    matches_by_pred = defaultdict(set)

    idtp = 0  # True Positive (TP)
    idfp = 0  # False Positive (FP)
    idfn = 0  # False Negative (FN)

    for frame_idx in sorted(gt_data.keys()):
        gt_objs = gt_data.get(frame_idx, [])
        pred_objs = pred_data.get(frame_idx, [])

        # Фильтрация пустых элементов (если pred_objs содержит не только (track_id, bbox))
        pred_objs = [obj for obj in pred_objs if len(obj) == 2]  # <-- Исправление здесь

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

        # Обработка несоответствий
        idfp += len(pred_objs) - len(matched_pred)
        idfn += len(gt_objs) - len(matched_gt)

    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
    return idf1

# Пример данных
gt_data = {
    0: [(1, [100, 100, 150, 150]), (2, [200, 200, 250, 250])],
    1: [(1, [105, 105, 155, 155]), (2, [205, 205, 255, 255])],
    2: [(1, [110, 110, 160, 160])],
}

pred_data = {
    0: [(1, [102, 102, 152, 152]), (3, [198, 198, 248, 248])],  # 1 совпадает, 3 вместо 2
    1: [(1, [106, 106, 156, 156]), []],  # оба правильные ID и боксы близкие
    2: [(1, [300, 300, 350, 350])],  # объект не совпадает с gt — будет FN + FP
}

idf1 = calculate_idf1(gt_data, pred_data)
print(f"IDF1: {idf1}")
