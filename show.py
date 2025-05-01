import cv2
import os
import xml.etree.ElementTree as ET


DEMO_NUMBER = 2

# === Параметры ===
VIDEO_PATH = f'./Demo{DEMO_NUMBER}/Video.mp4'
ANNOTATION_DIR = f'./Demo{DEMO_NUMBER}/Annotations/'  # Папка, где лежат XML файлы
RESIZE_TO = (640, 360)  # Ширина, Высота
CLASSES = ['drone', 'bird', 'plane']  # Пример классов

# === Функция для парсинга XML ===
def parse_voc(xml_path, scale_x, scale_y):
    boxes = []
    if not os.path.exists(xml_path):
        raise Exception(xml_path)

    root = ET.parse(xml_path).getroot()
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text) * scale_x)
        ymin = int(float(bndbox.find('ymin').text) * scale_y)
        xmax = int(float(bndbox.find('xmax').text) * scale_x)
        ymax = int(float(bndbox.find('ymax').text) * scale_y)
        boxes.append((class_name, [xmin, ymin, xmax, ymax]))
    return boxes

# === Загрузка видео ===
cap = cv2.VideoCapture(VIDEO_PATH)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale_x = RESIZE_TO[0] / orig_width
scale_y = RESIZE_TO[1] / orig_height

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, RESIZE_TO)

    # Путь до XML-файла, соответствующего текущему фрейму
    annotation_path = os.path.join(ANNOTATION_DIR, f"frame_{frame_idx:06d}.xml")  # 000001.xml

    boxes = parse_voc(annotation_path, scale_x, scale_y)
    print(boxes)
    for class_name, (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Video with VOC", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
