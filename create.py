import cv2

# === Пути к видео ===
video_path_1 = "./Demo1/Video.mp4"
video_path_2 = "./Demo2/Video.mp4"
output_path = "overlay_output.mp4"

# === Размер кадра ===
size = (250, 250)

# === Открытие видео ===
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

fps = int(cap1.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, size)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Изменение размера
    frame1 = cv2.resize(frame1, size)
    frame2 = cv2.resize(frame2, size)

    # Наложение (50% прозрачности для каждого)
    blended = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

    # Запись результата
    out.write(blended)

    # Для отображения (необязательно)
    cv2.imshow("Overlay", blended)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
