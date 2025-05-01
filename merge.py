import cv2
import numpy as np

def overlay_videos(video1_path, video2_path, output_path, overlay_alpha=0.5):
    """
    Накладывает два видео друг на друга и сохраняет результат.
    
    Параметры:
        video1_path (str): Путь к первому видео (фон).
        video2_path (str): Путь ко второму видео (накладываемое).
        output_path (str): Путь для сохранения результата.
        overlay_alpha (float): Прозрачность накладываемого видео (0.0 - 1.0).
    """
    # Открываем видеофайлы
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Проверяем, открылись ли видео
    if not cap1.isOpened() or not cap2.isOpened():
        print("Ошибка: не удалось открыть видеофайлы!")
        return
    
    # Получаем параметры видео (размеры, FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Если одно из видео закончилось, прерываем цикл
        if not ret1 or not ret2:
            break
        
        # Изменяем размер второго видео под размер первого
        frame2 = cv2.resize(frame2, (width, height))
        
        # Накладываем видео с прозрачностью
        overlay = cv2.addWeighted(frame1, 1 - overlay_alpha, 
                                 frame2, overlay_alpha, 0)
        
        # Записываем кадр в выходное видео
        out.write(overlay)
        
        # Для отображения процесса (опционально)
        cv2.imshow('Overlay Preview', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

# Пример использования

DEMO_N = 1
DEMO_K = 7

name_1 = f'./Demo{DEMO_N}/Video.mp4'
name_2 = f'./Demo{DEMO_K}/Video.mp4'
overlay_videos(name_1, name_2, f'Demo{DEMO_N}-Demo{DEMO_K}.mp4', overlay_alpha=0.5)