import configparser
import os

import cv2
import numpy as np

import config


def convert_to_skin_mask(image):
    # Конвертация в YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Диапазон цвета кожи
    lower_skin = config.LOWER_SKIN
    upper_skin = config.UPPER_SKIN
    # Создание маски кожи
    return cv2.inRange(ycrcb, lower_skin, upper_skin)


def clean_mask(skin_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    return skin_mask


def find_valid_contours(mask, min_area=10000, aspect_range=(0.3, 3.0)):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Фильтрация контуров
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:  # слишком маленькие
            continue
        # Проверка соотношения сторон
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_range[0] <= aspect_ratio <= aspect_range[1]:
            valid_contours.append(cnt)
    # Сортируем контуры по размеру (от большего к меньшему)
    return sorted(valid_contours, key=cv2.contourArea, reverse=True)


def debug_skin_mask_with_trackbars():
    """
    Отладочная функция — позволяет настраивать диапазон кожи в YCrCb через ползунки в реальном времени.
    """
    config_path = "config.py"

    def nothing(x):
        pass

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Trackbars")

    # Создание ползунков для диапазонов Y, Cr, Cb
    cv2.createTrackbar("Y Min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("Y Max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Cr Min", "Trackbars", 133, 255, nothing)
    cv2.createTrackbar("Cr Max", "Trackbars", 173, 255, nothing)
    cv2.createTrackbar("Cb Min", "Trackbars", 77, 255, nothing)
    cv2.createTrackbar("Cb Max", "Trackbars", 127, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получаем значения с ползунков
        y_min = cv2.getTrackbarPos("Y Min", "Trackbars")
        y_max = cv2.getTrackbarPos("Y Max", "Trackbars")
        cr_min = cv2.getTrackbarPos("Cr Min", "Trackbars")
        cr_max = cv2.getTrackbarPos("Cr Max", "Trackbars")
        cb_min = cv2.getTrackbarPos("Cb Min", "Trackbars")
        cb_max = cv2.getTrackbarPos("Cb Max", "Trackbars")

        # Применяем маску
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([y_min, cr_min, cb_min], dtype=np.uint8)
        upper = np.array([y_max, cr_max, cb_max], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        # Результат — наложение маски на исходное изображение
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Показываем
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"New range saved to {config_path}")
            save_skin_range_to_config(lower, upper, config_path)
            break

    cap.release()
    cv2.destroyAllWindows()


def save_skin_range_to_config(mask: tuple, lower: np.ndarray, upper: np.ndarray, config_path: str):
    """
    Заменяет константы MASK_COLOR, LOWER_SKIN и UPPER_SKIN в config.py на новые значения.
    """
    if not os.path.exists(config_path):
        print(f"Файл {config_path} не найден.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def format_array(arr):
        return f"np.array([{', '.join(map(str, arr.tolist()))}], dtype=np.uint8)"

    new_lower = f"LOWER_SKIN = {format_array(lower)}\n"
    new_upper = f"UPPER_SKIN = {format_array(upper)}\n"
    new_mask = f"MASK_COLOR = {mask}\n"

    new_lines = []
    for line in lines:
        if line.strip().startswith("LOWER_SKIN"):
            new_lines.append(new_lower)
        elif line.strip().startswith("UPPER_SKIN"):
            new_lines.append(new_upper)
        elif line.strip().startswith("MASK_COLOR"):
            new_lines.append(new_mask)
        else:
            new_lines.append(line)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
