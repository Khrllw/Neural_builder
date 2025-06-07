import importlib

import customtkinter as ctk
import cv2
import numpy as np
import threading
from tkinter import filedialog
from PIL import ImageTk, Image

import config

importlib.reload(config)
from neural_network import NeuralNetwork
from image_proc import ImageProcessor


class LiveDetectionFrame(ctk.CTkFrame):
    """
    Фрейм для запуска живого распознавания жестов с камеры.
    Позволяет загружать модель, включать/отключать детекцию и отображать маски.
    """

    def __init__(self, parent):
        super().__init__(parent)

        # Модель и состояния
        self.model = None
        self.gesture_names = []
        self.running = False
        self.capture_thread = None
        self.cap = None
        importlib.reload(config)

        # Построение GUI
        self._build_controls()
        self._build_display()

    def _build_controls(self):
        """Создаёт панель с кнопками и чекбоксами управления."""
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(fill="x", pady=(10, 5), padx=10)

        self.select_button = ctk.CTkButton(self.controls_frame, text="Загрузить модель", command=self.load_model)
        self.select_button.pack(side="left", padx=5)

        self.start_button = ctk.CTkButton(self.controls_frame, text="Запустить", command=self.start_detection, state="disabled")
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ctk.CTkButton(self.controls_frame, text="Остановить", command=self.stop_detection, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.show_mask_var = ctk.IntVar(value=0)
        self.checkbox_show_mask = ctk.CTkCheckBox(self.controls_frame, text="Показать маску", variable=self.show_mask_var)
        self.checkbox_show_mask.pack(side="left", padx=10)

        self.show_mask_bw_var = ctk.IntVar(value=0)
        self.checkbox_show_mask_bw = ctk.CTkCheckBox(self.controls_frame, text="Ч/Б маска", variable=self.show_mask_bw_var)
        self.checkbox_show_mask_bw.pack(side="left", padx=10)

    def _build_display(self):
        """Создаёт область для отображения видео с наложением."""
        self.image_container = ctk.CTkFrame(self)
        self.image_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_container.grid_rowconfigure(0, weight=1)
        self.image_container.grid_columnconfigure(0, weight=1)

        self.left_frame = ctk.CTkFrame(self.image_container, fg_color="#3A3A3A", corner_radius=10)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.left_label = ctk.CTkLabel(self.left_frame, text="Исходное изображение", font=ctk.CTkFont(size=14, weight="bold"))
        self.left_label.grid(row=0, column=0, sticky="ew", pady=(10, 5), padx=5)

        self.left_image_label = ctk.CTkLabel(self.left_frame, text="")
        self.left_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))

    def load_model(self):
        """Загружает модель из файла и активирует кнопку запуска."""
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl")])
        if path:
            self.model = NeuralNetwork()
            self.model.load_model(path)
            self.gesture_names = ["Paper", "Rock", "Scissors"]
            self.start_button.configure(state="normal")

    def start_detection(self):
        """Запускает поток захвата и обработки видео."""
        if self.model is None:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def stop_detection(self):
        """Останавливает захват и обработку видео."""
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def on_close(self):
        """Вызывается при закрытии фрейма — останавливает камеру и поток."""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def capture_loop(self):
        """Основной цикл обработки видеопотока."""
        self.cap = cv2.VideoCapture(0)
        stable_gestures = [[], []]
        stable_display = [None, None]
        stable_threshold = 3
        conf_threshold = 0.8

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            hands, bboxes, skin_mask = self.preprocess_frame(frame)

            overlay = self._apply_mask(frame, skin_mask)
            hands, bboxes = self.sort_hands_by_position(hands, bboxes)

            stable_display = self.process_hands(hands, bboxes, stable_gestures, stable_display,
                                                stable_threshold, conf_threshold, overlay)

            self.update_image(overlay, self.left_image_label)

        if self.cap:
            self.cap.release()

    def preprocess_frame(self, frame):
        """Предобработка кадра — извлечение рук и маски."""
        return ImageProcessor.preprocess_image(frame, max_hands=2)

    def get_skin_mask(self, frame):
        """Возвращает бинарную маску кожи на основе конфигурации."""
        if frame is None:
            return np.zeros((1, 1), dtype=np.uint8)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array(config.LOWER_SKIN, dtype=np.uint8)
        upper = np.array(config.UPPER_SKIN, dtype=np.uint8)

        mask = cv2.inRange(ycrcb, lower, upper)
        return mask

    def _apply_mask(self, frame, mask):
        """Накладывает маску на изображение в зависимости от режима отображения."""
        if mask is None or frame is None:
            return frame.copy() if frame is not None else None

        if self.show_mask_var.get():
            if self.show_mask_bw_var.get():
                # Показываем маску как черно-белую накладку
                mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                return cv2.addWeighted(frame, 0.5, mask_display, 0.5, 0)
            else:
                # Показываем маску с цветовой картой
                color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                # Оставляем только области, где маска активна
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                masked_color = cv2.bitwise_and(color_mask, mask_3ch)
                return cv2.addWeighted(frame, 0.7, masked_color, 0.3, 0)

        return frame.copy()

    def sort_hands_by_position(self, hands, bboxes):
        """Сортировка рук слева направо."""
        if len(bboxes) >= 2:
            hand_bbox_pairs = sorted(zip(hands, bboxes), key=lambda pair: pair[1][0])
            return zip(*hand_bbox_pairs)
        return hands, bboxes

    def process_hands(self, hands, bboxes, stable_gestures, stable_display,
                      stable_threshold, conf_threshold, frame):
        """Обработка рук и отображение предсказаний."""
        for i, (hand, bbox) in enumerate(zip(hands, bboxes)):
            x, y, w, h = bbox
            color = (0, 255, 0) if i == 0 else (255, 0, 0)

            probs = self.model.predict_proba(hand.reshape(1, -1))[0]
            gesture = np.argmax(probs)
            confidence = probs[gesture]

            stable_gestures[i].append(gesture)
            if len(stable_gestures[i]) > stable_threshold:
                stable_gestures[i].pop(0)

            most_common = max(set(stable_gestures[i]), key=stable_gestures[i].count)
            count = stable_gestures[i].count(most_common)

            if count == stable_threshold or confidence > conf_threshold:
                stable_display[i] = most_common

            if stable_display[i] is not None:
                gesture_name = self.gesture_names[stable_display[i]]
                cv2.putText(frame, f"Player {i + 1}: {gesture_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            for idx, line in enumerate([f"{name}: {p * 100:.1f}%" for name, p in zip(self.gesture_names, probs)]):
                cv2.putText(frame, line, (x, y + 20 + idx * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        return stable_display

    def update_image(self, frame, target_label):
        """Обновляет изображение на метке, если она ещё существует."""
        if not target_label.winfo_exists():
            return  # метка уничтожена, не обновляем
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        w = target_label.winfo_width()
        h = target_label.winfo_height()

        if w > 0 and h > 0:
            scale = min(w / pil_img.width, h / pil_img.height)
            pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)

        tk_image = ImageTk.PhotoImage(pil_img)
        target_label.configure(image=tk_image)
        target_label.image = tk_image


