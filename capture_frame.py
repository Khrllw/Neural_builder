import importlib

import cv2
import os
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
from tkinter import filedialog

import config
from config import LOWER_SKIN, UPPER_SKIN


class CaptureFrame(ctk.CTkFrame):
    """
    Виджет для захвата изображений с камеры и сохранения ROI (области интереса) в заданную директорию.
    """

    def __init__(self, parent, on_done=None):
        super().__init__(parent)
        self.on_done = on_done

        self.image_size = 250
        self.roi_x, self.roi_y = 100, 100

        self.collecting = False
        self.saved_count = 0
        self.num_images = 0
        self.class_name = ""
        self.output_dir = None
        importlib.reload(config)

        self.slider_values = {
            "y_min": LOWER_SKIN[0],
            "y_max": UPPER_SKIN[0],
            "cr_min": LOWER_SKIN[1],
            "cr_max": UPPER_SKIN[1],
            "cb_min": LOWER_SKIN[2],
            "cb_max": UPPER_SKIN[2]
        }

        self.setup_ui()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Не удалось открыть камеру")
        else:
            self.status_label.configure(text="Камера запущена")
            self.update_frame()

    def create_slider_block(self, parent, label_text, slider, pady=(15, 0)):
        """Создание блока с заголовком и слайдером."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=pady)

        label = ctk.CTkLabel(frame, text=label_text, anchor="w")
        label.pack(fill="x", padx=10)

        slider.pack(fill="x", padx=10)
        return frame

    def get_skin_mask(self, frame):
        """Преобразует изображение в маску кожи на основе YCrCb порогов."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([
            self.slider_values["y_min"],
            self.slider_values["cr_min"],
            self.slider_values["cb_min"]
        ], dtype=np.uint8)
        upper = np.array([
            self.slider_values["y_max"],
            self.slider_values["cr_max"],
            self.slider_values["cb_max"]
        ], dtype=np.uint8)
        return cv2.inRange(ycrcb, lower, upper)

    def select_directory(self):
        """Открывает диалог выбора директории и сохраняет путь."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.selected_dir_label.configure(
                text=f"Директория: {self.output_dir}", text_color="white")
            self.start_button.configure(state="normal")
        else:
            self.output_dir = None
            self.selected_dir_label.configure(
                text="Директория не выбрана", text_color="gray")
            self.start_button.configure(state="disabled")

    def update_roi_x(self, val):
        self.roi_x = int(val)
        self.roi_x_value_label.configure(text=str(self.roi_x))

    def update_roi_y(self, val):
        self.roi_y = int(val)
        self.roi_y_value_label.configure(text=str(self.roi_y))

    def update_roi_size(self, value):
        self.image_size = int(value)
        self.roi_size_value_label.configure(text=str(self.image_size))

    def resize_image_to_fit(self, img, container):
        """Подгоняет изображение под размеры контейнера."""
        container.update_idletasks()
        w, h = container.winfo_width(), container.winfo_height()

        if w <= 1 or h <= 1:
            return img

        h_img, w_img = img.shape[:2]
        scale = min(w / w_img, h / h_img)
        new_w, new_h = int(w_img * scale), int(h_img * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
        x_offset, y_offset = (w - new_w) // 2, (h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def start_capture(self):
        """Инициализирует процесс сбора изображений."""
        self.class_name = self.class_entry.get().strip()
        try:
            self.num_images = int(self.num_entry.get())
        except Exception:
            self.status_label.configure(text="Ошибка: введите корректное количество")
            return

        if not self.class_name:
            self.status_label.configure(text="Ошибка: введите имя класса")
            return
        if not self.output_dir:
            self.status_label.configure(text="Ошибка: выберите директорию")
            return

        self.class_dir = os.path.join(self.output_dir, self.class_name)
        os.makedirs(self.class_dir, exist_ok=True)

        self.saved_count = 0
        self.collecting = True

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="Сбор изображений начат")

    def stop_capture(self):
        """Останавливает сбор изображений."""
        self.collecting = False
        self.status_label.configure(
            text=f"Сбор остановлен. Сохранено {self.saved_count} изображений.")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_bar.set(0)

    def update_frame(self):
        """Обновляет видеокадр, рисует ROI, маску и сохраняет изображения."""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.configure(text="Ошибка захвата кадра")
            self.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        roi_x = min(max(self.roi_x, 0), w_frame - self.image_size)
        roi_y = min(max(self.roi_y, 0), h_frame - self.image_size)
        roi = frame[roi_y:roi_y + self.image_size, roi_x:roi_x + self.image_size].copy()

        cv2.rectangle(frame, (roi_x, roi_y),
                      (roi_x + self.image_size, roi_y + self.image_size),
                      (0, 255, 0), 2)

        mask = self.get_skin_mask(frame)

        progress = self.saved_count / self.num_images if self.num_images else 0
        self.progress_bar.set(progress)
        self.progress_percent_label.configure(text=f"{int(progress * 100)}%")

        resized_color = self.resize_image_to_fit(frame, self.color_image_label)

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_color, (roi_x, roi_y),
                      (roi_x + self.image_size, roi_y + self.image_size),
                      (0, 0, 255), 2)
        resized_mask = self.resize_image_to_fit(mask_color, self.mask_image_label)

        self.show_image_in_label(resized_color, self.color_image_label)
        self.show_image_in_label(resized_mask, self.mask_image_label)

        if self.collecting and self.saved_count < self.num_images:
            img_path = os.path.join(self.class_dir, f"img_{self.saved_count + 1:04}.jpg")
            success = cv2.imwrite(img_path, roi)
            if success:
                self.saved_count += 1
                self.status_label.configure(
                    text=f"Сохраняю: {self.saved_count} / {self.num_images}")
                progress = self.saved_count / self.num_images
                self.progress_bar.set(progress)
                self.progress_percent_label.configure(text=f"{int(progress * 100)}%")

            if self.saved_count >= self.num_images:
                self.finish_capture()

        self.after(30, self.update_frame)

    def finish_capture(self):
        """Завершение процесса сбора данных."""
        self.status_label.configure(text="Сбор завершён")
        self.collecting = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_bar.set(1.0)
        self.progress_percent_label.configure(text="100%")
        if callable(self.on_done):
            self.on_done()

    def show_image_in_label(self, img, label):
        """Отображает изображение в Label виджете."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = self.round_corners(img_pil, 10)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.imgtk = img_tk  # сохранение ссылки на изображение
        label.configure(image=img_tk, text="")

    @staticmethod
    def round_corners(im: Image.Image, radius: int):
        """Округляет углы изображения."""
        mask = Image.new("L", im.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, im.width, im.height), radius=radius, fill=255)
        im = im.convert("RGBA")
        im.putalpha(mask)
        return im

    def on_close(self):
        """Закрывает доступ к камере."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def setup_ui(self):

        # Сетка для фрейма
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.configure(corner_radius=10)

        # --- Панель управления (справа) ---
        self.control_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=(10, 15), pady=10)
        self.control_frame.grid_rowconfigure(20, weight=1)

        self.grid_columnconfigure(1, weight=0)
        self.control_frame.configure(width=300)  # минимальная ширина колонки с меню справа

        # Скрываем скроллбар, но оставляем прокрутку
        scrollbar = self.control_frame._scrollbar
        scrollbar.configure(width=0)  # ширина 0 — не виден

        # Заголовок
        title_label = ctk.CTkLabel(self.control_frame, text="Сбор изображений с камеры",
                                   font=ctk.CTkFont(size=18, weight="bold"),
                                   justify="left",
                                   anchor="w"
                                   )
        title_label.pack(fill="x", pady=(10, 0))

        # --- Инструкция пользователю ---
        instruction_label = ctk.CTkLabel(
            self.control_frame,
            text=(
                "Внизу отображается ЧБ маска — это результат обработки кадра.\n"
                "Если маска некорректна, настройте параметры во вкладке SkinMask.\n\n"
                "Правила:\n"
                "• Кожа — белая\n"
                "• Фон  — чёрный, желательно без шумов\n"
                "• Рука — полностью в рамке\n\n"
                "Изображение в рамке сохраняется в оригинальном формате.\n"
            ),
            wraplength=260,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=14, weight="normal")
        )
        instruction_label.pack(padx=0, pady=(10, 0), fill="x")

        # Ввод имени класса
        self.class_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Введите имя класса")
        self.class_entry.pack(pady=5, fill="x")

        # Ввод количества
        self.num_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Количество изображений")
        self.num_entry.pack(pady=5, fill="x")

        # Лейбл выбранной директории
        self.selected_dir_label = ctk.CTkLabel(
            self.control_frame,
            text="Директория не выбрана",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
            justify="left",
            wraplength=220
        )
        self.selected_dir_label.pack(pady=5, fill="x")

        # Кнопка выбора директории
        self.select_dir_button = ctk.CTkButton(self.control_frame, text="Выбрать директорию для сохранения",
                                               command=self.select_directory)
        self.select_dir_button.pack(pady=10, fill="x")

        # ROI X
        roi_x_label_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        roi_x_label_frame.pack(fill="x", padx=10, pady=(15, 0))

        ctk.CTkLabel(roi_x_label_frame, text="Положение ROI по X").pack(side="left")
        self.roi_x_value_label = ctk.CTkLabel(roi_x_label_frame, text=str(self.roi_x))
        self.roi_x_value_label.pack(side="right")

        self.roi_x_slider = ctk.CTkSlider(self.control_frame, from_=0, to=400, number_of_steps=400,
                                          command=self.update_roi_x)
        self.roi_x_slider.set(self.roi_x)
        self.roi_x_slider.pack(fill="x", padx=10)

        # ROI Y
        roi_y_label_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        roi_y_label_frame.pack(fill="x", padx=10, pady=(15, 0))

        ctk.CTkLabel(roi_y_label_frame, text="Положение ROI по Y").pack(side="left")
        self.roi_y_value_label = ctk.CTkLabel(roi_y_label_frame, text=str(self.roi_y))
        self.roi_y_value_label.pack(side="right")

        self.roi_y_slider = ctk.CTkSlider(self.control_frame, from_=0, to=400, number_of_steps=400,
                                          command=self.update_roi_y)
        self.roi_y_slider.set(self.roi_y)
        self.roi_y_slider.pack(fill="x", padx=10)

        # ROI Size
        roi_size_label_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        roi_size_label_frame.pack(fill="x", padx=10, pady=(15, 0))

        ctk.CTkLabel(roi_size_label_frame, text="Размер ROI").pack(side="left")
        self.roi_size_value_label = ctk.CTkLabel(roi_size_label_frame, text=str(self.image_size))
        self.roi_size_value_label.pack(side="right")

        self.image_size_slider = ctk.CTkSlider(self.control_frame, from_=0, to=400, number_of_steps=400,
                                               command=self.update_roi_size)
        self.image_size_slider.set(self.image_size)
        self.image_size_slider.pack(fill="x", padx=10)

        # Кнопки старт/стоп
        self.start_button = ctk.CTkButton(self.control_frame, text="Начать сбор", command=self.start_capture,
                                          state="disabled")
        self.start_button.pack(pady=(10, 5), fill="x")

        self.stop_button = ctk.CTkButton(self.control_frame, text="Остановить сбор", command=self.stop_capture,
                                         state="disabled")
        self.stop_button.pack(pady=5, fill="x")

        # --- Панель с видео (слева) ---
        self.image_container = ctk.CTkFrame(self, fg_color="transparent")
        self.image_container.grid(row=0, column=0, sticky="nsew", padx=(15, 0), pady=5)
        self.image_container.grid_rowconfigure(0, weight=0)  # для статуса камеры
        self.image_container.grid_rowconfigure(1, weight=0)  # для инструкции
        self.image_container.grid_rowconfigure(2, weight=1)  # для видео
        self.image_container.grid_rowconfigure(3, weight=1)  # для прогресс бара
        self.image_container.grid_columnconfigure(0, weight=1)

        # Создаем Frame для статусов, прикрепляем его к правому краю image_container
        status_frame = ctk.CTkFrame(self.image_container, fg_color="transparent")
        status_frame.grid(row=0, column=0, sticky="we", padx=5, pady=(5, 0))

        # Статус сбора данных — рядом с камерой
        self.status_label = ctk.CTkLabel(status_frame, text="Готов к сбору данных")
        self.status_label.pack(side="left", padx=0, pady=0)

        # Цветное видео (сверху)
        self.color_frame = ctk.CTkFrame(self.image_container, fg_color="#3A3A3A", corner_radius=10)
        self.color_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.color_frame.grid_rowconfigure(1, weight=1)
        self.color_frame.grid_columnconfigure(0, weight=1)

        color_label = ctk.CTkLabel(self.color_frame, text="Исходное изображение",
                                   font=ctk.CTkFont(size=14, weight="bold"))
        color_label.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        self.color_image_label = ctk.CTkLabel(self.color_frame)
        self.color_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))

        # Чёрно-белое видео (маска) (снизу)
        self.mask_frame = ctk.CTkFrame(self.image_container, fg_color="#3A3A3A", corner_radius=10)
        self.mask_frame.grid(row=3, column=0, sticky="nsew")
        self.mask_frame.grid_rowconfigure(1, weight=1)
        self.mask_frame.grid_columnconfigure(0, weight=1)

        mask_label = ctk.CTkLabel(self.mask_frame, text="ЧБ Маска",
                                  font=ctk.CTkFont(size=14, weight="bold"))
        mask_label.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        self.mask_image_label = ctk.CTkLabel(self.mask_frame)
        self.mask_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))

        # Контейнер для прогресс-бара и процента
        self.progress_container = ctk.CTkFrame(self.image_container, fg_color="transparent")
        self.progress_container.grid(row=4, column=0, sticky="ew", padx=0, pady=(5, 15))
        self.progress_container.grid_columnconfigure(0, weight=1)  # прогресс-бар растягивается
        self.progress_container.grid_columnconfigure(1, weight=0)  # метка справа

        # Прогресс-бар
        self.progress_bar = ctk.CTkProgressBar(self.progress_container)
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        # Метка с процентами справа от прогресс-бара
        self.progress_percent_label = ctk.CTkLabel(self.progress_container, text="0%", width=40, anchor="e")
        self.progress_percent_label.grid(row=0, column=1, sticky="e", padx=(10, 0))