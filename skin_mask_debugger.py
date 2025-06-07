import importlib
from tkinter import colorchooser
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

import config
from config import MASK_COLOR, LOWER_SKIN, UPPER_SKIN
from image_utils import save_skin_range_to_config
from popup import SuccessPopup


class SkinMaskFrame(ctk.CTkFrame):
    MASK_BG_COLOR = (30, 30, 30)
    FRAME_REFRESH_DELAY = 30  # ms
    ROUND_RADIUS = 10


    def __init__(self, parent):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0)
        self.trackbars = {}
        self.mask_color, self.lower_skin, self.upper_skin = self.load_skin_config()

        self.setup_ui()
        self.update_frame()

    # ---------- FRAME UPDATE LOGIC ----------
    def update_frame(self):
        """Считывает кадр, применяет маску, обновляет GUI."""
        frame = self.read_and_flip_frame()
        if frame is None:
            self.after(self.FRAME_REFRESH_DELAY, self.update_frame)
            return

        mask = self.get_skin_mask(frame)
        blended = self.create_blended_overlay(frame, mask, color=self.mask_color)

        top_img = self.resize_image_to_fit(blended, self.top_image_label)
        bottom_left = self.resize_image_to_fit(frame, self.bottom_left_image_label)
        bottom_right = self.resize_image_to_fit(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), self.bottom_right_image_label)

        self.show_image_in_label(top_img, self.top_image_label)
        self.show_image_in_label(bottom_left, self.bottom_left_image_label)
        self.show_image_in_label(bottom_right, self.bottom_right_image_label)

        self.after(self.FRAME_REFRESH_DELAY, self.update_frame)

    def read_and_flip_frame(self):
        """Считывает и отражает кадр с камеры."""
        ret, frame = self.cap.read()
        return cv2.flip(frame, 1) if ret else None

    def get_skin_mask(self, frame):
        """Возвращает бинарную маску кожи на основе YCrCb."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([self.trackbars["Y Min"].get(),
                          self.trackbars["Cr Min"].get(),
                          self.trackbars["Cb Min"].get()], dtype=np.uint8)
        upper = np.array([self.trackbars["Y Max"].get(),
                          self.trackbars["Cr Max"].get(),
                          self.trackbars["Cb Max"].get()], dtype=np.uint8)
        return cv2.inRange(ycrcb, lower, upper)

    @staticmethod
    def create_blended_overlay(frame, mask, color=(0, 0, 255)):
        """Накладывает маску с цветом на изображение."""
        overlay = frame.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # ---------- IMAGE PROCESSING ----------
    def resize_image_to_fit(self, img, container):
        """Масштабирует изображение под размер контейнера."""
        container.update_idletasks()
        w, h = container.winfo_width(), container.winfo_height()

        if w <= 1 or h <= 1:
            return img

        scale = min(w / img.shape[1], h / img.shape[0])
        new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((h, w, 3), self.MASK_BG_COLOR, dtype=np.uint8)
        x_offset, y_offset = (w - new_w) // 2, (h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def show_image_in_label(self, img, label):
        """Преобразует и вставляет OpenCV изображение в Label."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = self.round_corners(img_pil, self.ROUND_RADIUS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.imgtk = img_tk
        label.configure(image=img_tk, text="")

    @staticmethod
    def round_corners(im: Image.Image, radius: int):
        """Возвращает изображение с округленными углами."""
        mask = Image.new("L", im.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, im.width, im.height), radius=radius, fill=255)
        im = im.convert("RGBA")
        im.putalpha(mask)
        return im

    # ---------- USER INTERACTION ----------
    def choose_mask_color(self):
        """Открывает выбор цвета и сохраняет как BGR."""
        initial_rgb = self.mask_color[::-1]  # BGR → RGB
        color = colorchooser.askcolor(initialcolor=initial_rgb)
        if color[0] is not None:
            r, g, b = map(int, color[0])
            self.mask_color = (b, g, r)
            self.update_color_label()

    def update_color_label(self):
        """Обновляет фоновый цветной индикатор."""
        r, g, b = self.mask_color[::-1]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_label.configure(fg_color=hex_color)

    def save(self):
        """Сохраняет параметры маски в конфиг и показывает уведомление."""
        lower = np.array([
            self.trackbars["Y Min"].get(),
            self.trackbars["Cr Min"].get(),
            self.trackbars["Cb Min"].get()
        ], dtype=np.uint8)
        upper = np.array([
            self.trackbars["Y Max"].get(),
            self.trackbars["Cr Max"].get(),
            self.trackbars["Cb Max"].get()
        ], dtype=np.uint8)

        save_skin_range_to_config(self.mask_color, lower, upper, "config.py")
        SuccessPopup(self)

    # ---------- UI CREATION ----------
    def create_slider(self, label, from_, to, initial):
        """Создает слайдер и подписывает его."""
        frame = ctk.CTkFrame(self.control_frame)
        frame.pack(fill="x", pady=(10, 0), padx=10)

        ctk.CTkLabel(frame, text=label, width=70, anchor="w").pack(side="left")

        var = ctk.IntVar(value=initial)
        slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=to - from_, variable=var, width=150)
        slider.pack(side="left", padx=5)

        value_label = ctk.CTkLabel(frame, text=str(initial), width=30, anchor="e")
        value_label.pack(side="left")

        slider.configure(command=lambda v: value_label.configure(text=str(int(float(v)))))
        self.trackbars[label] = var

    def create_label_with_image(self, parent, title):
        """Создает контейнер с заголовком и Label для изображения."""
        frame = ctk.CTkFrame(parent, fg_color="#3A3A3A", corner_radius=10)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="ew", pady=5, padx=5)

        label = ctk.CTkLabel(frame)
        label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))
        return frame, label

    def on_close(self):
        """Закрывает доступ к камере."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def setup_ui(self):
        """Создает и размещает все виджеты интерфейса."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=0)
        self.configure(corner_radius=10)

        # Контейнер изображений
        self.image_container = ctk.CTkFrame(self, fg_color="transparent")
        self.image_container.grid(row=0, column=0, sticky="nsew", padx=(15, 0), pady=10)
        self.image_container.grid_rowconfigure(0, weight=3)
        self.image_container.grid_rowconfigure(1, weight=1)
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_columnconfigure(1, weight=1)

        # Верхнее изображение
        self.top_frame, self.top_image_label = self.create_label_with_image(self.image_container, "Маска")
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # Нижние изображения
        self.bottom_left_frame, self.bottom_left_image_label = self.create_label_with_image(
            self.image_container, "Исходное изображение")
        self.bottom_left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.bottom_right_frame, self.bottom_right_image_label = self.create_label_with_image(
            self.image_container, "ЧБ Отображение")
        self.bottom_right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # Панель управления
        self.control_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", width=300)
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=(10, 15), pady=10)
        self.control_frame.grid_rowconfigure(20, weight=1)
        self.control_frame._scrollbar.configure(width=0)

        # Заголовок и описание
        ctk.CTkLabel(self.control_frame, text="Настройка маски",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     anchor="w").pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(self.control_frame, text=(
            "Настройте ползунки так, чтобы маска покрывала только кожу, "
            "при этом не захватывая лишние области. Чем точнее настройки, "
            "тем лучше результат распознавания."),
                     wraplength=260, justify="left",
                     font=ctk.CTkFont(size=14)).pack(pady=(10, 15), fill="x")

        # Выбор цвета
        self.color_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.color_frame.pack(pady=10, padx=(0, 15), fill="x")

        self.color_label = ctk.CTkLabel(self.color_frame, text="", width=40, height=26, corner_radius=6)
        self.color_label.pack(side="left", padx=(0, 10))
        self.update_color_label()

        self.color_button = ctk.CTkButton(self.color_frame, text="Изменить цвет", command=self.choose_mask_color)
        self.color_button.pack(fill="x", expand=True)

        # Слайдеры YCrCb
        slider_values = [
            ("Y Min", self.lower_skin[0]), ("Y Max", self.upper_skin[0]),
            ("Cr Min", self.lower_skin[1]), ("Cr Max", self.upper_skin[1]),
            ("Cb Min", self.lower_skin[2]), ("Cb Max", self.upper_skin[2]),
        ]
        for label, val in slider_values:
            self.create_slider(label, 0, 255, val)

        # Кнопка сохранения
        ctk.CTkButton(self.control_frame, text="Сохранить", command=self.save).pack(pady=20, fill="x")

    @staticmethod
    def load_skin_config():
        importlib.reload(config)  # перезагрузить модуль, чтобы подтянуть актуальные значения
        return config.MASK_COLOR, config.LOWER_SKIN, config.UPPER_SKIN
