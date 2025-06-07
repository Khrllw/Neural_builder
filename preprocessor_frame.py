import os
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import cv2

import config
from image_proc import ImageProcessor
from config import DATASET_SUBFOLDERS


class PreprocessorFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.dataset_path_var = tk.StringVar()
        self.save_path_var = tk.StringVar(value="processed")
        self.img_width = tk.IntVar(value=config.IMG_SIZE[0])
        self.img_height = tk.IntVar(value=config.IMG_SIZE[1])

        # Настройки основного фрейма
        self.configure(corner_radius=10)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)  # левая колонка (лог) — 3 части
        self.grid_columnconfigure(1, weight=1)  # правая колонка (панель управления) — 1 часть

        self.build_ui()

    def build_ui(self):
        # === Настройки родительского фрейма ===
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)  # Левая колонка растягивается
        self.grid_columnconfigure(1, weight=0)  # Правая колонка фиксирована

        # === Левая часть (лог) ===
        self.log_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.log_frame.grid(row=0, column=0, sticky="nsew", padx=(15, 0), pady=5)

        self.log_frame.grid_rowconfigure(1, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)

        log_label = ctk.CTkLabel(self.log_frame, text="Логи обработки")
        log_label.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 0))

        self.log_textbox = ctk.CTkTextbox(self.log_frame)
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # === Правая часть (панель управления) с фиксированной шириной 300 ===
        # --- Панель управления (справа) ---
        self.control_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", width=300)
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=(10, 15), pady=10)
        # запретить изменение размеров

        self.control_frame.grid_rowconfigure(20, weight=1)

        # Скрываем скроллбар, но оставляем прокрутку
        scrollbar = self.control_frame._scrollbar
        scrollbar.configure(width=0)  # ширина 0 — не виден

        # Заголовок панели управления
        title_label = ctk.CTkLabel(
            self.control_frame,
            text="Обработка датасета",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
            justify="left"
        )
        title_label.pack(fill="x", padx=0, pady=(10, 0))

        # Инструкция
        instruction_label = ctk.CTkLabel(
            self.control_frame,
            text=(
                "Выберите исходную директорию с датасетом и директорию, в которую будет "
                "сохранён результат обработки.\n\n"
                "Изображения будут изменены до заданного размера, отфильтрованы и "
                "сохранены по классам.\n\n"
                "Параметры обработки находятся в конфигурации.\n\n"
                "⚠ Структура загружаемого датасета должна быть следующей:\n"
                "/путь/к/датасету/\n"
                "├── train/\n"
                "│   ├── класс_1/\n"
                "│   └── класс_2/\n"
                "└── test/\n"
                "       ├── класс_1/\n"
                "       └── класс_2/"
            ),
            wraplength=260,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=14)
        )
        instruction_label.pack(fill="x", padx=0, pady=(5, 15))

        # === Директория с датасетом ===
        dataset_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        dataset_frame.pack(fill="x", pady=5, padx=5)

        ctk.CTkLabel(dataset_frame, text="Директория с датасетом:").pack(anchor="w", pady=(0, 2))
        self.dataset_path_label = ctk.CTkLabel(
            dataset_frame,
            text="Директория не выбрана",
            font=ctk.CTkFont(size=12),
            wraplength=260,
            anchor="w",
            justify="left",
            text_color="gray"  # начальный цвет (серый, если пусто)
        )
        self.dataset_path_label.pack(fill="x")

        ctk.CTkButton(dataset_frame, text="Выбрать", command=self.browse_dataset_folder).pack(fill="x", pady=(5, 5))

        # === Директория для сохранения ===
        save_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        save_frame.pack(fill="x", pady=5, padx=5)

        ctk.CTkLabel(save_frame, text="Директория для сохранения:").pack(anchor="w", pady=(0, 2))
        self.save_path_label = ctk.CTkLabel(
            save_frame,
            textvariable=self.save_path_var,
            font=ctk.CTkFont(size=12),
            wraplength=260,
            anchor="w",
            justify="left",
            text_color="gray"# начальный цвет (серый, если пусто)
        )
        self.save_path_label.pack(fill="x")

        ctk.CTkButton(save_frame, text="Выбрать", command=self.browse_save_folder).pack(fill="x", pady=(5, 5))
        # === Размер изображений ===
        size_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        size_frame.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(size_frame, text="Размер изображения:").pack(anchor="w")
        row = ctk.CTkFrame(size_frame)
        row.pack(anchor="w")

        ctk.CTkEntry(row, textvariable=self.img_width, width=60).pack(side="left")
        ctk.CTkLabel(row, text="x").pack(side="left", padx=5)
        ctk.CTkEntry(row, textvariable=self.img_height, width=60).pack(side="left")
        # === Кнопка запуска ===
        ctk.CTkButton(
            self.control_frame,
            text="Запустить обработку",
            command=self.run_preprocessing
        ).pack(pady=15, padx=5, fill="x")

    def browse_dataset_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path_var.set(folder)  # если нужен для других целей
            self.dataset_path_label.configure(text=folder)
            self.update_label_color(self.dataset_path_label, chosen=True)
        else:
            self.dataset_path_label.configure(text="Директория не выбрана")
            self.update_label_color(self.dataset_path_label, chosen=False)

    def browse_save_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_path_var.set(folder)
            self.save_path_label.configure(text=folder)
            self.update_label_color(self.save_path_label, chosen=True)
        else:
            self.save_path_label.configure(text="Директория не выбрана")
            self.update_label_color(self.save_path_label, chosen=False)

    def update_label_color(self, label, chosen: bool):
        if chosen:
            label.configure(text_color="white")  # или другой цвет для выбранного пути
        else:
            label.configure(text_color="gray")  # серый цвет для пустого пути

    def log(self, message: str):
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")

    def run_preprocessing(self):
        dataset_path = self.dataset_path_var.get()
        save_path = self.save_path_var.get()
        img_size = (self.img_width.get(), self.img_height.get())

        if not os.path.isdir(dataset_path):
            self.log("❌ Указанная директория датасета не существует.")
            return

        self.log(
            f"\n▶ Запуск предобработки...\n📂 Источник: {dataset_path}\n💾 Сохранение: {save_path}\n📐 Размер: {img_size}")
        self.after(100, lambda: self.preprocess_and_save_dataset(dataset_path, save_path, img_size))

    def preprocess_and_save_dataset(self, dataset_path: str, save_dir: str = "processed", img_size=(224, 224)):
        self.log(f"\n🔄 Обработка из {dataset_path} в {save_dir}...")

        for subset in DATASET_SUBFOLDERS:
            subset_dir = os.path.join(dataset_path, subset)

            if not os.path.exists(subset_dir):
                self.log(f"⚠ Подпапка {subset} не найдена. Пропуск.")
                continue

            class_dirs = [d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))]
            for class_name in class_dirs:
                class_path = os.path.join(subset_dir, class_name)
                output_class_dir = os.path.join(save_dir, subset, class_name)
                os.makedirs(output_class_dir, exist_ok=True)

                for filename in os.listdir(class_path):
                    if not filename.lower().endswith(config.IMAGE_EXTENSIONS):
                        continue

                    input_path = os.path.join(class_path, filename)
                    img = cv2.imread(input_path)

                    if img is None:
                        self.log(f"⚠ Ошибка чтения файла: {filename}. Пропущен.")
                        continue

                    processed, _, _ = ImageProcessor.preprocess_image(
                        img,
                        size=img_size,
                        save_dir=output_class_dir,
                        base_filename=os.path.splitext(filename)[0]
                    )

                    if processed:
                        self.log(f"✅ {os.path.join(output_class_dir, filename)}")

        self.log("🎉 Обработка завершена.\n")

    def on_close(self):
        pass

