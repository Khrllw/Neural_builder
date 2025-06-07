import importlib

import customtkinter as ctk
import config

CONFIG_FILE = "config.py"


class SettingsFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # ----------------- Левый фрейм -----------------
        self.desc_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.desc_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.desc_frame.columnconfigure(0, weight=1)
        self.desc_frame.rowconfigure(1, weight=1)
        self.desc_frame.rowconfigure(3, weight=1)

        warning_text = (
            "⚠️ Внимание! Редактировать конфиг нужно очень внимательно!"
        )
        self.warning_label = ctk.CTkLabel(
            self.desc_frame,
            text=warning_text,
            text_color="red",
            font=ctk.CTkFont(weight="bold"),
        )
        self.warning_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.desc_text = ctk.CTkTextbox(
            self.desc_frame, state="disabled", wrap="word", corner_radius=10, height=280
        )
        self.desc_text.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        self.desc_nn_text = ctk.CTkTextbox(
            self.desc_frame, state="disabled", wrap="word", corner_radius=10, height=280
        )
        self.desc_nn_text.grid(row=3, column=0, sticky="nsew")

        self._fill_description()
        self._fill_nn_description()

        # ----------------- Правый фрейм (редактирование всего файла) -----------------
        self.edit_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.edit_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.edit_frame.columnconfigure(0, weight=1)
        self.edit_frame.rowconfigure(0, weight=1)


        self.textbox = ctk.CTkTextbox(
            self.edit_frame, corner_radius=10, font=ctk.CTkFont(size=14)
        )
        self.textbox.grid(row=0, column=0, sticky="nsew")

        self.save_btn = ctk.CTkButton(
            self.edit_frame, text="Сохранить конфигурацию", command=self._save_config
        )
        self.save_btn.grid(row=1, column=0, sticky="e", pady=(5, 0))

        self._load_config_text()
        # Убираем вызов подсветки, она не поддерживается в CTkTextbox
        # self._highlight_comments()

    def _fill_description(self):
        importlib.reload(config)
        desc = (
            "Общие параметры конфигурации:\n\n"
            "- LOWER_SKIN, UPPER_SKIN — диапазон цвета кожи в формате numpy array.\n"
            "- MASK_COLOR — цвет маски для обработки изображения.\n"
            "- IMG_SIZE — размер входного изображения (ширина, высота).\n"
            "- PADDING — отступы вокруг изображения.\n"
            "- IMAGE_EXTENSIONS — допустимые расширения файлов изображений.\n"
            "- CAMERA_INDEX — индекс используемой камеры.\n"
            "- CAPTURE_INSTRUCTION — инструкция для пользователя при сборе данных.\n"
            "- DATASET_SUBFOLDERS — подпапки для train/test/validation.\n"
            "- DEFAULT_DATA_DIR — папка с исходными данными.\n"
            "- SAMPLES_PER_CLASS — количество примеров на класс.\n"
            "- NUM_CLASSES — количество классов для классификации.\n"
            "- AUGMENTATION_SETTINGS — настройки аугментации изображений (флип, повороты, шум и др.)."
        )
        self.desc_text.configure(state="normal")
        self.desc_text.delete("0.0", "end")
        self.desc_text.insert("0.0", desc)
        self.desc_text.configure(state="disabled")

    def _fill_nn_description(self):
        importlib.reload(config)
        nn_desc = (
            "Параметры обучения нейросети:\n\n"
            "- INPUT_SIZE — размер входного слоя (число пикселей в изображении).\n"
            "- HIDDEN_SIZES — список размеров скрытых слоев.\n"
            "- OUTPUT_SIZE — количество выходных классов.\n"
            "- LEARNING_RATE — скорость обучения.\n"
            "- REG_LAMBDA — коэффициент L2-регуляризации.\n"
            "- MOMENTUM — параметр ускорения сходимости.\n"
            "- TARGET_ACCURACY — целевая точность обучения (в %).\n"
            "- PATIENCE — количество эпох без улучшения для ранней остановки.\n"
            "- MIN_DELTA — минимальное улучшение для учёта прогресса.\n"
            "- LR_DECAY_EPOCHS — частота снижения learning rate (эпохи).\n"
            "- LR_DECAY_FACTOR — коэффициент снижения learning rate.\n"
            "- EPOCHS — максимальное количество эпох обучения.\n"
            "- BATCH_SIZE — размер мини-батча.\n"
            "- VALIDATION_SPLIT — доля данных для валидации.\n"
            "- SEED — фиксированное зерно генератора случайных чисел."
        )
        self.desc_nn_text.configure(state="normal")
        self.desc_nn_text.delete("0.0", "end")
        self.desc_nn_text.insert("0.0", nn_desc)
        self.desc_nn_text.configure(state="disabled")

    def _load_config_text(self):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", content)
        except FileNotFoundError:
            self.textbox.insert("0.0", "# Файл config.py не найден.")

    # Метод подсветки удалён, т.к. CTkTextbox не поддерживает теги и стили

    def _save_config(self):
        content = self.textbox.get("0.0", "end").rstrip()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(content)
            print("Конфигурация успешно сохранена.")
            # self._highlight_comments()  # Вызывать не нужно
        except Exception as e:
            print("Ошибка при сохранении конфигурации:", e)

    def on_close(self):
        pass
