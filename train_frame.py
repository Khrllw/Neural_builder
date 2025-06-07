import importlib
import os

import customtkinter as ctk
from tkinter import filedialog
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
from image_proc import ImageProcessor
from neural_network import NeuralNetwork


class TrainFrame(ctk.CTkFrame):
    import config  # Импортируй свой config.py где описаны настройки
    def __init__(self, parent):
        super().__init__(parent)
        self.stop_training = False
        self.model = NeuralNetwork()
        self.dataset_path = ""
        self.is_ready_dataset = ctk.BooleanVar(value=False)
        self.training_thread = None

        self.loss_values = []
        self.accuracy_values = []

        # === Сетка ===
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # === Правая панель ===
        self.control_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", width=300)
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=(10, 15), pady=10)
        self.control_frame.grid_rowconfigure(20, weight=1)

        scrollbar = self.control_frame._scrollbar
        scrollbar.configure(width=0)  # скрыть

        ctk.CTkLabel(self.control_frame, text="Обучение нейросети",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 15), padx=10)
        importlib.reload(config)
        params_text = (
            f"Параметры обучения:\n"
            f"LR: {config.NeuralNetConfig.LEARNING_RATE}\n"
            f"Epochs: {config.NeuralNetConfig.EPOCHS}\n"
            f"Batch size: {config.NeuralNetConfig.BATCH_SIZE}\n"
            f"Momentum: {config.NeuralNetConfig.MOMENTUM}\n"
            f"Регуляризация: {config.NeuralNetConfig.REG_LAMBDA}\n"
            f"Целевая точность: {config.NeuralNetConfig.TARGET_ACCURACY}%\n\n"
            f"Аугментация:\n"
            f"Flip: {'Вкл' if config.AUGMENTATION_SETTINGS['flip'] else 'Выкл'}\n"
            f"Rotate angles: {config.AUGMENTATION_SETTINGS['rotate_angles']}\n"
            f"Shift offsets: {config.AUGMENTATION_SETTINGS['shift_offsets']}\n"
            f"Add noise: {'Вкл' if config.AUGMENTATION_SETTINGS['add_noise'] else 'Выкл'}"
        )
        ctk.CTkLabel(self.control_frame, text=params_text, justify="left", wraplength=300).pack(anchor="w", padx=10,
                                                                                                pady=(0, 15))

        ctk.CTkLabel(self.control_frame, text="Директория с датасетом:").pack(anchor="w", padx=10)
        self.dataset_label = ctk.CTkLabel(self.control_frame, text="Директория не выбрана", text_color="gray",
                                          wraplength=280, anchor="w", justify="left")
        self.dataset_label.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkButton(self.control_frame, text="Выбрать...", command=self.choose_dataset_dir).pack(fill="x", padx=10,
                                                                                                   pady=(0, 10))

        self.ready_checkbox = ctk.CTkCheckBox(self.control_frame,
                                              text="Датасет уже обработан и готов к обучению",
                                              variable=self.is_ready_dataset)
        self.ready_checkbox.pack(pady=10, anchor="w", padx=15)

        self.bp_var = ctk.BooleanVar(value=True)  # По умолчанию включено
        self.gradient_var = ctk.BooleanVar(value=True)  # По умолчанию включено

        self.bp_checkbox = ctk.CTkCheckBox(self.control_frame, text="Обучение обратным распространением",
                                           variable=self.bp_var)
        self.bp_checkbox.pack(anchor="w", padx=15, pady=(5, 0))

        self.gradient_checkbox = ctk.CTkCheckBox(self.control_frame, text="Обучение градиентным спуском",
                                                 variable=self.gradient_var)
        self.gradient_checkbox.pack(anchor="w", padx=15, pady=(0, 10))

        self.train_button = ctk.CTkButton(self.control_frame, text="Начать обучение", command=self.start_training)
        self.train_button.pack(pady=20, fill="x", padx=10)

        self.stop_button = ctk.CTkButton(self.control_frame, text="Остановить обучение", command=self.stop_training_fn,
                                         fg_color="red")
        self.stop_button.pack(pady=5, fill="x", padx=10)
        self.stop_button.configure(state="disabled")  # изначально неактивна

        # === Левая часть: графики и логи ===
        self.left_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(15, 5), pady=10)
        self.left_frame.grid_rowconfigure(0, weight=0)  # статус
        self.left_frame.grid_rowconfigure(1, weight=0)  # параметры
        self.left_frame.grid_rowconfigure(2, weight=3)  # plot_frame — главный блок
        self.left_frame.grid_rowconfigure(3, weight=1)  # log_textbox — снизу, но гибкий
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.left_frame, text="Статус: Ожидание", anchor="w", justify="left")
        self.status_label.grid(row=0, column=0, sticky="ew", pady=(0, 5), padx=10)

        self.training_params_label = ctk.CTkLabel(self.left_frame, text="LR: - | Другое: -", anchor="w", justify="left")
        self.training_params_label.grid(row=1, column=0, sticky="ew", pady=(0, 10), padx=10)

        # === Графики ===
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt

        self.plot_frame = ctk.CTkFrame(self.left_frame)
        self.plot_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(1, weight=1)

        custom_bg_color = "#3A3A3A"

        # === Потери ===
        self.fig_loss, self.ax_loss = plt.subplots(figsize=(6, 2.5), facecolor=custom_bg_color)
        self.ax_loss.set_facecolor(custom_bg_color)
        self.ax_loss.set_title("Потери", color='white')
        self.ax_loss.tick_params(colors='white')
        self.ax_loss.grid(True, color='gray', linestyle='--', linewidth=0.5)
        for spine in self.ax_loss.spines.values():
            spine.set_color('white')

        self.loss_line, = self.ax_loss.plot([], [], color='#FFD700', label="Loss")
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.plot_frame)
        self.canvas_loss.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=5)

        # === Точность ===
        self.fig_acc, self.ax_acc = plt.subplots(figsize=(6, 2.5), facecolor=custom_bg_color)
        self.ax_acc.set_facecolor(custom_bg_color)
        self.ax_acc.set_title("Точность", color='white')
        self.ax_acc.tick_params(colors='white')
        self.ax_acc.grid(True, color='gray', linestyle='--', linewidth=0.5)
        for spine in self.ax_acc.spines.values():
            spine.set_color('white')

        self.acc_line, = self.ax_acc.plot([], [], color='#00FF00', label="Accuracy")
        self.canvas_acc = FigureCanvasTkAgg(self.fig_acc, master=self.plot_frame)
        self.canvas_acc.get_tk_widget().grid(row=0, column=1, sticky="nsew", pady=5)

        # === Лог ===
        self.log_textbox = ctk.CTkTextbox(self.left_frame)
        self.log_textbox.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.log_textbox.configure(state="disabled")

    def stop_training_fn(self):
        self.stop_training = True
        self.update_status("Остановка обучения...")

    def choose_dataset_dir(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(title="Выбор директории с датасетом")
        if path:
            self.dataset_path = path
            self.dataset_label.configure(text=path, text_color="white")
            self.update_status("Директория выбрана")

    def update_status(self, text):
        self.status_label.configure(text=f"Статус: {text}")

    def update_training_info(self, epoch, total_epochs, loss, train_acc, val_acc):
        text = f"Эпоха {epoch}/{total_epochs} | Потеря: {loss:.4f} | Train: {train_acc:.2%} | Val: {val_acc:.2%}"
        self.training_params_label.configure(text=text)

    def update_graph(self, epoch, loss, val_acc):
        self.loss_values.append(loss)
        self.accuracy_values.append(val_acc)

        epochs = range(1, len(self.loss_values) + 1)
        self.loss_line.set_data(epochs, self.loss_values)
        self.acc_line.set_data(epochs, self.accuracy_values)
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()
        self.ax_acc.draw_idle()

    def append_log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")  # прокрутка вниз
        self.log_textbox.configure(state="disabled")

    def training_callback(self, epoch, total_epochs, loss, train_acc, val_acc):
        log_msg = (
            f"Эпоха {epoch}/{total_epochs} | Потеря: {loss:.4f} | "
            f"Точн. train: {train_acc:.2f}% | Точн. val: {val_acc:.2f}%"
        )
        self.append_log(log_msg)  # показать в Textbox
        self.update_plot(loss, train_acc)  # опционально: обновить график

    def update_plot(self, loss, accuracy):
        self.loss_values.append(loss)
        self.accuracy_values.append(accuracy)

        # Обновляем данные линий
        self.loss_line.set_data(range(len(self.loss_values)), self.loss_values)
        self.acc_line.set_data(range(len(self.accuracy_values)), self.accuracy_values)

        # Обновляем пределы осей и перерисовываем графики
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.canvas_loss.draw()

        self.ax_acc.relim()
        self.ax_acc.autoscale_view()
        self.canvas_acc.draw()

    def start_training(self, save_path="gesture_model_norm_last.pkl"):
        self.stop_training = False  # сбрасываем флаг
        self.stop_button.configure(state="normal")
        self.train_button.configure(state="disabled")

        if not self.dataset_path:
            self.update_status("Директория не выбрана")
            return

        self.loss_values.clear()
        self.accuracy_values.clear()
        self.model = NeuralNetwork()

        def training_thread():
            try:
                self.update_status("Загрузка данных из датасета...")
                if self.ready_checkbox.get():
                    data = ImageProcessor.load_dataset(self.dataset_path, save_dir="augmented", use_preprocessed=True)
                else:
                    data = ImageProcessor.load_dataset(self.dataset_path, save_dir="augmented", use_preprocessed=False)
                if not data:
                    self.update_status("Ошибка: не удалось загрузить данные из датасета")
                    return

                X_train = data['train']['X']
                y_train = data['train']['y']
                self.class_names = data['train']['classes']

                self.update_status(f"Загружено {X_train.shape[0]} обучающих образцов")
                # Можно показать распределение классов через update_status или лог
                unique, counts = np.unique(y_train, return_counts=True)
                class_dist = ", ".join([f"{self.class_names[c]}: {n}" for c, n in zip(unique, counts)])
                self.update_status(f"Распределение классов: {class_dist}")

                self.update_status("Аугментация данных...")
                img_proc = ImageProcessor()
                X_aug, y_aug = img_proc.augment_data(X_train, y_train)
                self.update_status(f"После аугментации: {X_aug.shape[0]} образцов")

                self.update_status("Обучение...")
                start_time = time.time()
                mode = "backprop_only" if not self.gradient_var.get() else "full"
                self.model.train(X_aug, y_aug, callback=self.training_callback, mode=mode)
                training_time = time.time() - start_time
                self.update_status(f"Обучение завершено за {training_time:.2f} секунд")

                self.model.save_model(save_path)


                if 'test' in data:
                    X_test = data['test']['X']
                    y_test = data['test']['y']
                    test_accuracy = self.model.accuracy(X_test, y_test)
                    self.update_status(f"Точность на тестовых данных: {test_accuracy:.2f}%")
                self.update_status(f"Модель сохранена в {save_path}")

                if 'validation' in data:
                    X_val = data['validation']['X']
                    y_val = data['validation']['y']
                    val_accuracy = self.model.accuracy(X_val, y_val)
                    self.update_status(f"Точность на валидационных данных: {val_accuracy:.2f}%")

            except Exception as e:
                self.update_status(f"Ошибка: {e}")
            finally:
                self.stop_button.configure(state="disabled")
                self.train_button.configure(state="normal")

        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()

    def on_close(self):
        pass

    def update_status(self, text):
        def task():
            self.status_label.configure(text=f"Статус: {text}")

        self.after(0, task)

    def update_training_info(self, epoch, total_epochs, loss, acc):
        text = f"Эпоха {epoch}/{total_epochs}\nПотеря: {loss:.4f}\nТочность: {acc:.4f}"

        def task():
            self.info_label.configure(text=text)

        self.after(0, task)

    def update_graph(self):
        def task():
            # self.loss_line.set_data(range(1, len(self.loss_values)+1), self.loss_values)
            # self.acc_line.set_data(range(1, len(self.acc_values)+1), self.acc_values)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()

        self.after(0, task)
