import customtkinter as ctk

class HelpFrame(ctk.CTkFrame):
    """
    Фрейм со справочной информацией о приложении.
    Красиво оформлен в стиле темы customtkinter.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.configure(corner_radius=20)
        self.pack_propagate(False)

        # Заголовок
        title = ctk.CTkLabel(
            self,
            text="О приложении",
            font=ctk.CTkFont(size=28, weight="bold"),
            pady=10
        )
        title.pack(pady=(20, 10))

        # Текст с описанием
        help_text = (
            "Добро пожаловать в приложение \"NeuralBuilder\"!\n\n"
            "Это приложение предназначено для распознавания жестов рук в реальном времени с помощью нейросети и компьютерного зрения.\n\n"
            "Основные возможности приложения:\n"
            "• Детекция жестов \"Камень\", \"Ножницы\", \"Бумага\" в режиме live-видео.\n"
            "• Создание собственного датасета изображений для обучения.\n"
            "• Предварительная обработка и очистка данных.\n"
            "• Настройка параметров маски кожи для повышения точности.\n"
            "• Обучение и дообучение нейросети с помощью ваших данных.\n"
            "• Гибкие настройки интерфейса и параметров распознавания.\n\n"
            "Используйте боковое меню для переключения между режимами работы.\n"
            "Нажмите клавишу Escape для быстрого свертывания окна.\n"
        )

        # Текстовое поле
        text_box = ctk.CTkLabel(
            self,
            text=help_text,
            font=ctk.CTkFont(size=14),
            justify="left",
            corner_radius=20,
            wraplength=700,
            padx=20,
            pady=15
        )
        text_box.pack(padx=20, pady=(0, 15), fill="x")

        # Нижний блок (в том же стиле, что и text_box)
        bottom_frame = ctk.CTkFrame(
            self,
            corner_radius=20,
            fg_color=ctk.ThemeManager.theme["CTkFrame"]["fg_color"],  # под стиль темы
        )
        bottom_frame.pack(fill="x", pady=(0, 20), padx=20)

        # Текст с пожеланием удачи
        good_luck_label = ctk.CTkLabel(
            bottom_frame,
            text="Желаем удачи в использовании!\nПусть ваши жесты всегда распознаются правильно 🤚",
            font=ctk.CTkFont(size=16, weight="bold"),
            justify="left",
            wraplength=700,
            padx=20,
            pady=15
        )
        good_luck_label.pack(fill="x")

    def on_close(self):
        pass
