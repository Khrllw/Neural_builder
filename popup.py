import customtkinter as ctk
import tkinter as tk


class SuccessPopup(ctk.CTkToplevel):
    def __init__(self, parent, message="Настройки успешно сохранены!"):
        super().__init__(parent)
        self.parent = parent

        # Блок взаимодействия с другими окнами
        self.grab_set()
        self.attributes("-topmost", True)
        self.overrideredirect(True)

        # Обновим размеры и положение родителя
        self.update_idletasks()
        self.parent.update_idletasks()
        px = self.parent.winfo_rootx()
        py = self.parent.winfo_rooty()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()

        # Размеры модального окна
        win_width = 320
        win_height = 140

        # Центрирование
        x = px + (pw - px) // 2 - win_width // 2
        y = py + (ph - py) // 2 - win_height // 2

        # Установка геометрии
        self.geometry(f"{win_width}x{win_height}+{x}+{y}")

        # Полупрозрачный тёмный фон
        self.overlay = tk.Toplevel(self.parent)
        self.overlay.geometry(f"{pw}x{ph}+{px}+{py}")
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-alpha", 0.7)
        self.overlay.configure(bg="black")
        self.overlay.lift()
        self.overlay.attributes("-topmost", True)

        # Основной скруглённый контейнер внутри окна
        self.main_frame = ctk.CTkFrame(self, fg_color="#2B2B2B", corner_radius=15)
        self.main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Содержимое
        label = ctk.CTkLabel(self.main_frame, text=message, text_color="white",
                             font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=(25, 10), padx=20)

        ok_button = ctk.CTkButton(self.main_frame, text="Ок", command=self.close_popup)
        ok_button.pack(pady=5)

    def close_popup(self):
        self.overlay.destroy()
        self.grab_release()
        self.destroy()
