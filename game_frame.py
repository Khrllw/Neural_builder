import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from collections import deque

class GameFrame(tk.Frame):
    def __init__(self, parent, model, gesture_names=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.model = model
        self.gesture_names = gesture_names if gesture_names else ["Paper", "Rock", "Scissors"]
        self.history = deque(maxlen=5)

        # Камера
        self.cap = cv2.VideoCapture(0)

        # Переменные игры
        self.player1_score = 0
        self.player2_score = 0
        self.draw_count = 0

        self.game_state = "ready"  # ready, countdown, result
        self.game_start_time = None
        self.countdown_duration = 3
        self.result_duration = 3

        self.player_choices = [None, None]
        self.winner = None

        self.stable_gestures = [[], []]
        self.stable_display = [None, None]
        self.stable_threshold = 3
        self.conf_threshold = 0.8

        # Виджеты Tkinter
        self.video_label = Label(self)
        self.video_label.pack()

        self.info_label = Label(self, text="", font=("Arial", 14))
        self.info_label.pack()

        # Запускаем цикл обновления видео
        self.update_frame()

        # Привязка нажатия пробела для старта игры
        parent.bind("<space>", self.start_countdown)

    def start_countdown(self, event=None):
        if self.game_state == "ready":
            self.game_state = "countdown"
            self.game_start_time = time.time()

    def determine_winner(self, gesture1, gesture2):
        if gesture1 == gesture2:
            return 0  # ничья
        # Камень=1, Ножницы=2, Бумага=0 для примера
        # Камень > Ножницы, Ножницы > Бумага, Бумага > Камень
        if (gesture1 == 0 and gesture2 == 2) or (gesture1 == 2 and gesture2 == 1) or (gesture1 == 1 and gesture2 == 0):
            return 1
        return 2

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        # TODO: Заменить ниже на вашу функцию получения рук и маски
        hands, bboxes = self.mock_detect_hands(frame)

        current_time = time.time()

        # Логика игры
        if self.game_state == "ready":
            self.info_label.config(text="Подготовьте руки. Нажмите пробел для старта.")
            if len(hands) >= 2:
                for i in range(2):
                    probs = self.model_predict(hands[i])  # np.array с вероятностями
                    gesture = np.argmax(probs)
                    self.player_choices[i] = gesture

                    confidence = probs[gesture]

                    self.stable_gestures[i].append(gesture)
                    if len(self.stable_gestures[i]) > self.stable_threshold:
                        self.stable_gestures[i].pop(0)

                    most_common = max(set(self.stable_gestures[i]), key=self.stable_gestures[i].count)
                    count = self.stable_gestures[i].count(most_common)

                    if count == self.stable_threshold or confidence > self.conf_threshold:
                        self.stable_display[i] = most_common

                    if i < len(bboxes):
                        x, y, w, h = bboxes[i]
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        if self.stable_display[i] is not None:
                            gesture_name = self.gesture_names[self.stable_display[i]]
                            cv2.putText(frame, f"Игрок {i+1}: {gesture_name}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        elif self.game_state == "countdown":
            time_left = self.countdown_duration - (current_time - self.game_start_time)
            if time_left > 0:
                count = int(np.ceil(time_left))
                self.info_label.config(text=f"Отсчет: {count}")
                cv2.putText(frame, str(count), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            else:
                self.game_state = "result"
                # Фиксируем выборы игроков
                for i in range(2):
                    if self.stable_display[i] is not None:
                        self.player_choices[i] = self.stable_display[i]
                # Определяем победителя
                self.winner = self.determine_winner(self.player_choices[0], self.player_choices[1])
                if self.winner == 1:
                    self.player1_score += 1
                    result_text = "Игрок 1 победил!"
                elif self.winner == 2:
                    self.player2_score += 1
                    result_text = "Игрок 2 победил!"
                else:
                    self.draw_count += 1
                    result_text = "Ничья!"
                self.info_label.config(text=result_text)
                self.game_start_time = time.time()

        elif self.game_state == "result":
            if current_time - self.game_start_time < self.result_duration:
                score_text = f"Счет: Игрок 1: {self.player1_score} — Игрок 2: {self.player2_score} — Ничьи: {self.draw_count}"
                self.info_label.config(text=score_text)
                # Показываем выборы игроков на экране
                for i in range(2):
                    if self.player_choices[i] is not None and i < len(bboxes):
                        x, y, w, h = bboxes[i]
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)
                        gesture_name = self.gesture_names[self.player_choices[i]]
                        cv2.putText(frame, f"Игрок {i+1}: {gesture_name}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Возвращаемся в готовое состояние
                self.game_state = "ready"
                self.info_label.config(text="Подготовьте руки. Нажмите пробел для старта.")
                self.stable_gestures = [[], []]
                self.stable_display = [None, None]
                self.player_choices = [None, None]

        # Отображаем кадр в Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(30, self.update_frame)

    def model_predict(self, hand_features):
        # Пример фиктивной функции предсказания модели
        # Замените на вызов вашей модели, например:
        # return self.model.predict_proba(hand_features.reshape(1, -1))[0]
        # Ниже случайный выбор для демонстрации
        probs = np.random.rand(len(self.gesture_names))
        probs /= probs.sum()
        return probs

    def mock_detect_hands(self, frame):
        # Ваша функция детекции рук + возврат bbox
        # Сейчас заглушка — 2 бокса в центре
        h, w, _ = frame.shape
        bbox1 = (w//4, h//3, 100, 100)
        bbox2 = (w//2 + 50, h//3, 100, 100)
        # Заглушка признаков руки
        hand1 = np.random.rand(20)
        hand2 = np.random.rand(20)
        return [hand1, hand2], [bbox1, bbox2]

    def close(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Игра Камень-Ножницы-Бумага")
    # Передайте сюда вашу модель
    dummy_model = None
    frame = GameFrame(root, dummy_model)
    frame.pack()

    def on_closing():
        frame.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
