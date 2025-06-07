import importlib
from typing import BinaryIO

import numpy as np
import pickle

import config
from config import NeuralNetConfig


class NeuralNetwork:
    def __init__(self):
        """
        Инициализирует архитектуру и параметры нейросети согласно заданной конфигурации.
        """

        # Архитектура сети
        self.grads_b = None
        self.grads_w = None
        self.z_values = None
        self.activations = None
        self.mutation_std = 0.01
        importlib.reload(config)
        self.layer_sizes = [NeuralNetConfig.INPUT_SIZE] + NeuralNetConfig.HIDDEN_SIZES + [NeuralNetConfig.OUTPUT_SIZE]

        # Параметры модели
        self.weights = []  # Матрицы весов для каждого слоя
        self.biases = []  # Векторы смещений для каждого слоя

        print("Layer sizes:", self.layer_sizes)
        for i, w in enumerate(self.weights):
            print(f"Weight matrix {i} shape: {w.shape}")

        # Инициализация весов и смещений
        for i in range(len(self.layer_sizes) - 1):
            # Инициализация Xavier/Glorot
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

        # Параметры оптимизации
        self.learning_rate = NeuralNetConfig.LEARNING_RATE
        self.reg_lambda = NeuralNetConfig.REG_LAMBDA
        self.momentum = NeuralNetConfig.MOMENTUM
        self.target_accuracy = NeuralNetConfig.TARGET_ACCURACY

        # Вспомогательные переменные для оптимизации
        self.v_weights = [np.zeros_like(w) for w in self.weights]  # "Скорость" для весов (для момента)
        self.v_biases = [np.zeros_like(b) for b in self.biases]  # "Скорость" для смещений (для момента)

        # Для ранней остановки
        self.best_weights = None
        self.best_biases = None

    # Основная функция активации для скрытых слоев
    @staticmethod
    def relu(x):
        """Функция активации ReLU"""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Производная функции ReLU для обратного распространения ошибки"""
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        """Функция softmax для преобразования выходов в вероятности"""
        # Устойчивый softmax
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        """
        Прямое распространение сигнала по сети.
        Вычисляет предсказания сети для входных данных
            :param x: входные данные
            :return: вероятности классов
        """
        # Временные хранилища для прямого прохода
        self.activations = [x]  # Активации каждого слоя
        self.z_values = []  # Входные значения перед активацией

        # Прямое распространение через все слои, кроме последнего
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))

        # Последний слой с softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))

        # Возвращает выходные вероятности
        return self.activations[-1]

    def compute_loss(self, y):
        """
        Вычисляет функцию потерь (кросс-энтропия + L2-регуляризация).
            :param y: истинные метки классов
            :return: значение потерь
        """
        # Устойчивая кросс-энтропия
        m = y.shape[0]
        probs = self.activations[-1]

        # Устойчивая кросс-энтропия
        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        correct_log_probs = -np.log(clipped_probs[range(m), y])
        data_loss = np.sum(correct_log_probs) / m

        # Регуляризация L2
        reg_loss = 0.5 * self.reg_lambda * sum(np.sum(w * w) for w in self.weights)

        return data_loss + reg_loss

    # Обратное распространение (Backward Pass)
    def backward(self, y):
        """
        Обратное распространение ошибки и обновление весов.
             :param y: истинные метки классов
        """
        m = y.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        # Ошибка на выходном слое
        delta = self.activations[-1].copy()
        delta[range(m), y] -= 1
        delta /= m

        # Градиенты для последнего слоя
        self.d_weights[-1] = np.dot(self.activations[-2].T, delta) + self.reg_lambda * self.weights[-1]
        self.d_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        # Обратное распространение через скрытые слои
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.relu_derivative(self.z_values[l])
            self.d_weights[l] = np.dot(self.activations[l].T, delta) + self.reg_lambda * self.weights[l]
            self.d_biases[l] = np.sum(delta, axis=0, keepdims=True)

    def update_backward_weights(self):
        for i in range(len(self.weights)):
            noise = np.random.normal(0, self.mutation_std, self.weights[i].shape)
            self.weights[i] += noise

            bias_noise = np.random.normal(0, self.mutation_std, self.biases[i].shape)
            self.biases[i] += bias_noise

    def update_weights(self):
        """
        Обновляет веса и смещения с использованием Momentum.
        Предполагается, что градиенты self.d_weights и self.d_biases уже вычислены.
        """

        # Обновление весов с momentum
        for i in range(len(self.weights)):
            self.v_weights[i] = self.momentum * self.v_weights[i] + self.learning_rate * self.d_weights[i]
            self.v_biases[i] = self.momentum * self.v_biases[i] + self.learning_rate * self.d_biases[i]
            self.weights[i] -= self.v_weights[i]
            self.biases[i] -= self.v_biases[i]

    def train(self, X, y, epochs=NeuralNetConfig.EPOCHS, batch_size=NeuralNetConfig.BATCH_SIZE,
              validation_split=NeuralNetConfig.VALIDATION_SPLIT, mode="full", callback=None):
        """
               Обучает модель на обучающей выборке.
                   :param mode: тип обучения нейросети
                   :param x: входные данные
                   :param y: метки классов
                   :param epochs: количество эпох
                   :param batch_size: размер мини-батча
                   :param validation_split: доля валидационной выборки
                   :param callback: функция обратного вызова с параметрами (epoch, epochs, loss, train_acc, val_acc)
        """

        # Разделение на обучающую и валидационную выборки
        X_train, y_train, X_val, y_val = self._split_data(X, y, validation_split)

        print("Min label:", np.min(y))
        print("Max label:", np.max(y))
        print("Output layer size:", self.layer_sizes[-1])

        best_val_acc = 0.0
        no_improve = 0

        patience = NeuralNetConfig.PATIENCE

        for epoch in range(epochs):
            # Уменьшение learning rate
            self._update_learning_rate(epoch)

            # Перемешивание данных
            # Проход одной эпохи
            epoch_loss, num_batches = self._train_one_epoch(X_train, y_train, batch_size, mode=mode)

            # Валидация
            self.forward(X_val)  # forward прогоняет валидационные данные
            val_loss = self.compute_loss(y_val)  # теперь compute_loss использует правильные self.activations
            val_acc = self.accuracy(X_val, y_val)
            train_acc = self.accuracy(X_train, y_train)

            # Средние потери за эпоху
            epoch_loss /= num_batches

            self._log_epoch(epoch, epoch_loss, val_loss, train_acc, val_acc)
            if callback is not None:
                callback(epoch, epochs, epoch_loss, train_acc, val_acc)
            if hasattr(callback, '__self__') and callback.__self__.stop_training:
                print("Остановка обучения по запросу пользователя")
                break
            # Условия ранней остановки
            if self._is_improved(val_acc, best_val_acc):
                best_val_acc = val_acc
                no_improve = 0
                self._save_best_weights()
            else:
                no_improve += 1
                if no_improve >= NeuralNetConfig.PATIENCE or epoch >= NeuralNetConfig.EPOCHS:
                    print(f"Early stopping at epoch {epoch}. Best val accuracy: {best_val_acc:.2f}%")
                    self._restore_best_weights()
                    break
            # Дополнительное условие остановки при достижении целевой точности
            if val_acc >= self.target_accuracy:
                print(f"Target accuracy of {self.target_accuracy}% reached at epoch {epoch}")
                break

        # Финал обучения
        self._restore_best_weights()
        print("Training completed")

    @staticmethod
    def _split_data(x, y, validation_split):
        """
        Делит выборку на обучающую и валидационную.
        """
        idx = np.random.permutation(x.shape[0])
        split = int(len(idx) * (1 - validation_split))
        train_idx, val_idx = idx[:split], idx[split:]
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    @staticmethod
    def _log_epoch(epoch, train_loss, val_loss, train_acc, val_acc):
        """
        Логирует метрики за эпоху.
        """
        print(f"[EPOCH {epoch}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    def _update_learning_rate(self, epoch):
        """
        Понижает learning rate каждые N эпох.
        """
        if epoch > 0 and epoch % 5 == 0:
            self.learning_rate *= 0.9
            print(f"[LR UPDATE] Epoch {epoch}: learning rate reduced to {self.learning_rate:.6f}")

    def _train_one_epoch(self, x_train, y_train, batch_size, mode="full"):
        """
        Выполняет одну эпоху обучения.
        """
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        epoch_loss = 0
        num_batches = int(np.ceil(x_train.shape[0] / batch_size))

        # Мини-батчи
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, x_train.shape[0])
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            # Прямое и обратное распространение
            self.forward(x_batch)

            if mode == "backprop_only":
                self.backward(y_batch)
                self.update_backward_weights()
            if mode == "full":
                self.backward(y_batch)
                self.update_weights()

            # Расчет потерь
            batch_loss = self.compute_loss(y_batch)
            if not np.isnan(batch_loss):
                epoch_loss += batch_loss
        return epoch_loss, num_batches

    @staticmethod
    def _is_improved(val_acc, best_val_acc, min_delta=NeuralNetConfig.MIN_DELTA):
        """
        Проверяет, улучшилась ли точность.
        """
        return val_acc > best_val_acc + min_delta

    def _save_best_weights(self):
        """
        Сохраняет лучшие веса и смещения.
        """
        self.best_weights = [w.copy() for w in self.weights]
        self.best_biases = [b.copy() for b in self.biases]
        print(f"[INFO] Best weights updated.")

    def _restore_best_weights(self):
        """
        Восстанавливает сохранённые лучшие веса.
        """
        if self.best_weights is not None and self.best_biases is not None:
            self.weights = [w.copy() for w in self.best_weights]
            self.biases = [b.copy() for b in self.best_biases]
            print("[INFO] Best weights restored.")

    def predict(self, x):
        """
        Предсказание меток классов.
            :param x: входные данные
            :return: предсказанные метки
        """
        return np.argmax(self.forward(x), axis=1)  # Индекс максимальной вероятности

    # Предсказание вероятностей
    def predict_proba(self, x):
        """
       Возвращает вероятности классов.
           :param x: входные данные
           :return: массив вероятностей по каждому классу
       """
        return self.forward(x)

    # Вычисление точности
    def accuracy(self, x, y):
        """Подсчет точности предсказаний"""
        predictions = self.predict(x)
        return np.mean(predictions == y) * 100

    # Сохранение модели
    def save_model(self, filename):
        """Сохранение весов и конфигурации модели"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'layer_sizes': self.layer_sizes
            }, f)

    # загрузка модели
    def load_model(self, filename):
        """Загрузка модели из файла"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
            self.layer_sizes = data['layer_sizes']
