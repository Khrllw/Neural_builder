# config.py
import numpy as np

# Параметры маски кожи
LOWER_SKIN = np.array([86, 133, 65], dtype=np.uint8)
UPPER_SKIN = np.array([255, 255, 255], dtype=np.uint8)
MASK_COLOR = (128, 0, 255)

# Общие настройки
IMG_SIZE = (80, 80)
PADDING = 20
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Сбор данных
CAMERA_INDEX = 0
CAPTURE_INSTRUCTION = "Нажмите 'c' для захвата, 'q' для выхода"

# Датасет
DATASET_SUBFOLDERS = ['train', 'test', 'validation']
DEFAULT_DATA_DIR = "Rock-Paper-Scissors"
SAMPLES_PER_CLASS = 100
NUM_CLASSES = 3

# Аугментация данных
AUGMENTATION_SETTINGS = {
    "flip": True, 
    "rotate_angles": [15, -5, 5, 15],
    "shift_offsets": [(-5, 0), (5, 0)],
    "add_noise": True
}


class NeuralNetConfig:
    # Архитектура сети
    INPUT_SIZE = IMG_SIZE[0] ** 2  # Например, для изображений 96x96 пикселей (MNIST)
    HIDDEN_SIZES = [128, 64]  # Количество нейронов в скрытых слоях
    OUTPUT_SIZE = 3  # Количество классов

    # Параметры обучения
    LEARNING_RATE = 0.0001  # Начальная скорость обучения
    REG_LAMBDA = 0.0001  # Коэффициент L2-регуляризации
    MOMENTUM = 0.9  # Моментум (ускоряет сходимость и сглаживает обновления)

    # Критерии остановки обучения
    TARGET_ACCURACY = 99  # Целевая точность (в процентах)
    PATIENCE = 5  # Кол-во эпох без улучшения до ранней остановки
    MIN_DELTA = 0.001  # Минимальное улучшение точности, чтобы не засчитать эпоху как «без улучшения»
    LR_DECAY_EPOCHS = 100  # Через сколько эпох понижается learning rate (каждые N эпох)
    LR_DECAY_FACTOR = 0.9  # Во сколько раз понижается learning rate при срабатывании понижения

    # Прочее
    EPOCHS = 120  # Максимальное количество эпох обучения
    BATCH_SIZE = 8  # Размер мини-батча (сколько примеров за раз обрабатывается)
    VALIDATION_SPLIT = 0.2  # Процент обучающей выборки, выделяемый под валидацию
    SEED = 50  # Фиксированное зерно для генератора случайных чисел (гарантирует повторяемость)