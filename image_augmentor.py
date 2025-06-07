import cv2
import numpy as np


class ImageAugmentor:
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape
        self.resize_center = (self.resize_shape[0] // 2, self.resize_shape[1] // 2)

    def augment_dataset(self, x, y,
                        flip=True,
                        rotate_angles=None,
                        shift_offsets=None,
                        add_noise=True):
        """
        Аугментирует датасет x и y с помощью заданных трансформаций
            :param x: np.ndarray — массив изображений в виде векторов (n_samples, H*W)
            :param y: np.ndarray — метки классов (n_samples,)
            :param flip: bool — применять ли отражения (вертикальное и горизонтальное)
            :param rotate_angles: list or None — углы поворота (в градусах), если None — не применять повороты
            :param shift_offsets: list or None — список смещений (dx, dy), если None — не применять сдвиги
            :param add_noise: bool — добавлять ли гауссов шум
            :return: tuple (X_aug, y_aug) — аугментированные данные и метки
        """
        augmented_x, augmented_y = [], []

        for i in range(len(x)):
            img = x[i].reshape(self.resize_shape)
            label = y[i]
            self.augment_single_image(
                img, label, augmented_x, augmented_y,
                flip=flip,
                rotate_angles=rotate_angles,
                shift_offsets=shift_offsets,
                add_noise=add_noise
            )
        print("Augmented labels min/max:", np.min(augmented_y), np.max(augmented_y))
        print("Augmented labels unique:", np.unique(augmented_y))

        return np.array(augmented_x), np.array(augmented_y)

    def augment_single_image(self, img, label, augmented_x, augmented_y,
                             flip, rotate_angles, shift_offsets, add_noise):
        """
        Применяет аугментации к одному изображению
            :param img: 2D numpy array изображения
            :param label: метка соответствующего изображения
            :param augmented_x: список для хранения новых изображений
            :param augmented_y: список для хранения меток
            :param flip: bool — выполнять ли отражения
            :param rotate_angles: list or None — список углов для поворота
            :param shift_offsets: list or None — список смещений
            :param add_noise: bool — добавлять ли шум
        """
        # augmented_x.append(img.flatten())
        # augmented_y.append(label)

        flipped = []

        if flip:
            # Вертикальное отражение
            flipped_v = cv2.flip(img, 0)
            # Горизонтальное отражение
            # flipped_h = cv2.flip(img, 1)

            self._append_augmented(flipped_v, label, augmented_x, augmented_y)
            # self._append_augmented(flipped_h, label, augmented_x, augmented_y)
            flipped.append(flipped_v)

        # Применить повороты и шум ко всем вариантам
        for variant in [img] + flipped:
            if rotate_angles:
                self.apply_rotations(variant, label, rotate_angles, augmented_x, augmented_y)
            if add_noise:
                self.apply_noise(variant, label, augmented_x, augmented_y)

        if shift_offsets:
            self.apply_shifts(img, label, shift_offsets, augmented_x, augmented_y)

    def apply_rotations(self, img, label, angles, augmented_x, augmented_y):
        """
        Применяет повороты к изображению на заданные углы
            :param img: входное изображение размером self.resize_shape
            :param label: метка класса изображения
            :param angles: список углов поворота изображения (в градусах)
            :param augmented_x: список, в который добавляются преобразованные изображения
            :param augmented_y: список меток, соответствующих изображениям
        """
        for angle in angles:
            # Создаём матрицу поворота с центром изображения
            rot_matrix = cv2.getRotationMatrix2D(self.resize_center, angle, 1.0)
            # Применяем аффинное преобразование (поворот) к изображению
            rotated = cv2.warpAffine(img, rot_matrix, self.resize_shape)
            # Преобразуем в вектор и добавляем в список аугментированных данных
            self._append_augmented(rotated.flatten(), label, augmented_x, augmented_y)

    def apply_shifts(self, img, label, offsets, augmented_x, augmented_y):
        """
        Применяет сдвиги к изображению по указанным векторам смещения
            :param img: входное изображение размером self.resize_shape
            :param label: метка класса изображения
            :param offsets: список кортежей (dx, dy) — смещения по x и y
            :param augmented_x: список, в который добавляются преобразованные изображения
            :param augmented_y: список меток, соответствующих изображениям
        """

        for dx, dy in offsets:
            transform_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(img, transform_matrix, self.resize_shape)
            self._append_augmented(shifted.flatten(), label, augmented_x, augmented_y)

    def apply_noise(self, img, label, augmented_x, augmented_y):
        """
        Добавляет гауссов шум к изображению
           :param img: входное изображение
           :param label: метка класса изображения
           :param augmented_x: список, в который добавляются преобразованные изображения
           :param augmented_y: список меток, соответствующих изображениям
        """
        noise = np.random.normal(0, 0.1, self.resize_shape)
        noisy_img = np.clip(img + noise, 0, 1)
        self._append_augmented(noisy_img.flatten(), label, augmented_x, augmented_y)

    @staticmethod
    def _append_augmented(img, label, x_list, y_list):
        x_list.append(img.flatten())
        y_list.append(label)
