# depth_maker/depth_maker_v1/constructor_layered_image.py

from .utils import (
    visualize_with_grid,
    load_image,
    add_alpha_channel,
    ensure_directory,
    setup_logging,
    save_image,
)

import cv2
import os
import logging
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from PIL import Image


class LogoOverlayPipeline:
    def __init__(self, dir_path=None, background_filename=None, logos_info=None,
                 combined_image_path=r'results\combined_image.png', bg_color=(0, 0, 0),
                 threshold=10, layered_image=None, background_size=None,
                 resize_by="width", enable_logging=True):
        """
        Initialize the pipeline for overlaying logos on a background.

        Parameters:
        - resize_by (str): Dimension to resize by ("width" or "height").
        """
        # Use pathlib for cross-platform path handling
        self.dir_path = Path(dir_path) if dir_path else None
        self.combined_image_path = self.dir_path / Path(combined_image_path) if dir_path else Path(combined_image_path)
        self.bg_color = bg_color
        self.threshold = threshold
        self.resize_by = resize_by  # Save the resize_by argument
        self.enable_logging_flag = enable_logging

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if enable_logging:
            self.enable_logging_method()
        else:
            self.disable_logging_method()

        if layered_image:
            # Initialize from a layered image object
            self.layered_image = layered_image
            self.background = layered_image.background
            self.logos_info = layered_image.logos
        else:
            # Initialize from separate background and logos info
            if background_filename is None or dir_path is None:
                self.logger.error("Either provide a layered_image or both dir_path and background_filename.")
                raise ValueError("Either provide a layered_image or both dir_path and background_filename.")
            self.background_filename = Path(background_filename)
            self.logos_info = logos_info if logos_info is not None else []
            full_background_path = self.dir_path / self.background_filename

            # Load the background image using the utility function
            self.background = load_image(full_background_path, cv2.IMREAD_COLOR)
            if background_size:
                self.background = self.resize_logo(self.background, background_size, resize_by=self.resize_by)
            self.background_height, self.background_width = self.background.shape[:2]



    def enable_logging_method(self, log_level=logging.INFO):
        """
        Включает логирование с указанным уровнем.
        
        Parameters:
        - log_level (int): Уровень логирования (например, logging.INFO, logging.DEBUG).
        """
        self.logger.setLevel(log_level)
        self.logger.disabled = False
        self.enable_logging_flag = True
        self.logger.info("Логирование включено.")

    def disable_logging_method(self):
        """
        Отключает логирование.
        """
        self.logger.disabled = True
        self.enable_logging_flag = False
        print("Логирование отключено.")


    def create_test_logo(self, filename='test_logo.png', size=(200, 200), color=(255, 255, 255)):
        """
        Creates an artificial image with a white square on a black background.
        """
        logo_size = (size[0] + 40, size[1] + 40)  # Adding padding to create black border
        logo = np.zeros((logo_size[1], logo_size[0], 4), dtype=np.uint8)
        x_start = 20
        y_start = 20
        logo[y_start:y_start + size[1], x_start:x_start + size[0], :3] = color
        logo[y_start:y_start + size[1], x_start:x_start + size[0], 3] = 255  # Alpha channel

        save_path = self.dir_path / filename
        save_image(logo, save_path)
        self.logger.info(f"Test logo saved as {save_path}")



    def prepare_logo_alpha(self, logo):
        """
        Prepares the alpha channel for the logo by making the background transparent.
        Utilizes the utility function `add_alpha_channel`.
        """
        try:
            logo_with_alpha = add_alpha_channel(logo)
            if logo_with_alpha.shape[2] < 4:
                self.logger.error("Logo image does not have an alpha channel after processing.")
                raise ValueError("Logo image does not have an alpha channel after processing.")

            # Create mask based on background color and threshold
            lower = np.array(self.bg_color) - self.threshold
            upper = np.array(self.bg_color) + self.threshold
            mask_bg = cv2.inRange(logo_with_alpha[:, :, :3], lower, upper)

            # Update alpha channel
            logo_with_alpha[:, :, 3] = np.where(mask_bg > 0, 0, 255)
            self.logger.debug("Alpha channel prepared based on background color and threshold.")
            return logo_with_alpha
        except Exception as e:
            self.logger.error(f"Error in preparing logo alpha: {e}")
            raise


    def crop_logo(self, logo):
        """
        Crops the logo to remove any fully transparent, black, or noisy background.

        Parameters:
        - logo (np.ndarray): Logo image with or without an alpha channel.

        Returns:
        - cropped_logo (np.ndarray): Cropped logo image.
        """
        if logo.shape[2] == 4:  # Если есть альфа-канал
            alpha_channel = logo[:, :, 3]
            _, thresh = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # Более гибкий порог

        # Убираем шум
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped_logo = logo[y:y+h, x:x+w]
        else:
            cropped_logo = logo  # Если границы не определены, возвращаем исходное изображение

        return cropped_logo


    def resize_logo(self, logo, target_size, resize_by="width"):
        """
        Resizes the logo while maintaining aspect ratio.
        """
        logo = self.crop_logo(logo)  # Crop the logo to remove unnecessary background
    
        original_height, original_width = logo.shape[:2]
        if resize_by == "width":
            scale = target_size / original_width
            new_size = (target_size, int(original_height * scale))
        elif resize_by == "height":
            scale = target_size / original_height
            new_size = (int(original_width * scale), target_size)
        else:
            self.logger.error("resize_by must be 'width' or 'height'.")
            raise ValueError("resize_by must be 'width' or 'height'.")
    
        logo_pil = Image.fromarray(logo)
        try:
            logo_resized_pil = logo_pil.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:
            logo_resized_pil = logo_pil.resize(new_size, Image.ANTIALIAS)
    
        # Return only the resized logo
        return np.array(logo_resized_pil)


    def rotate_logo(self, logo, angle):
        """
        Rotates the given logo (RGBA) image around its center by the specified angle.
    
        Parameters:
        - logo (np.ndarray): RGBA logo image to rotate.
        - angle (float): Angle in degrees. Positive values rotate counterclockwise.
    
        Returns:
        - rotated_logo (np.ndarray): Rotated RGBA logo image.
        """
        (h, w) = logo.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        # Calculate the new bounding box dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
    
        # Adjust the rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
    
        # Perform the rotation with a transparent border
        rotated_logo = cv2.warpAffine(
            logo,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # Transparent background
        )
        return rotated_logo
        
    def reflect_logo(self, logo, mode):
        """
        Reflects the given RGBA logo image according to the specified mode.
    
        Parameters:
        - logo (np.ndarray): RGBA logo image to reflect.
        - mode (str): Reflection mode ('horizontal', 'vertical', 'both', 'none').
    
        Returns:
        - reflected_logo (np.ndarray): Reflected RGBA logo image.
        """
        if mode == 'horizontal':
            self.logger.debug("Reflecting logo horizontally.")
            return cv2.flip(logo, 1)
        elif mode == 'vertical':
            self.logger.debug("Reflecting logo vertically.")
            return cv2.flip(logo, 0)
        elif mode == 'both':
            self.logger.debug("Reflecting logo both horizontally and vertically.")
            return cv2.flip(logo, -1)
        else:
            self.logger.debug("No reflection applied to the logo.")
            return logo

    
    def overlay_multiple_logos(self):
        """
        Overlays multiple logos onto a background image and creates a layered image object.
    
        Returns:
        - LayeredImageObject: The layered image object containing background and logos.
        """
        layered_image = LayeredImageObject(self.background)
    
        for logo_info in self.logos_info:
            logo_path = os.path.join(self.dir_path, logo_info['filename'])
            logo = load_image(logo_path, cv2.IMREAD_UNCHANGED)
    
            # Resize logo before adding it
            target_size = logo_info['sizes']  # Получаем размер из `logo_info`
            resize_by = logo_info.get('resize_by', "width")  # Определяем, по какой стороне менять размеры
            logo = self.resize_logo(logo, target_size, resize_by=resize_by)  # Здесь изменяем размеры
    
            # Prepare the alpha channel for the logo
            logo = self.prepare_logo_alpha(logo)
    
            # Apply rotation if specified
            rotation_angle = logo_info.get('rotation_angle', 0)  # Угол поворота
            if rotation_angle != 0:
                logo = self.rotate_logo(logo, rotation_angle)
    
            # Apply reflection if specified
            reflection = logo_info.get('reflection', 'none')  # Отражение (horizontal, vertical, both, none)
            if reflection != 'none':
                logo = self.reflect_logo(logo, reflection)
    
            # Extract coordinates and alpha
            x2, y2 = logo_info['point']
            alpha = logo_info.get('alpha', 1.0)
    
            # Добавляем обработанный логотип в LayeredImageObject
            layered_image.add_logo(logo, x2, y2, alpha)
    
        return layered_image




    def save_combined_image(self, layered_image):
        """
        Сохраняет финальное изображение с обрезкой до размеров фона.
        """
        combined_image = layered_image.render(crop_to_background=True)
        ensure_directory(self.combined_image_path.parent)
        save_image(combined_image, self.combined_image_path)
        self.logger.info(f"Combined image saved to {self.combined_image_path}.")

    def visualize_with_grid(self, combined_image, figsize=(14, 9), mode='color', title="Combined Image with Logos and Coordinate Grid"):
        """
        Визуализирует изображение с сеткой и логотипами.
        """
        visualize_with_grid(combined_image, self.logos_info, figsize, mode, title)
    


class LayeredImageObject:
    def __init__(self, background, logos=None, logger=None):
        self.background = background
        self.logos = logos if logos is not None else []
        self.logger = logger if logger else logging.getLogger(__name__)

        # Определяем размеры холста
        self.canvas_width = background.shape[1]
        self.canvas_height = background.shape[0]
        
        
    def add_logo(self, logo, x2, y2, alpha=1.0):
        """
        Adds a logo as a layer, cropping it if it exceeds background boundaries.
    
        Parameters:
        - logo (np.ndarray): Logo image with alpha channel.
        - x2, y2 (int): Coordinates defining the bottom-right corner to place the logo.
        - alpha (float): Transparency factor for the logo.
        """
        logo_height, logo_width = logo.shape[:2]
        x1 = x2 - logo_width
        y1 = y2 - logo_height
    
        # Обрезаем логотип, если он выходит за границы фона
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(self.canvas_width, x2)
        crop_y2 = min(self.canvas_height, y2)
    
        if crop_x1 > x1 or crop_y1 > y1 or crop_x2 < x2 or crop_y2 < y2:
            self.logger.info(f"Cropping logo at ({x1}, {y1}, {x2}, {y2}) to fit within background bounds.")
            # Обрезаем логотип
            logo_cropped = logo[max(0, -y1):logo_height - max(0, y2 - self.canvas_height),
                                max(0, -x1):logo_width - max(0, x2 - self.canvas_width)]
            x1, y1 = crop_x1, crop_y1
            x2, y2 = crop_x2, crop_y2
        else:
            logo_cropped = logo
    
        cropped_height, cropped_width = logo_cropped.shape[:2]
        self.logos.append({
            'image': logo_cropped,
            'x2': x2,
            'y2': y2,
            'alpha': alpha,
            'coords': (x2, y2),
            'final_size': (cropped_width, cropped_height)
        })


    def save(self, filepath):
        """
        Сохраняет объект слоя в файл.

        Parameters:
        - filepath (str): Путь для сохранения объекта.
        """
        data = {
            'background': self.background,
            'logos': self.logos,
            'canvas_size': (self.canvas_width, self.canvas_height),  # Сохраняем размеры холста
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Layered image object saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Загружает объект слоя из файла.

        Parameters:
        - filepath (str): Путь к файлу для загрузки.

        Returns:
        - LayeredImageObject: Загруженный объект слоя.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        layered_image = LayeredImageObject(data['background'])
        layered_image.logos = data['logos']
        layered_image.canvas_width, layered_image.canvas_height = data.get('canvas_size', layered_image.background.shape[:2])

        return layered_image


    def render(self, include_background=True, crop_to_background=True):
        """
        Рендерит итоговое комбинированное изображение с учетом возможности обрезки по фону.
    
        Parameters:
        - include_background (bool): Включать ли фон в итоговое изображение.
        - crop_to_background (bool): Обрезать ли изображение по размеру фона.
    
        Returns:
        - combined_img (np.ndarray): Итоговое изображение.
        """
        # Определяем размеры холста
        max_x = max((logo['x2'] for logo in self.logos), default=0)
        max_y = max((logo['y2'] for logo in self.logos), default=0)
        self.canvas_width = max(max_x, self.background.shape[1])
        self.canvas_height = max(max_y, self.background.shape[0])
    
        # Создаем холст
        combined_img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
    
        # Добавляем фон
        if include_background:
            combined_img[:self.background.shape[0], :self.background.shape[1]] = self.background
    
        # Накладываем логотипы
        for logo_info in self.logos:
            logo = logo_info['image']
            x2, y2 = logo_info['x2'], logo_info['y2']
            alpha = logo_info['alpha']
            combined_img = self.overlay_logo(combined_img, logo, x2, y2, alpha)
    
        # Обрезаем по размеру фона, если указано
        if crop_to_background:
            combined_img = combined_img[:self.background.shape[0], :self.background.shape[1]]
    
        return combined_img
        

    
    def overlay_logo(self, background, logo, x2, y2, alpha=1.0):
        """
        Overlays the processed logo onto the background image at the specified coordinates with transparency.
        """
        try:
            logo_height, logo_width = logo.shape[:2]
            x1 = x2 - logo_width
            y1 = y2 - logo_height
    
            # Обрезаем логотип, если он выходит за пределы фона
            crop_x1 = max(0, x1)
            crop_y1 = max(0, y1)
            crop_x2 = min(background.shape[1], x2)
            crop_y2 = min(background.shape[0], y2)
    
            roi_width = crop_x2 - crop_x1
            roi_height = crop_y2 - crop_y1
    
            # Корректируем область логотипа
            logo_cropped = logo[max(0, -y1):logo_height - max(0, y2 - background.shape[0]),
                                max(0, -x1):logo_width - max(0, x2 - background.shape[1])]
    
            roi = background[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
    
            if logo_cropped.shape[2] == 4:  # Если есть альфа-канал
                logo_bgr = logo_cropped[:, :, :3].astype(np.float32)
                logo_alpha_channel = (logo_cropped[:, :, 3] / 255.0) * alpha
                logo_alpha = cv2.merge([logo_alpha_channel] * 3)
            else:
                logo_bgr = logo_cropped.astype(np.float32)
                logo_alpha = np.ones(logo_bgr.shape, dtype=np.float32) * alpha
    
            blended_roi = logo_bgr * logo_alpha + roi * (1 - logo_alpha)
            blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
    
            combined_img = background.copy()
            combined_img[crop_y1:crop_y2, crop_x1:crop_x2] = blended_roi
    
            return combined_img
        except Exception as e:
            self.logger.error(f"Error in overlaying logo: {e}")
            raise
    
