# depth_maker/depth_maker_v1/utils.py

import os
import sys
import cv2
import numpy as np
import requests
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import logging


def setup_logging(log_level=logging.INFO, log_file: str = None, 
enable_logging: bool = True):
    """
    Настраивает конфигурацию логирования.
    
    Параметры:
        log_level (int): Уровень логирования (например, logging.INFO, logging.DEBUG).
        log_file (str, optional): Если указан, логи также будут записываться в этот файл.
        enable_logging (bool): Включить ил
        и отключить логирование.
    """
    logger = logging.getLogger()
    
    # Удаляем все существующие обработчики, чтобы избежать дублирования
    if logger.hasHandlers():
        logger.handlers.clear()
    
    if enable_logging:
        handlers = [logging.StreamHandler()]
        if log_file:
            # Убедитесь, что директория для log_file существует
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        logging.info("Логирование включено.")
    else:
        # Если логирование отключено, устанавливаем уровень на WARNING, чтобы минимизировать вывод
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning("Логирование отключено.")




# ---------------------
# Model Management
# ---------------------

def download_model(url: str, save_path: str):
    """
    Downloads a model from the specified URL to the given save path with a progress bar.

    Parameters:
        url (str): The URL to download the model from.
        save_path (str): The local path where the model will be saved
    """
    
    if os.path.exists(save_path):
        logging.info(f"Model checkpoint already exists at {save_path}. Skipping download.")
        return


    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download the model: {e}")
        raise

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        with open(save_path, "wb") as file, tqdm(
            desc=save_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            file=sys.stdout  # Попытка направить вывод в stdout
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        logging.info(f"Model successfully downloaded and saved to: {save_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the model: {e}")
        raise


def validate_model_path(model_path: str):
    """
    Validates whether the model exists at the specified path.

    Parameters:
        model_path (str): The path to the model file.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model checkpoint not found at {model_path}. Please download it first.")
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please download it first.")
    else:
        logging.info(f"Model checkpoint found at {model_path}.")
# ---------------------
# Image Processing Helpers
# ---------------------


def add_alpha_channel(image: np.ndarray, alpha: int = 255) -> np.ndarray:
    """
    Adds an alpha channel to an image if it doesn't have one.

    Parameters:
        image (np.ndarray): The input image.
        alpha (int): The default alpha value.

    Returns:
        np.ndarray: The image with an alpha channel.
    """
    if len(image.shape) < 3 or image.shape[2] < 3:
        logging.error("Input image must have at least 3 channels (BGR).")
        raise ValueError("Input image must have at least 3 channels (BGR).")

    if image.shape[2] == 4:
        logging.info("Image already has an alpha channel.")
        return image
    else:
        b, g, r = cv2.split(image)
        alpha_channel = np.ones(b.shape, dtype=b.dtype) * alpha
        image_with_alpha = cv2.merge((b, g, r, alpha_channel))
        logging.info("Alpha channel added to the image.")
        return image_with_alpha


# ---------------------
# Additional Utilities 
# ---------------------


def save_image(image: np.ndarray, save_path: str):
    """
    Saves an image to the specified path.

    Parameters:
        image (np.ndarray): The image to save.
        save_path (str): The path where the image will be saved.
    """
    try:
        if len(image.shape) == 2:
            # Grayscale image
            Image.fromarray(image).save(save_path)#
        elif image.shape[2] == 4:
            # Convert BGRA to RGBA for saving
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            Image.fromarray(image).save(save_path)#
        elif image.shape[2] == 3:
            # Convert BGR to RGB for saving
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(image).save(save_path)

        else:
            logging.error("Unsupported image format for saving.")
            raise ValueError("Unsupported image format for saving.")

    except Exception as e:
        logging.error(f"An error occurred while saving the image to {save_path}: {e}")
        raise

def ensure_directory(path: Path):
    """
    Ensures that the specified directory exists. Creates it if it doesn't.

    Parameters:
        path (Path): The directory path to ensure.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory ensured at: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


def load_image(path, mode=cv2.IMREAD_UNCHANGED):
    """
    Loads an image from the specified path using OpenCV, with Pillow as a fallback.

    Parameters:
        path (str or Path): Path to the image file.
        mode (int): OpenCV flag for image loading mode.

    Returns:
        np.ndarray: Loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    path = Path(path)
    try:
        image = cv2.imread(str(path), mode)
        if image is not None:
            logging.info(f"Image loaded successfully from {path} using OpenCV.")
            return image
        else:
            # If OpenCV fails, use Pillow as a fallback
            image_pil = Image.open(path)
            image = np.array(image_pil)

            # Convert to BGR or BGRA if in RGB/RGBA mode for OpenCV compatibility
            if image_pil.mode == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image_pil.mode == "RGBA":
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

            logging.info(f"Image loaded successfully from {path} using Pillow.")
            return image
    except Exception as e:
        logging.error(f"Failed to load image from {path}. Error: {e}")
        raise FileNotFoundError(f"Image not found or cannot be loaded at {path}. Error: {e}")

# ---------------------
# Visualize Management
# ---------------------
    
def visualize_with_grid(combined_image, processed_logos, figsize=(14, 9), mode='depth', title="Combined Image with Logos and Coordinate Grid"):
    """
    Visualizes the combined image with a coordinate grid and logo positions.

    Parameters:
    - combined_image (np.ndarray): The combined image to visualize.
    - processed_logos (list): List of processed logos with their coordinates.
    - figsize (tuple): Figure size for the visualization.
    - mode (str): Visualization mode ('depth', 'gray', or 'color').
    - title (str): Title for the plot.
    """
    # Проверяем каналы изображения
    if combined_image.shape[2] == 4:
        # RGBA для matplotlib
        display_image = cv2.cvtColor(combined_image, cv2.COLOR_BGRA2RGBA)
    else:
        # RGB для matplotlib
        display_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=figsize)

    # Обрабатываем режимы отображения
    if mode == 'depth':
        # Преобразуем в градации серого, затем применяем цветовую карту magma
        gray_image = cv2.cvtColor(display_image, cv2.COLOR_RGBA2GRAY) if display_image.shape[2] == 4 else cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        im = ax.imshow(gray_image, cmap='magma')
        plt.colorbar(im, ax=ax)
    elif mode == 'gray':
        # Отображаем в градациях серого без colorbar
        gray_image = cv2.cvtColor(display_image, cv2.COLOR_RGBA2GRAY) if display_image.shape[2] == 4 else cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        ax.imshow(gray_image, cmap='gray')
    else:
        # По умолчанию отображаем в цвете
        ax.imshow(display_image)

    # Настраиваем заголовок и оси
    ax.set_title(title)
    ax.axis('on')

    # Определяем размеры изображения и шаг сетки
    height, width = combined_image.shape[:2]
    step_size = 50

    # Рисуем сетку
    for x in range(0, width, step_size):
        ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5)
    for y in range(0, height, step_size):
        ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)

    # Отображаем координаты логотипов
    for i, logo_info in enumerate(processed_logos):
        # Если 'coords' отсутствует, но есть 'point', создаем 'coords' на основе 'point'
        if 'coords' not in logo_info and 'point' in logo_info:
            logo_info['coords'] = logo_info['point']

        # Проверяем, что 'coords' существует
        if 'coords' in logo_info:
            x2, y2 = logo_info['coords']
            ax.plot(x2, y2, marker='o', markersize=5, color='blue')
            ax.text(
                x2 + 5, y2 + 5, f"Logo {i + 1}\n({x2}, {y2})", 
                color='blue', fontsize=10, backgroundcolor='white'
            )
        else:
            print(f"Warning: Logo {i + 1} does not have 'coords' or 'point'.")

    # Добавляем шкалы
    ax.set_xticks(range(0, width, step_size))
    ax.set_yticks(range(0, height, step_size))
    ax.set_xticklabels(range(0, width, step_size))
    ax.set_yticklabels(range(0, height, step_size))

    plt.tight_layout()
    plt.show()

