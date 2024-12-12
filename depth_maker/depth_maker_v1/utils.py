# depth_maker/depth_maker_v1/utils.py

import os
import cv2
import numpy as np
import requests
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# ---------------------
# Model Management
# ---------------------

def download_model(url: str, save_path: str):
    """
    Downloads a model from the specified URL to the given save path with a progress bar.

    Parameters:
        url (str): The URL to download the model from.
        save_path (str): The local path where the model will be saved.
    """
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

def resize_image(image: np.ndarray, target_size: tuple, maintain_aspect_ratio: bool = True, interpolation: int = cv2.INTER_AREA, padding_color: list = [0, 0, 0]):
    """
    Resizes an image to the target size. Optionally maintains aspect ratio by adding padding.

    Parameters:
        image (np.ndarray): The image to resize.
        target_size (tuple): The desired size as (width, height).
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio.
        interpolation (int): OpenCV interpolation method.
        padding_color (list): RGB color for padding borders.

    Returns:
        np.ndarray: The resized (and padded) image.
    """
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        logging.error("target_size must be a tuple of (width, height).")
        raise ValueError("target_size must be a tuple of (width, height).")

    if maintain_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(image, new_size, interpolation=interpolation)
        # Add padding to match target_size
        delta_w = target_w - new_size[0]
        delta_h = target_h - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        logging.info(f"Image resized to {new_size} with padding added.")
        return new_image
    else:
        resized = cv2.resize(image, target_size, interpolation=interpolation)
        logging.info(f"Image resized to {target_size} without maintaining aspect ratio.")
        return resized



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



def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image around its center by the specified angle.

    Parameters:
        image (np.ndarray): The image to rotate.
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated image.
    """
    if len(image.shape) < 2:
        logging.error("Input image must have at least 2 dimensions.")
        raise ValueError("Input image must have at least 2 dimensions.")

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # Compute the new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    logging.info(f"Image rotated by {angle} degrees. New size: ({new_w}, {new_h}).")
    return rotated


def reflect_image(image: np.ndarray, mode: str = 'horizontal') -> np.ndarray:
    """
    Reflects the image based on the specified mode.

    Parameters:
        image (np.ndarray): The image to reflect.
        mode (str): The reflection mode - 'horizontal', 'vertical', or 'both'.

    Returns:
        np.ndarray: The reflected image.
    """
    if mode == 'horizontal':
        reflected = cv2.flip(image, 1)
    elif mode == 'vertical':
        reflected = cv2.flip(image, 0)
    elif mode == 'both':
        reflected = cv2.flip(image, -1)
    else:
        logging.error("Invalid reflection mode. Choose from 'horizontal', 'vertical', or 'both'.")
        raise ValueError("Invalid reflection mode. Choose from 'horizontal', 'vertical', or 'both'.")
    
    logging.info(f"Image reflected {mode}.")
    return reflected


# ---------------------
# Visualization Helpers
# ---------------------

def display_image(image: np.ndarray, title: str = "Image", cmap: str = None, figsize: tuple = (10, 8)):
    """
    Displays an image using matplotlib.

    Parameters:
        image (np.ndarray): The image to display.
        title (str): The title of the plot.
        cmap (str, optional): Colormap for grayscale images.
        figsize (tuple, optional): Size of the figure.
    """
    plt.figure(figsize=figsize)
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap if cmap else 'gray')
    elif image.shape[2] == 4:
        # Convert BGRA to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        plt.imshow(image)
    elif image.shape[2] == 3:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        logging.error("Unsupported image format for display.")
        raise ValueError("Unsupported image format for display.")
    
    plt.title(title)
    plt.axis('off')
    plt.show()


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
    if len(image.shape) == 2:
        # Grayscale image
        cv2.imwrite(save_path, image)
    elif image.shape[2] == 4:
        # Convert RGBA to BGRA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(save_path, image)
    elif image.shape[2] == 3:
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
    else:
        logging.error("Unsupported image format for saving.")
        raise ValueError("Unsupported image format for saving.")
    logging.info(f"Image saved to {save_path}.")


def ensure_directory(path: str):
    """
    Ensures that the specified directory exists. Creates it if it doesn't.

    Parameters:
        path (str): The directory path to ensure.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory ensured at: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


# ---------------------
# Logging Utilities
# ---------------------

def setup_logging(log_level=logging.INFO, log_file: str = None):
    """
    Sets up the logging configuration.

    Parameters:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): If provided, logs will also be written to this file.
    """
    logger = logging.getLogger()
    if not logger.hasHandlers():
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        logging.info("Logging is set up.")
    else:
        logging.info("Logging is already configured.")