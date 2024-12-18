# depth_maker/depth_maker_v1/constructor_layered_image.py

from .utils import (
    resize_image,
    add_alpha_channel,
    rotate_image,
    reflect_image,
    save_image,
    ensure_directory,
    display_image,
    setup_logging,
    load_image  
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
                 enable_logging=True):

        # Use pathlib for cross-platform path handling
        self.dir_path = Path(dir_path) if dir_path else None
        self.combined_image_path = self.dir_path / Path(combined_image_path) if dir_path else Path(combined_image_path)
        self.bg_color = bg_color
        self.threshold = threshold
        

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enable_logging_flag = enable_logging

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
                self.background = self.resize_logo(self.background, background_size[0], background_size[1])
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


    def resize_logo(self, logo, target_width, target_height):
        """
        Resizes the logo to fit within the specified dimensions while maintaining aspect ratio.

        Parameters:
        - logo (np.ndarray): Logo image with or without alpha channel.
        - target_width (int): Desired width for the logo.
        - target_height (int): Desired height for the logo.

        Returns:
        - logo_resized (np.ndarray): Resized logo image with proper alpha channel.
        """
        logo = self.crop_logo(logo)  # Crop the logo to remove unnecessary background

        logo_pil = Image.fromarray(logo)
        try:
            logo_resized_pil = logo_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        except AttributeError:
            logo_resized_pil = logo_pil.resize((target_width, target_height), Image.ANTIALIAS)

        logo_resized = np.array(logo_resized_pil)
        return logo_resized

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
            target_width, target_height = logo_info['sizes']
            logo = self.resize_logo(logo, target_width, target_height)

            # Prepare the alpha channel for the logo
            logo = self.prepare_logo_alpha(logo)

            x2, y2 = logo_info['point']
            alpha = logo_info.get('alpha', 1.0)

            layered_image.add_logo(logo, x2, y2, alpha)

        return layered_image


    def save_combined_image(self, layered_image):
        """
        Saves the final combined image by rendering all layers.
        Utilizes the utility function `save_image`.
        """
        #try:
        combined_image = layered_image.render()
        #ensure_directory(self.combined_image_path.parent)
        save_image(combined_image, self.combined_image_path)
        self.logger.info(f"Combined image saved to {self.combined_image_path}.")
        #except Exception as e:
         #   self.logger.error(f"Failed to save combined image to {self.combined_image_path}. Error: {e}")
        #    print(f"Failed to save combined image to {self.combined_image_path}. Error: {e}")
        #    raise


    @staticmethod
    def visualize_with_grid(combined_image):
        """
        Visualizes the original background and the combined image with logos, with a coordinate grid overlay.

        Parameters:
        - combined_image (np.ndarray): Background image with the logos overlayed.
        """
        combined_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(combined_rgb)
        ax.set_title('Combined Image with Logos and Coordinate Grid')
        ax.axis('on')

        # Draw grid lines
        height, width = combined_image.shape[:2]
        step_size = 50  # Set the distance between grid lines

        # Draw vertical lines
        for x in range(0, width, step_size):
            ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5)
        # Draw horizontal lines
        for y in range(0, height, step_size):
            ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()



class LayeredImageObject:
    def __init__(self, background, logos=None, logger=None):
        self.background = background
        self.logos = logos if logos is not None else []
        self.logger = logger if logger else logging.getLogger(__name__)


    def add_logo(self, logo, x2, y2, alpha=1.0):
        """
        Adds a logo as a layer.

        Parameters:
        - logo (np.ndarray): Logo image with alpha channel.
        - x2, y2 (int): Coordinates defining the bottom-right corner to place the logo.
        - alpha (float): Transparency factor for the logo.
        """
        logo_height, logo_width = logo.shape[:2]
        self.logos.append({'image': logo, 'x2': x2, 'y2': y2, 'alpha': alpha, 'coords': (x2, y2), 'original_size': (logo_width, logo_height)})

    def save(self, filepath):
        """
        Saves the layered image object to disk.

        Parameters:
        - filepath (str): Path to save the object.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Layered image object saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Loads a layered image object from disk.

        Parameters:
        - filepath (str): Path to load the object from.

        Returns:
        - LayeredImageObject: Loaded layered image object.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def render(self, include_background=True):
        """
        Renders the final combined image by overlaying all logos on the background.

        Parameters:
        - include_background (bool): Whether to include the background in the final render or not.

        Returns:
        - combined_img (np.ndarray): The final combined image.
        """
        if include_background:
            combined_img = self.background.copy()
        else:
            combined_img = np.zeros_like(self.background)

        for logo_info in self.logos:
            logo = logo_info['image']
            x2, y2 = logo_info['x2'], logo_info['y2']
            alpha = logo_info['alpha']
            self.logger.debug(f"Overlaying logo at coordinates: ({x2}, {y2}) with alpha: {alpha}")
            #print(f"Overlaying logo at coordinates: ({x2}, {y2}) with alpha: {alpha}")
            combined_img = self.overlay_logo(combined_img, logo, x2, y2, alpha)
        return combined_img

    def overlay_logo(self, background, logo, x2, y2, alpha=1.0):
        """
        Overlays the logo onto the background image at the specified coordinates with transparency.
        Utilizes utility functions from `utils.py`.
        """
        try:
            logo_height, logo_width = logo.shape[:2]
            x1, y1 = x2 - logo_width, y2 - logo_height

            # Ensure coordinates are within the background bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(background.shape[1], x2)
            y2 = min(background.shape[0], y2)

            # Recalculate width and height after adjusting coordinates
            roi_width = x2 - x1
            roi_height = y2 - y1

            if roi_width <= 0 or roi_height <= 0:
                logging.warning(f"Logo position ({x2}, {y2}) is outside the background dimensions. Skipping overlay.")
                return background

            # Adjust logo size if it exceeds the ROI
            if logo_width > roi_width or logo_height > roi_height:
                logo = resize_image(
                    logo,
                    target_size=(roi_width, roi_height),
                    maintain_aspect_ratio=True,
                    padding_color=[0, 0, 0]
                )
                logo_height, logo_width = logo.shape[:2]

            # Adjust logo if it still exceeds the ROI
            logo = logo[:roi_height, :roi_width]

            # Check if the logo has an alpha channel
            if logo.shape[2] == 4:
                logo_rgb = logo[:, :, :3]
                mask = logo[:, :, 3] / 255.0 * alpha
                inv_mask = 1.0 - mask
                for c in range(0, 3):
                    background[y1:y2, x1:x2, c] = (mask * logo_rgb[:, :, c] +
                                                   inv_mask * background[y1:y2, x1:x2, c])
            else:
                # If no alpha channel, apply overall alpha
                background[y1:y2, x1:x2] = (alpha * logo + (1 - alpha) * background[y1:y2, x1:x2]).astype(np.uint8)

            return background
        except Exception as e:
            logging.error(f"Error in overlaying logo: {e}")
            raise