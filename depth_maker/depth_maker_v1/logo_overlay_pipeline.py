# depth_maker/depth_maker_v1/logo_overlay_pipeline.py

from .utils import (
    resize_image,
    add_alpha_channel,
    rotate_image,
    reflect_image,
    save_image,          # Import the utility function
    ensure_directory,
    display_image,
    setup_logging
)
import cv2
import os
import logging

class LogoOverlayPipeline:
    def __init__(self, 
                 background_path: str, 
                 logos_info: list, 
                 output_path: str,
                 background_size: tuple = (1280, 940)):
        """
        Initializes the LogoOverlayPipeline with background image, logos information, and output path.
        """
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.background_path = background_path
        self.logos_info = logos_info
        self.output_path = output_path
        self.background_size = background_size
        self.background = self.load_and_prepare_background()
        self.combined_image = self.background.copy()
    
    def load_and_prepare_background(self):
        """
        Loads and resizes the background image.
        
        Returns:
            np.ndarray: The prepared background image.
        """
        if not os.path.exists(self.background_path):
            self.logger.error(f"Background image not found at {self.background_path}.")
            raise FileNotFoundError(f"Background image not found at {self.background_path}.")
        
        background = cv2.imread(self.background_path, cv2.IMREAD_COLOR)
        if background is None:
            self.logger.error(f"Failed to load background image from {self.background_path}.")
            raise ValueError(f"Failed to load background image from {self.background_path}.")
        
        background = resize_image(background, self.background_size, maintain_aspect_ratio=True, padding_color=[255, 255, 255])
        self.logger.info(f"Background image loaded and resized to {self.background_size}.")
        return background
    
    def process_logos(self):
        """
        Processes each logo by resizing, adding alpha channels, rotating/reflection if needed, and overlaying onto the background.
        """
        for idx, logo_info in enumerate(self.logos_info):
            logo_path = logo_info.get('path')
            size = logo_info.get('size', (100, 100))
            position = logo_info.get('position', (0, 0))
            alpha = logo_info.get('alpha', 1.0)
            rotation = logo_info.get('rotation', 0)
            reflection = logo_info.get('reflection', None)  # 'horizontal', 'vertical', or 'both'
            
            self.logger.debug(f"Processing logo {idx}: {logo_path}")
            
            # Load logo image with alpha channel
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None:
                self.logger.warning(f"Logo image not found at {logo_path}. Skipping.")
                continue
            
            # Ensure logo has alpha channel
            logo = add_alpha_channel(logo)
            
            # Resize logo
            logo = resize_image(logo, size, maintain_aspect_ratio=True, padding_color=[0, 0, 0])
            
            # Rotate if needed
            if rotation != 0:
                logo = rotate_image(logo, rotation)
            
            # Reflect if needed
            if reflection:
                logo = reflect_image(logo, mode=reflection)
            
            # Adjust alpha
            if alpha < 1.0:
                logo[:, :, 3] = (logo[:, :, 3] * alpha).astype(np.uint8)
            
            # Overlay logo onto background
            self.combined_image = self.overlay_logo(self.combined_image, logo, position)
            self.logger.info(f"Logo {idx} overlaid at position {position}.")
    
    def overlay_logo(self, background, logo, position):
        """
        Overlays a logo onto the background image at the specified position.
        
        Parameters:
            background (np.ndarray): The background image.
            logo (np.ndarray): The logo image with alpha channel.
            position (tuple): (x, y) coordinates where the logo will be placed.
        
        Returns:
            np.ndarray: The combined image.
        """
        x, y = position
        h_logo, w_logo = logo.shape[:2]
        h_bg, w_bg = background.shape[:2]
        
        if x >= w_bg or y >= h_bg:
            self.logger.warning(f"Logo position {position} is outside the background dimensions. Skipping overlay.")
            return background
        
        # Calculate the region of interest
        end_x = min(x + w_logo, w_bg)
        end_y = min(y + h_logo, h_bg)
        logo = logo[0:(end_y - y), 0:(end_x - x)]
        
        # Separate logo channels
        if logo.shape[2] == 4:
            logo_rgb = logo[:, :, :3]
            mask = logo[:, :, 3] / 255.0
            inv_mask = 1.0 - mask
            for c in range(0, 3):
                background[y:end_y, x:end_x, c] = (mask * logo_rgb[:, :, c] +
                                                   inv_mask * background[y:end_y, x:end_x, c])
        else:
            background[y:end_y, x:end_x] = logo
        
        return background
    

    def run_pipeline(self):
        """
        Runs the entire logo overlay pipeline.
        """
        self.logger.info("Starting logo overlay pipeline.")
        self.process_logos()
        self.save_combined_image()
        self.logger.info("Logo overlay pipeline completed successfully.")

    
    def save_combined_image(self):
        """
        Saves the combined image to the specified output path using the utility function.
        """
        try:
            ensure_directory(os.path.dirname(self.output_path))
            save_image(self.combined_image, self.output_path)
            self.logger.info(f"Combined image saved to {self.output_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save combined image to {self.output_path}: {e}")
