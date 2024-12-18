# depth_maker/depth_maker_v1/constructor_layered_image.py

import os
import cv2
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from PIL import Image
import requests
import warnings
import pickle

from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from depth_maker.dm_v1.constructor_layered_image import LayeredImageObject  # 


class StylizedLayeredImageObject:
    def __init__(self, layered_image_obj, background_params=None, logos_params=None, final_processing_params=None,
                 depth_model_checkpoint=None, depth_encoder='vitl', enable_logging=True):
        """
        Initialize StylizedLayeredImageObject with parameters for background and logos.
        Optionally, provide a depth model checkpoint to enable depth estimation.

        Parameters:
        - layered_image_obj (LayeredImageObject): The base layered image containing background and logos.
        - background_params (dict): Parameters for processing background image.
        - logos_params (list of dict): Parameters for each logo processing.
        - final_processing_params (dict): Post-processing parameters.
        - depth_model_checkpoint (str): Path to Depth-Anything-V2 model checkpoint.
        - depth_encoder (str): Encoder type for Depth-Anything-V2 ('vits', 'vitb', 'vitl', 'vitg').
        - enable_logging (bool): Whether to enable logging for this object.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enable_logging_flag = enable_logging
        if not enable_logging:
            self.logger.disabled = True

        self.layered_image_obj = layered_image_obj
        self.final_processing_params = final_processing_params or {}
        self.depth_model_checkpoint = depth_model_checkpoint
        self.depth_encoder = depth_encoder
        self.depth_model = None

        if self.depth_model_checkpoint:
            self.depth_model = self.load_depth_model(self.depth_model_checkpoint, self.depth_encoder)

        # Process the background
        self.background, self.background_depth = self.process_image(
            image=layered_image_obj.background,
            params=background_params if background_params else {},
            is_background=True
        )

        if self.background.shape[2] == 4:
            # Keep only the first three channels (BGR)
            self.background = self.background[:, :, :3]

        # Process each logo
        self.processed_logos = []
        num_logos = len(layered_image_obj.logos)
        if logos_params is None:
            logos_params = [{} for _ in range(num_logos)]
        elif len(logos_params) < num_logos:
            logos_params.extend([{} for _ in range(num_logos - len(logos_params))])

        for idx, (logo_info, logo_param) in enumerate(zip(layered_image_obj.logos, logos_params)):
            processed_logo, depth_map = self.process_image(
                image=logo_info['image'],
                params=logo_param,
                is_background=False
            )
            self.processed_logos.append({
                'image': processed_logo,
                'coords': logo_info['coords'],
                'alpha': logo_info['alpha'],
                'original_size': logo_info['original_size'],
                'depth_map': depth_map
            })

    def enable_logging_method(self, log_level=logging.INFO):
        """
        Enable logging for this object.
        """
        self.logger.setLevel(log_level)
        self.logger.disabled = False
        self.enable_logging_flag = True
        self.logger.info("Логирование включено.")

    def disable_logging_method(self):
        """
        Disable logging for this object.
        """
        self.logger.disabled = True
        self.enable_logging_flag = False
        # Используем print, т.к. логирование отключено
        print("Логирование отключено.")

    def load_depth_model(self, checkpoint_path: str, encoder='vitl', device: str = "cuda" if torch.cuda.is_available() else "cpu") -> DepthAnythingV2:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if encoder not in model_configs:
            self.logger.error(f"Unsupported encoder type: {encoder}. Supported types: {list(model_configs.keys())}")
            raise ValueError(f"Unsupported encoder type: {encoder}.")

        model = DepthAnythingV2(**model_configs[encoder])

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.logger.info(f"Depth model loaded with encoder '{encoder}' on device '{device}'.")
        return model

    def process_image(self, image, params, is_background=False):
        method = params.get('method', 'none')
        brightness = params.get('brightness', 0)
        contrast = params.get('contrast', 1.0)
        rotation_angle = params.get('rotation_angle', 0)
        reflection = params.get('reflection', 'none')

        original_has_alpha = (image.shape[2] == 4)
        if original_has_alpha:
            b, g, r, original_alpha = cv2.split(image)
            original_rgb = cv2.merge((b, g, r))
        else:
            original_rgb = image
            original_alpha = None

        depth_map = None

        if method == 'none':
            processed = original_rgb.copy()
        elif method == 'lineart':
            self.logger.info("Applying lineart method.")
            processed = self.generate_lineart(
                original_rgb,
                gaussian_sigma=params.get('gaussian_sigma', 6.0),
                intensity_threshold=params.get('intensity_threshold', 8),
                detect_resolution=params.get('detect_resolution', 0),
                upscale_method=params.get('upscale_method', 'INTER_CUBIC')
            )
        elif method == 'canny':
            self.logger.info("Applying canny method.")
            processed = self.generate_canny(
                original_rgb,
                low_threshold=params.get('low_threshold', 100),
                high_threshold=params.get('high_threshold', 200),
                detect_resolution=params.get('detect_resolution', 0),
                upscale_method=params.get('upscale_method', 'INTER_CUBIC')
            )
        elif method == 'depth_anth':
            if not self.depth_model:
                self.logger.error("Depth model not loaded. Can't apply 'depth_anth'.")
                raise ValueError("Depth model not loaded.")
            self.logger.info("Generating depth map...")
            depth_input = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
            depth_map = self.depth_model.infer_image(depth_input)
            self.logger.info("Depth map generation complete.")

            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5) * 255
            depth_norm = depth_norm.astype(np.uint8)
            depth_norm = self.apply_brightness_contrast(depth_norm, brightness=brightness, contrast=contrast)

            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

            if original_has_alpha:
                depth_rgba = cv2.merge((depth_colormap, original_alpha))
            else:
                depth_rgba = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2BGRA)
                depth_rgba[:, :, 3] = 255
            processed = depth_rgba
        else:
            self.logger.error(f"Unknown processing method: {method}")
            raise ValueError(f"Unknown processing method: {method}")

        if method != 'depth_anth':
            processed = self.apply_brightness_contrast(processed, brightness=brightness, contrast=contrast)

        if method != 'depth_anth':
            if method == 'none' and original_has_alpha:
                if processed.shape[:2] != original_alpha.shape[:2]:
                    original_alpha = cv2.resize(original_alpha, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA)
                rgba = cv2.merge((processed[:, :, 0], processed[:, :, 1], processed[:, :, 2], original_alpha))
            else:
                if method in ['lineart', 'canny']:
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                elif method == 'depth_anth':
                    mask = None

                if method in ['lineart', 'canny']:
                    if original_has_alpha:
                        if original_alpha.shape[:2] != processed.shape[:2]:
                            original_alpha = cv2.resize(original_alpha, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA)
                        combined_alpha = cv2.bitwise_and(original_alpha, mask)
                    else:
                        combined_alpha = mask
                    rgba = cv2.merge((processed[:, :, 0], processed[:, :, 1], processed[:, :, 2], combined_alpha))
                else:
                    rgba = processed.copy()

        #    if not is_background and rotation_angle != 0:
        #       rgba = self.rotate_rgba(rgba, rotation_angle)

        #   if reflection != 'none':
        #       rgba = self.reflect_image(rgba, reflection)
        else:
            rgba = processed
            
        # После того, как вы установили rgba, вынесите логику вращения и отражения вне зависимости от метода:
        if not is_background and rotation_angle != 0:
            rgba = self.rotate_rgba(rgba, rotation_angle)

        if reflection != 'none':
            rgba = self.reflect_image(rgba, reflection)
        
        return rgba, depth_map

    def rotate_rgba(self, rgba_image, angle):
        (h, w) = rgba_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            rgba_image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        self.logger.debug(f"Rotated image by {angle} degrees.")
        return rotated

    def reflect_image(self, rgba_image, mode):
        if mode == 'horizontal':
            self.logger.debug("Reflecting image horizontally.")
            return cv2.flip(rgba_image, 1)
        elif mode == 'vertical':
            self.logger.debug("Reflecting image vertically.")
            return cv2.flip(rgba_image, 0)
        elif mode == 'both':
            self.logger.debug("Reflecting image both horizontally and vertically.")
            return cv2.flip(rgba_image, -1)
        return rgba_image

    def apply_brightness_contrast(self, image, brightness=0, contrast=1.0):
        img_float = image.astype(np.float32)
        img_float = img_float * contrast + brightness
        img_float = np.clip(img_float, 0, 255).astype(np.uint8)
        return img_float

    def generate_lineart(self, input_image, gaussian_sigma=6.0, intensity_threshold=8, detect_resolution=0, upscale_method="INTER_CUBIC"):
        self.logger.debug("Generating line art.")
        input_image, _ = self.common_input_validate(input_image, "np")
        input_image, remove_pad = self.resize_image_with_pad(input_image, detect_resolution, upscale_method)

        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), gaussian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        valid_pixels = intensity[intensity > intensity_threshold]
        if len(valid_pixels) == 0:
            median_intensity = 1.0
        else:
            median_intensity = np.median(valid_pixels)
        if median_intensity < 1e-5:
            median_intensity = 1.0
        intensity /= max(16, median_intensity)
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)

        detected_map = self.HWC3(remove_pad(detected_map))
        return detected_map

    def generate_canny(self, input_image, low_threshold=100, high_threshold=200, detect_resolution=0, upscale_method="INTER_CUBIC"):
        self.logger.debug("Generating canny edges.")
        input_image, _ = self.common_input_validate(input_image, "np")
        input_image, remove_pad = self.resize_image_with_pad(input_image, detect_resolution, upscale_method)

        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        detected_map = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        detected_map = self.HWC3(remove_pad(detected_map))
        return detected_map

    def common_input_validate(self, input_image, output_type, **kwargs):
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use input_image=... instead.", DeprecationWarning)
            input_image = kwargs.pop("img")

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"

        if type(output_type) is bool:
            warnings.warn("Passing True or False to output_type is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if input_image is None:
            self.logger.error("input_image must be defined.")
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        return (input_image, output_type)

    def resize_image_with_pad(self, input_image, resolution, upscale_method="", skip_hwc3=False, mode='edge'):
        if skip_hwc3:
            img = input_image
        else:
            img = self.HWC3(input_image)
        H_raw, W_raw, _ = img.shape
        if resolution == 0:
            return img, lambda x: x
        k = float(resolution) / float(min(H_raw, W_raw))
        H_target = int(np.round(float(H_raw) * k))
        W_target = int(np.round(float(W_raw) * k))
        img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        H_pad, W_pad = self.pad64(H_target), self.pad64(W_target)
        img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

        def remove_pad(x):
            return x[:H_target, :W_target, ...]

        return img_padded, remove_pad

    def HWC3(self, x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C in [1, 3, 4]
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y

    def pad64(self, x):
        return (64 - x % 64) % 64

    def overlay_logo(self, background, logo, x2, y2, alpha=1.0, original_width=None, original_height=None):
        logo_height, logo_width = logo.shape[:2]
        x1, y1 = x2 - logo_width, y2 - logo_height

        self.logger.debug(f"Logo dimensions: width={logo_width}, height={logo_height}")
        self.logger.debug(f"Initial placement coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(background.shape[1], x2)
        y2 = min(background.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            self.logger.warning("Logo position adjusted outside background dimensions. Skipping overlay.")
            return background

        self.logger.debug(f"Adjusted placement coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        roi_width = x2 - x1
        roi_height = y2 - y1
        self.logger.debug(f"Calculated ROI dimensions: width={roi_width}, height={roi_height}")

        if logo_width > roi_width or logo_height > roi_height:
            self.logger.info(f"Resizing logo from ({logo_width}, {logo_height}) to fit within ROI ({roi_width}, {roi_height})")
            logo = cv2.resize(logo, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
            logo_height, logo_width = logo.shape[:2]
            self.logger.debug(f"Resized logo dimensions: width={logo_width}, height={logo_height}")

        roi = background[y1:y2, x1:x2].astype(np.float32)

        if logo.shape[2] == 4:
            logo_bgr = logo[:, :, :3].astype(np.float32)
            logo_alpha_channel = (logo[:, :, 3] / 255.0) * alpha
            logo_alpha = cv2.merge([logo_alpha_channel, logo_alpha_channel, logo_alpha_channel])
        else:
            logo_bgr = logo[:, :, :3].astype(np.float32)
            logo_gray = cv2.cvtColor(logo_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            logo_mask = (logo_gray > 0).astype(np.float32) * alpha
            logo_alpha = cv2.merge([logo_mask, logo_mask, logo_mask])

        blended_roi = logo_bgr * logo_alpha + roi * (1 - logo_alpha)
        blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

        combined_img = background.copy()
        combined_img[y1:y2, x1:x2] = blended_roi
        return combined_img

    def generate_depth_anything(self):
        if not self.depth_model:
            self.logger.error("Depth model not loaded. Cannot generate depth.")
            raise ValueError("Depth model not loaded.")

        background_method = self.layered_image_obj.background.get('method', 'none') if isinstance(self.layered_image_obj.background, dict) else 'none'
        if background_method == 'depth_anth':
            self.logger.info("Generating depth for background...")
            background_bgr = cv2.cvtColor(self.background, cv2.COLOR_BGRA2BGR) if self.background.shape[2] == 4 else self.background.copy()
            self.background_depth = self.depth_model.infer_image(background_bgr)
            self.logger.info("Depth generation for background complete.")
        else:
            self.background_depth = None

        for idx, logo_info in enumerate(self.processed_logos):
            logo_method = self.layered_image_obj.logos[idx].get('method', 'none') if isinstance(self.layered_image_obj.logos[idx], dict) else 'none'
            if logo_method == 'depth_anth':
                self.logger.info(f"Generating depth for logo {idx}...")
                logo_rgba = logo_info['image']
                logo_bgr = cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2BGR) if logo_rgba.shape[2] == 4 else logo_rgba.copy()
                depth_map = self.depth_model.infer_image(logo_bgr)

                if logo_rgba.shape[2] == 4:
                    mask = (logo_rgba[:, :, 3] > 0).astype(np.float32)
                    depth_map = depth_map * mask

                self.processed_logos[idx]['depth_map'] = depth_map
                self.logger.info(f"Depth generation for logo {idx} complete.")
            else:
                self.processed_logos[idx]['depth_map'] = None

    def render(self):
        combined_img = self.background.copy()

        for logo_info in self.processed_logos:
            logo = logo_info['image']
            x2, y2 = logo_info['coords']
            alpha = logo_info['alpha']
            original_width, original_height = logo_info['original_size']
            self.logger.debug(f"Overlaying logo at coordinates: ({x2}, {y2}) with alpha: {alpha}")
            combined_img = self.overlay_logo(combined_img, logo, x2, y2, alpha, original_width, original_height)

        combined_img = self.final_post_processing(combined_img)
        return combined_img

    def final_post_processing(self, combined_img):
        method = self.final_processing_params.get('method', 'none')
        if method == 'none':
            return combined_img
        elif method == 'threshold':
            threshold_value = self.final_processing_params.get('threshold_value', 128)
            gray = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return mask_bgr
        else:
            self.logger.error(f"Unknown final processing method: {method}")
            raise ValueError(f"Unknown final processing method: {method}")

    def visualize_with_grid(self, title="Combined Image with Logos and Coordinate Grid"):
        combined_image = self.render()
        combined_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(combined_rgb)
        ax.set_title(title)
        ax.axis('on')

        height, width = combined_image.shape[:2]
        step_size = 50

        for x in range(0, width, step_size):
            ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5)
        for y in range(0, height, step_size):
            ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    def save_combined_image(self, filepath):
        combined_image = self.render()
        self.logger.info("Combined image rendered.")

        if combined_image.shape[2] == 4:
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGRA2RGBA)
        elif combined_image.shape[2] == 3:
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        Image.fromarray(combined_image).save(filepath)
        self.logger.info(f"Combined image saved to {filepath}")


    def save_stylized_layered_image_object(self, filepath: str):
        """
        Saves a new LayeredImageObject derived from the current stylized state.

        This will create a new LayeredImageObject using the processed background and logos, 
        allowing you to load it later and apply further brightness/contrast adjustments without 
        re-running expensive operations like depth estimation.
        """
        # Создаем новый layered image object из текущего фонового изображения и обработанных логотипов
        stylized_layered_image = LayeredImageObject(self.background)  # фон уже в BGR

        for logo_info in self.processed_logos:
            stylized_layered_image.add_logo(
                logo_info['image'],
                logo_info['coords'][0],
                logo_info['coords'][1],
                logo_info['alpha']
            )

        # Сохраняем объект
        stylized_layered_image.save(filepath)
        self.logger.info(f"Stylized layered image object saved to {filepath}")
