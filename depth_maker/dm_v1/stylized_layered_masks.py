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
from .constructor_layered_image import LayeredImageObject  # 
from .utils import visualize_with_grid

class StylizedLayeredImageObject:
    def __init__(self, layered_image_obj, background_params=None, logos_params=None, final_processing_params=None,
                 depth_model_checkpoint=None, depth_encoder='vitl', enable_logging=True):
        """
        Initialize StylizedLayeredImageObject with parameters for background and logos.
        Optionally, provide a depth model checkpoint to enable depth estimation.

        Parameters:
        - layered_image_obj (LayeredImageObject): The base layered image containing background and logos.
        - background_params (dict): Parameters for processing background image.
        - logos_params (list of dict): Parameters for each logo processing, including sizes, resize_by, and point.
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

        # Load depth model if needed
        if not self.depth_model and self.depth_model_checkpoint:
            self.depth_model = self.load_depth_model(self.depth_model_checkpoint, self.depth_encoder)

        # Добавляем значения по умолчанию для final_processing_params
        default_processing_params = {
            "method": "none",
            "colormap": None,       # По умолчанию цветовая карта не применяется
            "resolution_scale": 1.0,  # Масштабирование
            "invert": False          # Инверсия
        }
        self.final_processing_params = {**default_processing_params, **(final_processing_params or {})}


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
        
        # Ensure logos_params has the correct length by extending it with default dictionaries if necessary
        if logos_params is None:
            logos_params = [{} for _ in range(num_logos)]
        elif len(logos_params) < num_logos:
            logos_params.extend([{} for _ in range(num_logos - len(logos_params))])

        # Process each logo with its respective parameters
        for idx, (logo_info, logo_param) in enumerate(zip(layered_image_obj.logos, logos_params)):
            # Handle resizing by `sizes` and `resize_by`
            target_size = logo_param.get('sizes', None)
            resize_by = logo_param.get('resize_by', 'width')
            if target_size:
                logo_info['image'] = self.resize_logo(logo_info['image'], target_size, resize_by)

            # Handle position update (`point`)
            point = logo_param.get('point', None)
            if point:
                self.logger.info(f"Updating position for logo {idx} to {point}.")
                logo_info['coords'] = point

            # Process the logo image
            processed_logo, depth_map = self.process_image(
                image=logo_info['image'],
                params=logo_param,
                is_background=False
            )

            # Append the processed logo data
            self.processed_logos.append({
                'image': processed_logo,
                'coords': logo_info['coords'],
                'alpha': logo_info['alpha'],
                'final_size': (processed_logo.shape[1], processed_logo.shape[0]),
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
        self.logger.setLevel(logging.NOTSET)
        self.logger.disabled = True
        self.enable_logging_flag = False
        # Используем print, т.к. логирование отключено
        print("Логирование отключено.")

    def load_depth_model(self, checkpoint_path: str, encoder='vitl', device: str = "cuda" if torch.cuda.is_available() else "cpu") -> DepthAnythingV2:
        """
        Load the Depth-Anything-V2 model.

        Parameters:
        - checkpoint_path (str): Path to the model checkpoint.
        - encoder (str): Encoder type.
        - device (str): Device to load the model on.

        Returns:
        - model (DepthAnythingV2): Loaded depth model.
        """     
        
        
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
        """
        Process a single image (background or logo) according to the specified parameters.
        Steps:
        - Extract original alpha if present.
        - Apply chosen method (none, lineart, canny, depth_anth).
        - Apply brightness/contrast (for non-depth methods or after depth colormap).
        - Reintroduce alpha channel (for lineart/canny/depth_anth, via a thresholded mask or existing alpha).
        - If logo (not background), apply rotation if specified.
        - Apply reflection if specified.
        - For 'depth_anth' method, replace image with depth colormap and apply brightness/contrast.
    
        Returns:
        - processed_rgba (np.ndarray): The processed image (either stylized or depth colormap).
        - depth_map (np.ndarray or None): The depth map if 'depth_anth' is specified, else None.
        """
        # Значения по умолчанию для параметров
        default_params = {
            "method": "none",
            "colormap": None,
            "resolution_scale": 1.0,
            "invert": False,
            "brightness": 0,
            "contrast": 1.0,
            "rotation_angle": 0,
            "reflection": "none",
        }
        # Объединение параметров с их значениями по умолчанию
        params = {**default_params, **params}
    
        # Извлечение параметров обработки
        method = params["method"]
        brightness = params["brightness"]
        contrast = params["contrast"]
        rotation_angle = params["rotation_angle"]
        reflection = params["reflection"]
    
        # Проверка на наличие альфа-канала
        original_has_alpha = image.shape[2] == 4
        if original_has_alpha:
            b, g, r, original_alpha = cv2.split(image)
            original_rgb = cv2.merge((b, g, r))
        else:
            original_rgb = image
            original_alpha = None
    
        # Инициализация карты глубины
        depth_map = None
    
        # 1. Применение метода обработки
        if method == "none":
            processed = original_rgb.copy()
        elif method == "lineart":
            self.logger.info("Applying lineart method.")
            processed = self.generate_lineart(
                original_rgb,
                gaussian_sigma=params.get("gaussian_sigma", 6.0),
                intensity_threshold=params.get("intensity_threshold", 8),
                detect_resolution=params.get("detect_resolution", 0),
                upscale_method=params.get("upscale_method", "INTER_CUBIC"),
            )
        elif method == "canny":
            self.logger.info("Applying canny method.")
            processed = self.generate_canny(
                original_rgb,
                low_threshold=params.get("low_threshold", 100),
                high_threshold=params.get("high_threshold", 200),
                detect_resolution=params.get("detect_resolution", 0),
                upscale_method=params.get("upscale_method", "INTER_CUBIC"),
            )
        elif method == "depth_anth":
            self.logger.info("Applying depth_anth method.")
            depth_input = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
            depth_map = self._infer_depth(depth_input, model_type="depth_anything")
    
            # Применение маски альфа-канала (если есть)
            if original_has_alpha:
                alpha_mask = (original_alpha > 0).astype(np.float32)
                depth_map *= alpha_mask
    
            # Обработка карты глубины
            depth_map = self._process_depth_map(depth_map, params)
    
            # Преобразование глубины в RGBA
            processed = cv2.cvtColor(depth_map, cv2.COLOR_BGR2BGRA)
            processed[:, :, 3] = original_alpha if original_has_alpha else 255
        else:
            self.logger.error(f"Unknown processing method: {method}")
            raise ValueError(f"Unsupported processing method: {method}")
    
        # 2. Применение яркости и контраста (если не 'depth_anth')
        if method != "depth_anth":
            processed = self.apply_brightness_contrast(processed, brightness=brightness, contrast=contrast)
    
        # 3. Восстановление альфа-канала для методов, отличных от 'depth_anth'
        if method != "depth_anth":
            if original_has_alpha:
                if processed.shape[:2] != original_alpha.shape[:2]:
                    original_alpha = cv2.resize(original_alpha, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA)
                rgba = cv2.merge((processed[:, :, 0], processed[:, :, 1], processed[:, :, 2], original_alpha))
            else:
                rgba = processed
        else:
            rgba = processed
    
        # Применение вращения
        if not is_background and rotation_angle != 0:
            rgba = self.rotate_rgba(rgba, rotation_angle)
    
        # Применение отражения
        if reflection != "none":
            rgba = self.reflect_image(rgba, reflection)
    
        return rgba, depth_map
    


    def rotate_rgba(self, rgba_image, angle):

        """
        Rotates the given RGBA image around its center by the specified angle.

        Parameters:
        - rgba_image (np.ndarray): RGBA image to rotate.
        - angle (float): Angle in degrees. Positive values rotate counterclockwise.

        Returns:
        - rotated (np.ndarray): Rotated RGBA image.
        """

        (h, w) = rgba_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        # Rotate with BORDER_CONSTANT and a transparent (0,0,0,0) background
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
        """
        Reflects the given RGBA image according to the mode:
        - 'horizontal': reflect left-right
        - 'vertical': reflect top-bottom
        - 'both': reflect both horizontally and vertically
        - 'none': no reflection
        """
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


    def resize_logo(self, logo, target_size, resize_by="width"):
        """
        Resizes the logo while maintaining aspect ratio.
        """
        # Code for resizing the logo
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
    
        return np.array(logo_resized_pil)


    
    def apply_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """
        Adjusts brightness and contrast of the given image.

        brightness: int, positive to brighten, negative to darken
        contrast: float, >1.0 increases contrast, between 0 and 1 decreases contrast, 1.0 no change
        """
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

    
    def generate_depth_anything(self):
        """
        Генерирует карты глубины для всех объектов (фон и логотипы).
        """
        # Обрабатываем фон
        if self.background_method == 'depth_anth':
            self.logger.info("Processing depth for background...")
            depth_input = cv2.cvtColor(self.background, cv2.COLOR_BGR2RGB)
            depth_map = self._infer_depth(depth_input, model_type='depth_anything')
            self.background_depth = self._process_depth_map(depth_map, self.final_processing_params)
    
        # Обрабатываем логотипы
        for idx, logo_info in enumerate(self.processed_logos):
            if logo_info.get('method') == 'depth_anth':
                self.logger.info(f"Processing depth for logo {idx}...")
                depth_input = cv2.cvtColor(logo_info['image'], cv2.COLOR_BGRA2RGB)
                depth_map = self._infer_depth(depth_input, model_type='depth_anything')
                logo_info['depth_map'] = self._process_depth_map(depth_map, logo_info.get('params', {}))

    
    def _infer_depth(self, input_image, model_type='depth_anything', model_params=None):
        """
        Универсальный метод для инференса глубины.
    
        Parameters:
        - input_image (np.ndarray): Входное изображение (RGB).
        - model_type (str): Тип модели глубины ('depth_anything', 'another_model').
        - model_params (dict): Дополнительные параметры для модели.
    
        Returns:
        - depth_map (np.ndarray): Сгенерированная карта глубины.
        """
        if model_type == 'depth_anything':
            # Используем DepthAnythingV2
            if not self.depth_model:
                raise ValueError("Depth model is not loaded. Provide a valid checkpoint.")
            self.logger.debug(f"Inferring depth using {model_type}...")
            depth_map = self.depth_model.infer_image(input_image)
        elif model_type == 'another_model':
            # Пример для другой модели
            self.logger.debug("Inferring depth using another model...")
            # Здесь можно вызвать другой метод/модель для инференса глубины
            depth_map = some_other_model_inference(input_image, **(model_params or {}))
        else:
            raise ValueError(f"Unsupported depth model type: {model_type}")
    
        return depth_map

    
    def _process_depth_map(self, depth_map, params):
        """
        Обрабатывает карту глубины с учётом заданных параметров.
        
        Parameters:
        - depth_map (np.ndarray): Карта глубины для обработки.
        - params (dict): Параметры обработки (resolution_scale, invert, colormap).
        
        Returns:
        - processed_depth (np.ndarray): Обработанная карта глубины.
        """
        if depth_map is None or not isinstance(depth_map, np.ndarray):
            self.logger.warning("Invalid depth map provided. Returning unprocessed depth map.")
            return depth_map
    
        # Normalize depth map to 0-255
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
        depth_map = (depth_map * 255).astype(np.uint8)
    
        # Apply inversion if specified
        if params.get("invert", False):
            self.logger.debug("Inverting depth map.")
            depth_map = 255 - depth_map
    
        # Apply resolution scaling
        resolution_scale = params.get("resolution_scale", 1.0)
        if resolution_scale != 1.0:
            self.logger.debug(f"Scaling depth map with resolution scale: {resolution_scale}.")
            new_width = int(depth_map.shape[1] * resolution_scale)
            new_height = int(depth_map.shape[0] * resolution_scale)
            depth_map = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
        # Apply colormap if specified
        colormap_name = params.get("colormap", None)
        if colormap_name:
            colormap_dict = {
                'magma': cv2.COLORMAP_MAGMA,
                'jet': cv2.COLORMAP_JET,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'plasma': cv2.COLORMAP_PLASMA,
                'autumn': cv2.COLORMAP_AUTUMN,
                'bone': cv2.COLORMAP_BONE,
                'cool': cv2.COLORMAP_COOL,
                'hot': cv2.COLORMAP_HOT,
                'hsv': cv2.COLORMAP_HSV,
                'inferno': cv2.COLORMAP_INFERNO,
                'parula': cv2.COLORMAP_PARULA,
                'pink': cv2.COLORMAP_PINK,
                'spring': cv2.COLORMAP_SPRING,
                'summer': cv2.COLORMAP_SUMMER,
                'turbo': cv2.COLORMAP_TURBO,
                'twilight': cv2.COLORMAP_TWILIGHT,
                'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
                'winter': cv2.COLORMAP_WINTER
            }
            colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_MAGMA)
            self.logger.debug(f"Applying colormap: {colormap_name}.")
            depth_map = cv2.applyColorMap(depth_map, colormap)
    
         # Если всё ещё одноканальное, делаем 3 канала
        if depth_map.ndim == 2:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        
        return depth_map


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

            
    def overlay_logo(self, background, logo, x2, y2, alpha=1.0):
        """
        Overlays the processed logo onto the background image (BGR) at the specified
        bottom-right coordinates (x2, y2) with partial cropping instead of resizing.
        """
        # Проверяем корректность входных данных
        if len(background.shape) < 3 or len(logo.shape) < 3:
            self.logger.error(f"Background or logo has invalid shape. Background: {background.shape}, Logo: {logo.shape}")
            raise ValueError("Invalid shape for background or logo.")
    
        try:
            logo_height, logo_width = logo.shape[:2]
            # Нижний правый угол лого (x2,y2) => вычисляем верхний левый (x1,y1)
            x1 = x2 - logo_width
            y1 = y2 - logo_height
        except Exception as e:
            self.logger.error(f"Error overlaying logo: {e}")
            raise
    
        # Фон (background) имеет размер [h, w]
        h_bg, w_bg = background.shape[:2]
        
        # Координаты, ограниченные (clamped) размерами фона
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w_bg, x2)
        crop_y2 = min(h_bg, y2)
    
        # Если после обрезки логотип вообще не попадает на фон, можно вернуть background как есть
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            self.logger.debug("Logo is completely outside the background — nothing to overlay.")
            return background.copy()
    
        # Теперь вычислим, какой участок самого массива `logo` надо взять,
        # чтобы он соответствовал "врезающейся" части:
        # Сдвиг по x, если x1 < 0, значит логотип «вылез» слева => нужно обрезать.
        # Аналогично, если x2 > w_bg, логотип «вылез» справа.
        logo_crop_x1 = max(0, -x1)               # сколько «проскочили» слева
        logo_crop_y1 = max(0, -y1)               # сколько «проскочили» сверху
        logo_crop_x2 = logo_width - max(0, x2 - w_bg)   # обрезаем справа, если x2 > w_bg
        logo_crop_y2 = logo_height - max(0, y2 - h_bg)  # обрезаем снизу, если y2 > h_bg
    
        # Получаем именно ту часть лого, которая попадает на фон
        logo_cropped = logo[logo_crop_y1:logo_crop_y2, logo_crop_x1:logo_crop_x2]
    
        # Формируем ROI (участок фона) по тем же координатам
        roi_width = crop_x2 - crop_x1
        roi_height = crop_y2 - crop_y1
        roi = background[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
    
        # Убеждаемся, что размеры ROI и обрезанного лого совпадают
        ch, cw = logo_cropped.shape[:2]
        if ch != roi_height or cw != roi_width:
            self.logger.error(
                f"Mismatch in ROI size and cropped logo size: "
                f"ROI=({roi_height},{roi_width}), Logo=({ch},{cw})"
            )
            return background.copy()
    
        # Разделяем каналы
        if logo_cropped.shape[2] == 4:
            # BGRA => BGR + alpha
            logo_bgr = logo_cropped[:, :, :3].astype(np.float32)
            logo_alpha_channel = (logo_cropped[:, :, 3] / 255.0) * alpha
            logo_alpha = cv2.merge([logo_alpha_channel] * 3)
        else:
            # BGR или что-то ещё
            logo_bgr = logo_cropped.astype(np.float32)
            logo_alpha = np.ones(logo_bgr.shape, dtype=np.float32) * alpha
    
        # Смешиваем ROI и лого
        blended_roi = logo_bgr * logo_alpha + roi * (1.0 - logo_alpha)
        blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
    
        # Копируем результат обратно
        combined_img = background.copy()
        combined_img[crop_y1:crop_y2, crop_x1:crop_x2] = blended_roi
    
        return combined_img

    

        
    def render(self, include_background=True, crop_to_background=True):
        """
        Рендерит итоговое комбинированное изображение с учетом возможности обрезки по фону.
        """
        self.logger.info("Starting render process.")
    
        # Определяем размеры холста
        try:
            max_x = max((logo['coords'][0] for logo in self.processed_logos), default=0)
            max_y = max((logo['coords'][1] for logo in self.processed_logos), default=0)
        except Exception as e:
            self.logger.error(f"Error calculating canvas size: {e}")
            raise
    
        self.canvas_width = max(max_x, self.background.shape[1])
        self.canvas_height = max(max_y, self.background.shape[0])
    
        self.logger.info(f"Canvas size determined: {self.canvas_width}x{self.canvas_height}")
    
        # Создаем холст
        combined_img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
    
        # Добавляем фон
        if include_background:
            combined_img[:self.background.shape[0], :self.background.shape[1]] = self.background
            self.logger.info("Background added to canvas.")
    
        # Накладываем логотипы
        for idx, logo_info in enumerate(self.processed_logos):
            try:
                logo = logo_info['image']
                x2, y2 = logo_info['coords']
                alpha = logo_info['alpha']
                self.logger.info(f"Overlaying logo {idx} at ({x2}, {y2}) with alpha {alpha}.")
                combined_img = self.overlay_logo(combined_img, logo, x2, y2, alpha)
            except Exception as e:
                self.logger.error(f"Error overlaying logo {idx}: {e}")
                continue
    
        # Обрезаем по размеру фона, если указано
        if crop_to_background:
            combined_img = combined_img[:self.background.shape[0], :self.background.shape[1]]
            self.logger.info("Cropped combined image to background size.")
    
        # Проверяем валидность изображения
        if combined_img is None or len(combined_img.shape) < 3:
            self.logger.error(
                f"Render returned invalid image with shape: {combined_img.shape if combined_img is not None else 'None'}"
            )
            raise ValueError("Invalid image generated in render method.")
    
        # Вызов финальной пост-обработки, если метод не 'none'
        if self.final_processing_params.get("method", "none") != "none":
            combined_img = self.final_post_processing(combined_img)
            self.logger.info("Final post-processing applied.")
    
        self.logger.info(f"Render completed successfully with final image shape: {combined_img.shape}")
        return combined_img


        
    def final_post_processing(self, combined_img):
        """
        Применяет финальную обработку карты глубины.
        """
        if combined_img is None or not isinstance(combined_img, np.ndarray):
            self.logger.error("Invalid input image for final post-processing.")
            raise ValueError("Input image is None or not a valid numpy array.")
    
        self.logger.info(f"Starting final post-processing with image shape: {combined_img.shape}")
        
        # Параметры по умолчанию
        default_processing_params = {
            "method": "none",
            "colormap": None,
            "resolution_scale": 1.0,
            "invert": False
        }
    
        # Слияние параметров
        processing_params = {**default_processing_params, **self.final_processing_params}
    
        method = processing_params.get("method", "none")
        if method == "depth_anth":
            if not self.depth_model:
                depth_model_checkpoint = processing_params.get("depth_model_checkpoint")
                depth_encoder = processing_params.get("depth_encoder", "vitl")
                if not depth_model_checkpoint:
                    self.logger.error("Depth model checkpoint not provided. Cannot apply 'depth_anth'.")
                    raise ValueError("Depth model checkpoint not provided.")
                self.logger.info("Depth model not loaded. Attempting to load it now...")
                self.depth_model = self.load_depth_model(depth_model_checkpoint, depth_encoder)
            
            self.logger.info("Applying depth processing to the combined image.")
            depth_input = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            depth_map = self._infer_depth(depth_input, model_type='depth_anything')
            
            # Обработка depth_map
            combined_img = self._process_depth_map(depth_map, processing_params)
        
        self.logger.info(f"Final post-processing completed with image shape: {combined_img.shape}")
        return combined_img

    
    def visualize_with_grid(self, figsize=(14, 9), mode='depth', title="Combined Image with Logos and Coordinate Grid"):
        combined_image = self.render()
        visualize_with_grid(combined_image, self.processed_logos, figsize, mode, title)



    def save_combined_mask(self, filepath):
        """
        Parameters:
        - filepath (str): Path to save the image.
        - alpha (float): Transparency factor (not used here).
        """
        combined_image = self.render()
        self.logger.info("Combined image rendered.")

        # If the image has 4 channels (BGRA), convert to RGBA
        if combined_image.shape[2] == 4:
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGRA2RGBA)
        # If the image has 3 channels (BGR), convert to RGB
        elif combined_image.shape[2] == 3:
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        # save
        Image.fromarray(combined_image).save(filepath)
        self.logger.info(f"Combined mask saved to {filepath}")


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
