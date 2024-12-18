# depth_maker/depth_maker_v1/constructor_layered_image.py

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
import warnings
import pickle

from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


class StylizedLayeredImageObject:
    def __init__(self, layered_image_obj, background_params=None, logos_params=None, final_processing_params=None, depth_model_checkpoint=None, depth_encoder='vitl'):
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
        """
        self.layered_image_obj = layered_image_obj
        self.final_processing_params = final_processing_params or {}
        self.depth_model_checkpoint = depth_model_checkpoint
        self.depth_encoder = depth_encoder
        self.depth_model = None

        # Load depth model if checkpoint is provided
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
                'depth_map': depth_map  # Store depth map if generated
            })


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
            raise ValueError(f"Unsupported encoder type: {encoder}. Supported types: {list(model_configs.keys())}")

        model = DepthAnythingV2(**model_configs[encoder])

        # Set weights_only=True to resolve FutureWarning and enhance security
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Depth model loaded with encoder '{encoder}' on device '{device}'.")
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
        method = params.get('method', 'none')
        brightness = params.get('brightness', 0)
        contrast = params.get('contrast', 1.0)
        rotation_angle = params.get('rotation_angle', 0)
        reflection = params.get('reflection', 'none')

        # Check if original image has an alpha channel
        original_has_alpha = (image.shape[2] == 4)
        if original_has_alpha:
            # Extract original alpha
            b, g, r, original_alpha = cv2.split(image)
            original_rgb = cv2.merge((b, g, r))
        else:
            original_rgb = image
            original_alpha = None

        # Initialize depth_map
        depth_map = None

        # 1. Apply chosen method
        if method == 'none':
            processed = original_rgb.copy()
        elif method == 'lineart':
            processed = self.generate_lineart(
                original_rgb,
                gaussian_sigma=params.get('gaussian_sigma', 6.0),
                intensity_threshold=params.get('intensity_threshold', 8),
                detect_resolution=params.get('detect_resolution', 0),
                upscale_method=params.get('upscale_method', 'INTER_CUBIC')
            )
        elif method == 'canny':
            processed = self.generate_canny(
                original_rgb,
                low_threshold=params.get('low_threshold', 100),
                high_threshold=params.get('high_threshold', 200),
                detect_resolution=params.get('detect_resolution', 0),
                upscale_method=params.get('upscale_method', 'INTER_CUBIC')
            )
        elif method == 'depth_anth':
            # For depth_anth, generate depth map and replace image with depth colormap
            if not self.depth_model:
                raise ValueError("Depth model not loaded. Provide a valid model checkpoint during initialization.")

            print("Generating depth map...")
            # Ensure the image is in RGB for the depth model
            depth_input = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
            depth_map = self.depth_model.infer_image(depth_input)
            print("Depth map generation complete.")

            # Normalize the depth map to 0-255
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5) * 255
            depth_norm = depth_norm.astype(np.uint8)

            # Apply brightness and contrast adjustments to the normalized depth map
            depth_norm = self.apply_brightness_contrast(depth_norm, brightness=brightness, contrast=contrast)

            # Apply color map for visualization
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

            # If original image had alpha, retain it; else, create full opacity
            if original_has_alpha:
                depth_rgba = cv2.merge((depth_colormap, original_alpha))
            else:
                depth_rgba = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2BGRA)
                depth_rgba[:, :, 3] = 255  # Set alpha to fully opaque

            processed = depth_rgba
        else:
            raise ValueError(f"Unknown processing method: {method}")

        # 2. Apply brightness and contrast (if not 'depth_anth' as depth_colormap is already adjusted)
        if method != 'depth_anth':
            processed = self.apply_brightness_contrast(processed, brightness=brightness, contrast=contrast)

        # 3. Reintroduce alpha channel for non-depth methods
        if method != 'depth_anth':
            if method == 'none' and original_has_alpha:
                # Just reattach original alpha, resize if dimensions changed
                if processed.shape[:2] != original_alpha.shape[:2]:
                    original_alpha = cv2.resize(original_alpha, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA)
                rgba = cv2.merge((processed[:, :, 0], processed[:, :, 1], processed[:, :, 2], original_alpha))
            else:
                # For lineart/canny: create a binary mask or use existing alpha
                if method in ['lineart', 'canny']:
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                elif method == 'depth_anth':
                    # Already handled above
                    mask = None

                if method in ['lineart', 'canny']:
                    if original_has_alpha:
                        # Combine mask with original_alpha
                        if original_alpha.shape[:2] != processed.shape[:2]:
                            original_alpha = cv2.resize(original_alpha, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA)
                        combined_alpha = cv2.bitwise_and(original_alpha, mask)
                    else:
                        # No original alpha, use mask as alpha
                        combined_alpha = mask

                    rgba = cv2.merge((processed[:, :, 0], processed[:, :, 1], processed[:, :, 2], combined_alpha))
                else:
                    rgba = processed.copy()

            # 4. Rotate RGBA image if rotation is specified (logos only)
            if not is_background and rotation_angle != 0:
                rgba = self.rotate_rgba(rgba, rotation_angle)

            # 5. Reflection if specified
            if reflection != 'none':
                rgba = self.reflect_image(rgba, reflection)

            # 6. Depth map was already handled for 'depth_anth'
        else:
            rgba = processed  # For 'depth_anth', processed is the depth colormap with brightness/contrast applied

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
            return cv2.flip(rgba_image, 1)
        elif mode == 'vertical':
            return cv2.flip(rgba_image, 0)
        elif mode == 'both':
            return cv2.flip(rgba_image, -1)
        return rgba_image  # 'none' case

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
        input_image, _ = self.common_input_validate(input_image, "np")
        input_image, remove_pad = self.resize_image_with_pad(input_image, detect_resolution, upscale_method)

        print(f"Generating line art with params: gaussian_sigma={gaussian_sigma}, intensity_threshold={intensity_threshold}, detect_resolution={detect_resolution}")
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
        input_image, _ = self.common_input_validate(input_image, "np")
        input_image, remove_pad = self.resize_image_with_pad(input_image, detect_resolution, upscale_method)

        print(f"Generating Canny edges with params: low_threshold={low_threshold}, high_threshold={high_threshold}, detect_resolution={detect_resolution}")
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
        """
        Overlays the processed logo onto the background image.

        Parameters:
        - background (np.ndarray): The background image (BGR).
        - logo (np.ndarray): The processed logo image with alpha channel (BGRA).
        - x2, y2 (int): Coordinates for placing the bottom-right corner of the logo.
        - alpha (float): Overall transparency factor for the logo.
        - original_width, original_height (int): Original size of the logo before processing.

        Returns:
        - combined_img (np.ndarray): The background image with the logo overlaid.
        """
        logo_height, logo_width = logo.shape[:2]
        x1, y1 = x2 - logo_width, y2 - logo_height

        print(f"Logo dimensions: width={logo_width}, height={logo_height}")
        print(f"Initial placement coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Ensure coordinates are within the background bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(background.shape[1], x2)
        y2 = min(background.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            print("Warning: After adjustment, x2 must be greater than x1 and y2 must be greater than y1. Skipping overlay.")
            return background

        print(f"Adjusted placement coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        roi_width = x2 - x1
        roi_height = y2 - y1
        print(f"Calculated ROI dimensions: width={roi_width}, height={roi_height}")

        # Resize logo if it exceeds the ROI
        if logo_width > roi_width or logo_height > roi_height:
            print(f"Resizing logo from ({logo_width}, {logo_height}) to fit within ROI ({roi_width}, {roi_height})")
            logo = cv2.resize(logo, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
            logo_height, logo_width = logo.shape[:2]
            print(f"Resized logo dimensions: width={logo_width}, height={logo_height}")

        # Extract ROI from background
        roi = background[y1:y2, x1:x2].astype(np.float32)

        # Ensure logo is in BGRA format
        if logo.shape[2] == 4:
            # Separate BGR and Alpha
            logo_bgr = logo[:, :, :3].astype(np.float32)
            logo_alpha_channel = (logo[:, :, 3] / 255.0) * alpha  # Scale alpha
            logo_alpha = cv2.merge([logo_alpha_channel, logo_alpha_channel, logo_alpha_channel])
        else:
            # No alpha channel, create mask from intensity
            logo_bgr = logo[:, :, :3].astype(np.float32)
            logo_gray = cv2.cvtColor(logo_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            logo_mask = (logo_gray > 0).astype(np.float32) * alpha
            logo_alpha = cv2.merge([logo_mask, logo_mask, logo_mask])

        # Debug shapes
        print(f"roi shape: {roi.shape}")
        print(f"logo_bgr shape: {logo_bgr.shape}")
        print(f"logo_alpha shape: {logo_alpha.shape}")

        # Perform alpha blending
        blended_roi = logo_bgr * logo_alpha + roi * (1 - logo_alpha)
        blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

        # Combine back into the background
        combined_img = background.copy()
        combined_img[y1:y2, x1:x2] = blended_roi
        return combined_img

    def generate_depth_anything(self):
        """
        Generate depth maps for the background and each logo using the Depth-Anything-V2 model.
        This method should be called after processing images.
        """
        if not self.depth_model:
            raise ValueError("Depth model not loaded. Provide a valid model checkpoint during initialization.")

        # 1. Generate depth for background if it was processed with 'depth_anth'
        background_method = self.layered_image_obj.background.get('method', 'none') if isinstance(self.layered_image_obj.background, dict) else 'none'
        if background_method == 'depth_anth':
            print("Generating depth for background...")
            background_bgr = cv2.cvtColor(self.background, cv2.COLOR_BGRA2BGR) if self.background.shape[2] == 4 else self.background.copy()
            self.background_depth = self.depth_model.infer_image(background_bgr)
            print("Depth generation for background complete.")
        else:
            self.background_depth = None

        # 2. Generate depth for each logo if processed with 'depth_anth'
        for idx, logo_info in enumerate(self.processed_logos):
            logo_method = self.layered_image_obj.logos[idx].get('method', 'none') if isinstance(self.layered_image_obj.logos[idx], dict) else 'none'
            if logo_method == 'depth_anth':
                print(f"Generating depth for logo {idx}...")
                logo_rgba = logo_info['image']
                logo_bgr = cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2BGR) if logo_rgba.shape[2] == 4 else logo_rgba.copy()
                depth_map = self.depth_model.infer_image(logo_bgr)

                # Apply alpha mask to depth_map to isolate logo area
                if logo_rgba.shape[2] == 4:
                    mask = (logo_rgba[:, :, 3] > 0).astype(np.float32)
                    depth_map = depth_map * mask

                self.processed_logos[idx]['depth_map'] = depth_map
                print(f"Depth generation for logo {idx} complete.")
            else:
                self.processed_logos[idx]['depth_map'] = None

    def render(self):
        """
        Renders the final combined image by overlaying all processed logos onto the background.
        """
        combined_img = self.background.copy()

        # Overlay processed logos
        for logo_info in self.processed_logos:
            logo = logo_info['image']
            x2, y2 = logo_info['coords']
            alpha = logo_info['alpha']
            original_width, original_height = logo_info['original_size']
            print(f"Overlaying logo at coordinates: ({x2}, {y2}) with alpha: {alpha}")
            combined_img = self.overlay_logo(
                combined_img, logo, x2, y2, alpha, original_width, original_height
            )

        # Apply final processing (e.g., mask generation)
        combined_img = self.final_post_processing(combined_img)

        return combined_img


    def final_post_processing(self, combined_img):
        """
        Applies final processing steps to the combined image based on final_processing_params.
        Currently supports:
        - 'method': 'none' or 'threshold'
        - 'threshold_value': int for thresholding if method='threshold'
        """
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
            raise ValueError(f"Unknown final processing method: {method}")

    def visualize_with_grid(self, title="Combined Image with Logos and Coordinate Grid"):
        """
        Visualizes the combined image with a coordinate grid.

        Parameters:
        - title (str): Title for the plot.
        """
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
        """
        Parameters:
        - filepath (str): Path to save the image.
        - alpha (float): Transparency factor (not used here).
        """
        combined_image = self.render()
        print("Combined image rendered.")

        # If the image has 4 channels (BGRA), convert to RGBA
        if combined_image.shape[2] == 4:
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGRA2RGBA)
        # If the image has 3 channels (BGR), convert to RGB
        elif combined_image.shape[2] == 3:
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        # Save the image using OpenCV (which now has the correct channel order)
        Image.fromarray(combined_image).save(filepath)#
        print(f"Combined image saved to {filepath}")