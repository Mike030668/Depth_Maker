# tests/test_utils.py

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import numpy as np
#import cv2

from depth_maker.depth_maker_v1.utils import (
    download_model,
    validate_model_path,
    resize_image,
    add_alpha_channel,
    rotate_image,
    reflect_image,
    display_image,
    save_image,
    ensure_directory,
    setup_logging
)

class TestUtils(unittest.TestCase):
    
    # ---------------------
    # Model Management Tests
    # ---------------------
    
    @patch('depth_maker.depth_maker_v1.utils.requests.get')
    @patch('depth_maker.depth_maker_v1.utils.open', new_callable=mock_open)
    @patch('depth_maker.depth_maker_v1.utils.os.makedirs')
    def test_download_model_success(self, mock_makedirs, mock_file, mock_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b'data1', b'data2'])
        mock_response.headers = {'content-length': '10'}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Call the function
        download_model('http://example.com/model.pth', '/fake/path/model.pth')
        
        # Assertions
        mock_get.assert_called_once_with('http://example.com/model.pth', stream=True)
        mock_makedirs.assert_called_once_with('/fake/path', exist_ok=True)
        mock_file.assert_called_once_with('/fake/path/model.pth', 'wb')
        mock_response.raise_for_status.assert_called_once()
        mock_file().write.assert_any_call(b'data1')
        mock_file().write.assert_any_call(b'data2')
    
    @patch('depth_maker.depth_maker_v1.utils.requests.get')
    def test_download_model_failure(self, mock_get):
        # Setup mock to raise an HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        # Call the function and expect an exception
        with self.assertRaises(requests.exceptions.HTTPError):
            download_model('http://example.com/model.pth', '/fake/path/model.pth')
    
    def test_validate_model_path_exists(self):
        with patch('depth_maker.depth_maker_v1.utils.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            # Should not raise an exception
            try:
                validate_model_path('/fake/path/model.pth')
            except FileNotFoundError:
                self.fail("validate_model_path() raised FileNotFoundError unexpectedly!")
    
    def test_validate_model_path_not_exists(self):
        with patch('depth_maker.depth_maker_v1.utils.os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                validate_model_path('/fake/path/model.pth')
    
    # ---------------------
    # Image Processing Helpers Tests
    # ---------------------
    
    def test_resize_image_aspect_ratio(self):
        # Create a dummy image (100x200)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        target_size = (300, 300)
        resized = resize_image(image, target_size, maintain_aspect_ratio=True)
        self.assertEqual(resized.shape, (300, 300, 3))
    
    def test_resize_image_no_aspect_ratio(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        target_size = (300, 300)
        resized = resize_image(image, target_size, maintain_aspect_ratio=False)
        self.assertEqual(resized.shape, (300, 300, 3))
    
    def test_resize_image_invalid_target_size(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            resize_image(image, (300,), maintain_aspect_ratio=True)
    
    def test_add_alpha_channel_existing(self):
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            result = add_alpha_channel(image)
            mock_logging.info.assert_called_with("Image already has an alpha channel.")
            self.assertTrue((result == image).all())
    
    def test_add_alpha_channel_new(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            result = add_alpha_channel(image, alpha=128)
            mock_logging.info.assert_called_with("Alpha channel added to the image.")
            self.assertEqual(result.shape, (100, 100, 4))
            self.assertTrue((result[:, :, 3] == 128).all())
    
    def test_add_alpha_channel_invalid_input(self):
        image = np.zeros((100, 100), dtype=np.uint8)  # Grayscale image
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            with self.assertRaises(ValueError):
                add_alpha_channel(image)
            mock_logging.error.assert_called_with("Input image must have at least 3 channels (BGR).")
    
    def test_rotate_image(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            rotated = rotate_image(image, 90)
            mock_logging.info.assert_called_with("Image rotated by 90 degrees. New size: (200, 200).")
            self.assertEqual(rotated.shape, (200, 200, 3))
    
    def test_rotate_image_invalid_input(self):
        image = "not an image"
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            with self.assertRaises(ValueError):
                rotate_image(image, 45)
            mock_logging.error.assert_called_with("Input image must have at least 2 dimensions.")
    
    def test_reflect_image_horizontal(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            reflected = reflect_image(image, mode='horizontal')
            mock_logging.info.assert_called_with("Image reflected horizontal.")
            self.assertTrue((reflected == image).all())  # Since image is all zeros, reflection has no visible change
    
    def test_reflect_image_invalid_mode(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            with self.assertRaises(ValueError):
                reflect_image(image, mode='diagonal')
            mock_logging.error.assert_called_with("Invalid reflection mode. Choose from 'horizontal', 'vertical', or 'both'.")
    
    # ---------------------
    # Additional Utilities Tests
    # ---------------------
    
    @patch('depth_maker.depth_maker_v1.utils.plt.show')
    def test_display_image_rgb(self, mock_show):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch('depth_maker.depth_maker_v1.utils.cv2.cvtColor', return_value=image):
            display_image(image, title="Test RGB Image")
            mock_show.assert_called_once()
    
    @patch('depth_maker.depth_maker_v1.utils.cv2.imwrite', return_value=True)
    @patch('depth_maker.depth_maker_v1.utils.cv2.cvtColor')
    def test_save_image_rgb(self, mock_cvtColor, mock_imwrite):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        save_path = '/fake/path/image.png'
        save_image(image, save_path)
        mock_cvtColor.assert_called_with(image, cv2.COLOR_RGB2BGR)
        mock_imwrite.assert_called_with(save_path, image)
    
    @patch('depth_maker.depth_maker_v1.utils.cv2.imwrite', return_value=False)
    @patch('depth_maker.depth_maker_v1.utils.cv2.cvtColor')
    def test_save_image_failure(self, mock_cvtColor, mock_imwrite):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        save_path = '/fake/path/image.png'
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            save_image(image, save_path)
            mock_logging.error.assert_called_with("Unsupported image format for saving.")
            self.assertFalse(mock_imwrite.called)
    
    @patch('depth_maker.depth_maker_v1.utils.os.makedirs')
    def test_ensure_directory_exists(self, mock_makedirs):
        path = '/fake/path/directory'
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            ensure_directory(path)
            mock_makedirs.assert_called_with(path, exist_ok=True)
            mock_logging.info.assert_called_with(f"Directory ensured at: {path}")
    
    @patch('depth_maker.depth_maker_v1.utils.os.makedirs', side_effect=OSError("Permission denied"))
    def test_ensure_directory_failure(self, mock_makedirs):
        path = '/fake/path/directory'
        with patch('depth_maker.depth_maker_v1.utils.logging') as mock_logging:
            with self.assertRaises(OSError):
                ensure_directory(path)
            mock_logging.error.assert_called_with(f"Failed to create directory {path}: Permission denied")
    
    @patch('depth_maker.depth_maker_v1.utils.logging.info')
    @patch('depth_maker.depth_maker_v1.utils.logging.basicConfig')
    def test_setup_logging_console(self, mock_basicConfig, mock_info):
        setup_logging(log_level=logging.DEBUG)
        mock_basicConfig.assert_called_with(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[unittest.mock.ANY]
        )
        mock_info.assert_called_with("Logging is set up.")
    
    @patch('depth_maker.depth_maker_v1.utils.logging.FileHandler')
    @patch('depth_maker.depth_maker_v1.utils.logging.StreamHandler')
    @patch('depth_maker.depth_maker_v1.utils.logging.basicConfig')
    def test_setup_logging_with_file(self, mock_basicConfig, mock_streamHandler, mock_fileHandler):
        setup_logging(log_level=logging.ERROR, log_file='app.log')
        mock_basicConfig.assert_called_with(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[mock_streamHandler.return_value, mock_fileHandler.return_value]
        )
        mock_streamHandler.assert_called_once()
        mock_fileHandler.assert_called_once_with('app.log')
    
    def test_setup_logging_already_configured(self):
        # Mock that the logger already has handlers
        with patch('depth_maker.depth_maker_v1.utils.logging.getLogger') as mock_getLogger:
            mock_logger = MagicMock()
            mock_logger.hasHandlers.return_value = True
            mock_getLogger.return_value = mock_logger
            with patch('depth_maker.depth_maker_v1.utils.logging.info') as mock_info:
                setup_logging()
                mock_info.assert_called_with("Logging is already configured.")
    
    # ---------------------
    # Additional Tests as Needed
    # ---------------------
    
    # You can add more tests for edge cases and other utility functions as you expand your utils.py

if __name__ == '__main__':
    unittest.main()

