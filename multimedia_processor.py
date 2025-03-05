import cv2
import numpy as np
from openvino.runtime import Core
import tensorflow as tf
from PIL import Image
from typing import List, Dict, Union, Tuple
import logging

class MultimediaProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ie = Core()
        # Initialize TensorFlow for deep learning tasks
        self.model = None
        self._setup_tensorflow()

    def _setup_tensorflow(self):
        """Initialize TensorFlow model for image classification."""
        try:
            # Load a pre-trained model (e.g., MobileNetV2)
            self.model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorFlow model: {e}")
            raise

    def process_image(self, image_path: str) -> Dict[str, Union[str, List[Dict[str, float]]]]:
        """Process an image and extract features using both OpenVINO and TensorFlow.

        Args:
            image_path (str): Path to the image file

        Returns:
            Dict containing processed results including features and classifications
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_array = self._preprocess_image(image)

            # Get TensorFlow predictions
            predictions = self._get_tf_predictions(image_array)

            # Extract features using OpenVINO
            features = self._extract_features_openvino(image_array)

            return {
                'path': image_path,
                'classifications': predictions,
                'features': features
            }

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return {}

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize image to model's required size
        image = image.resize((224, 224))
        # Convert to array and preprocess
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        return np.expand_dims(image_array, axis=0)

    def _get_tf_predictions(self, image_array: np.ndarray) -> List[Dict[str, float]]:
        """Get predictions using TensorFlow model."""
        predictions = self.model.predict(image_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
        
        return [
            {'label': label, 'confidence': float(score)}
            for _, label, score in decoded_predictions[0][:5]  # Top 5 predictions
        ]

    def _extract_features_openvino(self, image_array: np.ndarray) -> List[float]:
        """Extract features using OpenVINO."""
        try:
            # Convert the preprocessed image to OpenVINO format
            # Note: This is a simplified version. In production, you'd load a specific
            # OpenVINO model and use it for feature extraction
            features = np.mean(image_array, axis=(1, 2)).flatten()
            return features.tolist()
        except Exception as e:
            self.logger.error(f"Error extracting features with OpenVINO: {e}")
            return []

    def process_video(self, video_path: str, sample_rate: int = 1) -> Dict[str, any]:
        """Process video and extract key frames for analysis.

        Args:
            video_path (str): Path to the video file
            sample_rate (int): Number of frames to skip between samples

        Returns:
            Dict containing processed video results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames_processed = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    # Convert frame to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)
                    
                    # Process frame as image
                    processed_frame = self._preprocess_image(image)
                    predictions = self._get_tf_predictions(processed_frame)
                    
                    frames_processed.append({
                        'frame_number': frame_count,
                        'predictions': predictions
                    })

                frame_count += 1

            cap.release()
            return {
                'path': video_path,
                'total_frames': frame_count,
                'processed_frames': frames_processed
            }

        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            return {}