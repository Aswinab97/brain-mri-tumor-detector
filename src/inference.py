from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .model_utils import (
    get_model_architecture_without_weights,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MRI_GRAYSCALE_MIN,
    MRI_GRAYSCALE_MAX
)


class InvalidImageError(Exception):
    """Raised when the input is not a valid image."""


class NotBrainMRIError(Exception):
    """Raised when the image is valid but clearly not a brain MRI."""


class BrainTumorClassifier:
    """
    Brain tumor classifier supporting multiple models with ensemble prediction.

    - Validates that the uploaded file is an image.
    - Applies stricter heuristics to reject obvious non‑MRI images.
    - Supports single model or ensemble prediction from multiple models.
    - Uses pretrained models (ResNet18, VGG16, DenseNet121, EfficientNetB0, MobileNetV2).
    """

    def __init__(
        self, 
        model_path: Optional[str] = None,
        model_paths: Optional[Dict[str, str]] = None,
        use_ensemble: bool = True
    ) -> None:
        """
        Initialize the classifier.

        Args:
            model_path: Path to a single model checkpoint (legacy support)
            model_paths: Dictionary mapping model names to checkpoint paths
            use_ensemble: Whether to use ensemble prediction (average of multiple models)
        """
        self.model_path = model_path
        self.model_paths = model_paths or {}
        self.use_ensemble = use_ensemble
        self.models: Dict[str, nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        
        # Load models if paths are provided
        if self.model_paths:
            self._load_models()
    
    def _get_model_architecture(self, model_name: str) -> nn.Module:
        """Get model architecture by name (uses shared utility)."""
        return get_model_architecture_without_weights(model_name, num_classes=1)
    
    def _load_models(self) -> None:
        """Load all available models."""
        for model_name, model_path in self.model_paths.items():
            if not Path(model_path).exists():
                print(f"Warning: Model file not found: {model_path}")
                continue
            
            try:
                model = self._get_model_architecture(model_name)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models[model_name] = model
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")

    def _predict_single_model(
        self, 
        image: Image.Image, 
        model: nn.Module
    ) -> Tuple[int, float]:
        """
        Run prediction with a single model.

        Args:
            image: PIL Image
            model: PyTorch model

        Returns:
            Tuple of (predicted_label, probability)
        """
        # Preprocess image
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probability = torch.sigmoid(output).item()
            predicted_label = int(probability > 0.5)
        
        return predicted_label, probability
    
    def _ensemble_predict(
        self, 
        image: Image.Image
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Run ensemble prediction using all loaded models.

        Args:
            image: PIL Image

        Returns:
            Tuple of (predicted_label, avg_probability, individual_predictions)
        """
        if not self.models:
            raise ValueError("No models loaded for ensemble prediction")
        
        predictions = {}
        probabilities = []
        
        for model_name, model in self.models.items():
            _, prob = self._predict_single_model(image, model)
            predictions[model_name] = prob
            probabilities.append(prob)
        
        # Average probabilities
        avg_probability = sum(probabilities) / len(probabilities)
        predicted_label = int(avg_probability > 0.5)
        
        return predicted_label, avg_probability, predictions

    def _validate_image(self, image: Image.Image) -> None:
        """
        Raise:
          - InvalidImageError if image is invalid
          - NotBrainMRIError if it looks clearly not like a brain MRI
        """
        # Basic validity: mode and size
        if image.mode not in ("RGB", "L"):
            raise InvalidImageError("Unsupported image mode")

        width, height = image.size  # type: Tuple[int, int]

        # Reject too small or too big (most MRIs are moderate size)
        if width < 160 or height < 160:
            raise NotBrainMRIError("Image too small to be a brain MRI")
        if width > 1200 or height > 1200:
            raise NotBrainMRIError(
                "Image resolution is unusually large for a single MRI slice."
            )

        # Brain MRI slices are quite close to square
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 1.2:
            raise NotBrainMRIError(
                "Image does not look like a brain MRI (unusual aspect ratio)."
            )

        # Reject very colorful images (screenshots, photos, etc.)
        if image.mode == "RGB":
            # Downsample to speed up stats
            thumb = image.resize((64, 64))
            pixels = list(thumb.getdata())

            # Simple color “spread” measure
            diffs = [
                abs(r - g) + abs(g - b) + abs(b - r)
                for (r, g, b) in pixels
            ]
            avg_diff = sum(diffs) / len(diffs)

            # Count how many pixels are very bright or very saturated
            bright_or_saturated = 0
            for (r, g, b) in pixels:
                if max(r, g, b) > 230:
                    bright_or_saturated += 1
                if max(r, g, b) - min(r, g, b) > 80:
                    bright_or_saturated += 1

            ratio_bright_sat = bright_or_saturated / (len(pixels) * 2.0)

            # Typical MRIs are mostly mid‑gray with low color variation
            if avg_diff > 30 or ratio_bright_sat > 0.15:
                raise NotBrainMRIError(
                    "Image colors / brightness suggest it is not a typical brain MRI scan."
                )

    def predict_image_from_pil(self, image: Image.Image) -> dict:
        """
        Accepts a PIL image and returns a prediction dict
        or raises a validation error.

        Returns:
            Dictionary with keys: label, label_name, probability, 
            and optionally model_predictions for ensemble
        """
        self._validate_image(image)

        # If models are loaded and we have multiple models, use ensemble
        if self.models and self.use_ensemble and len(self.models) > 1:
            label, probability, model_preds = self._ensemble_predict(image)
            return {
                "label": label,
                "label_name": "tumor" if label == 1 else "no_tumor",
                "probability": probability,
                "model_predictions": model_preds,
                "prediction_type": "ensemble"
            }
        elif self.models:
            # Use first available model for single model prediction
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            label, probability = self._predict_single_model(image, model)
            return {
                "label": label,
                "label_name": "tumor" if label == 1 else "no_tumor",
                "probability": probability,
                "model_name": model_name,
                "prediction_type": "single"
            }
        else:
            # Fallback to dummy prediction when no models are available
            # This ensures the API works even without trained models
            return {
                "label": 0,
                "label_name": "no_tumor",
                "probability": 0.95,
                "prediction_type": "dummy"
            }

    def predict(self, image_bytes: bytes) -> str:
        """
        Alternate interface: accept raw bytes and return a string.
        """
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise InvalidImageError("Could not open image")

        self._validate_image(img)
        return "No Tumor (dummy prediction)"