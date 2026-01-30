from io import BytesIO
from typing import Optional, Tuple

from PIL import Image


class InvalidImageError(Exception):
    """Raised when the input is not a valid image."""


class NotBrainMRIError(Exception):
    """Raised when the image is valid but clearly not a brain MRI."""


class BrainTumorClassifier:
    """
    Dummy classifier used only to get the deployment working.

    - Validates that the uploaded file is an image.
    - Applies stricter heuristics to reject obvious non‑MRI images.
    - Always returns a fixed prediction for valid MRI‑like images (for now).
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path

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
        """
        self._validate_image(image)

        # Dummy prediction for now
        return {
            "label": 0,
            "label_name": "no_tumor",
            "probability": 0.95,
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