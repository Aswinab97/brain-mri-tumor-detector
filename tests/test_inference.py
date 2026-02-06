"""
Tests for multi-model inference and ensemble prediction.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import BrainTumorClassifier, InvalidImageError, NotBrainMRIError
from src.model_utils import MRI_GRAYSCALE_MIN, MRI_GRAYSCALE_MAX


def create_valid_mri_image(size=(220, 220)):
    """
    Create a synthetic grayscale image that resembles an MRI scan.
    
    Uses mid-range grayscale values typical of brain MRI images.
    """
    img_array = np.random.randint(
        MRI_GRAYSCALE_MIN, 
        MRI_GRAYSCALE_MAX, 
        size, 
        dtype=np.uint8
    )
    return Image.fromarray(img_array, mode='L').convert('RGB')


def test_classifier_initialization():
    """Test that classifier initializes without models."""
    classifier = BrainTumorClassifier()
    assert classifier.models == {}
    assert classifier.use_ensemble is True


def test_classifier_dummy_prediction():
    """Test dummy prediction when no models are loaded."""
    classifier = BrainTumorClassifier()
    img = create_valid_mri_image()
    
    result = classifier.predict_image_from_pil(img)
    
    assert "label" in result
    assert "label_name" in result
    assert "probability" in result
    assert result["prediction_type"] == "dummy"
    assert result["label_name"] in ["tumor", "no_tumor"]


def test_image_validation_invalid_size():
    """Test that too small images are rejected."""
    classifier = BrainTumorClassifier()
    small_img = create_valid_mri_image(size=(100, 100))
    
    with pytest.raises(NotBrainMRIError):
        classifier.predict_image_from_pil(small_img)


def test_image_validation_aspect_ratio():
    """Test that images with unusual aspect ratios are rejected."""
    classifier = BrainTumorClassifier()
    img_array = np.random.randint(80, 120, (200, 400), dtype=np.uint8)
    wide_img = Image.fromarray(img_array, mode='L').convert('RGB')
    
    with pytest.raises(NotBrainMRIError):
        classifier.predict_image_from_pil(wide_img)


def test_image_validation_colorful():
    """Test that colorful images are rejected."""
    classifier = BrainTumorClassifier()
    # Create a very colorful image (outside MRI grayscale range)
    img_array = np.random.randint(0, 255, (220, 220, 3), dtype=np.uint8)
    colorful_img = Image.fromarray(img_array)
    
    with pytest.raises(NotBrainMRIError):
        classifier.predict_image_from_pil(colorful_img)


def test_predict_method():
    """Test the predict method that takes bytes."""
    classifier = BrainTumorClassifier()
    img = create_valid_mri_image()
    
    # Convert to bytes
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    
    result = classifier.predict(img_bytes)
    assert isinstance(result, str)


def test_model_paths_dict():
    """Test that classifier can be initialized with model_paths dict."""
    # Test with non-existent paths (should handle gracefully)
    classifier = BrainTumorClassifier(
        model_paths={
            "resnet18": "nonexistent/path.pth"
        },
        use_ensemble=False
    )
    assert len(classifier.models) == 0


def test_ensemble_flag():
    """Test ensemble flag setting."""
    classifier = BrainTumorClassifier(use_ensemble=False)
    assert classifier.use_ensemble is False
    
    classifier = BrainTumorClassifier(use_ensemble=True)
    assert classifier.use_ensemble is True
