"""
Tests for new API endpoints related to multi-model support.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_models_endpoint():
    """Test the /models endpoint returns model information."""
    response = client.get("/models")
    assert response.status_code == 200
    
    data = response.json()
    assert "loaded_models" in data
    assert "ensemble_mode" in data
    assert "prediction_type" in data
    assert isinstance(data["loaded_models"], list)
    assert isinstance(data["ensemble_mode"], bool)
    assert data["prediction_type"] in ["ensemble", "single", "dummy"]


def test_predict_includes_model_info():
    """Test that prediction response includes model information."""
    # Create a simple test image
    import io
    from PIL import Image
    import numpy as np
    
    # Create MRI-like image
    img_array = np.random.randint(80, 120, (220, 220), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L').convert('RGB')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Make prediction request
    response = client.post(
        "/predict",
        files={"file": ("test.png", buffer, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "filename" in data
    assert "label" in data
    assert "label_name" in data
    assert "probability" in data
    assert "prediction_type" in data
    
    # Check prediction type is valid
    assert data["prediction_type"] in ["ensemble", "single", "dummy"]
