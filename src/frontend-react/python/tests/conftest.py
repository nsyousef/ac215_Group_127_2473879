"""Shared pytest fixtures for APIManager tests."""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_save_dir(monkeypatch, tmp_path):
    """Create a temporary directory and patch SAVE_DIR."""
    import api_manager

    # Patch the SAVE_DIR in api_manager module
    monkeypatch.setattr(api_manager, "SAVE_DIR", tmp_path)

    return tmp_path


@pytest.fixture
def mock_vision_encoder():
    """Create a mock VisionEncoder for unit tests.

    Returns embeddings with dimension 2048 to match ResNet101 output.
    """
    mock_encoder = MagicMock()

    # Mock encode method to return a dummy embedding with ResNet101 dimension (2048)
    dummy_embedding = np.random.rand(2048).astype(np.float32)
    mock_encoder.encode.return_value = dummy_embedding

    # Mock get_model_info
    mock_encoder.get_model_info.return_value = {
        "device": "cpu",
        "embedding_dim": 2048,
        "model_name": "resnet101",
        "pretrained": True,
        "pooling_type": "avg",
        "img_size": (348, 348),
    }

    mock_encoder.get_embedding_dim.return_value = 2048

    return mock_encoder


@pytest.fixture
def sample_image(tmp_path):
    """Create a dummy PIL image for tests."""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Save to tmp file
    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    return str(img_path)


@pytest.fixture
def real_test_image(tmp_path):
    """Create a real test image for integration tests."""
    # Create a more realistic test image with some patterns
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)

    # Add a circular pattern (simulating a lesion)
    center = (112, 112)
    radius = 40
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if dist < radius:
                img_array[i, j] = [200, 100, 80]  # Reddish color
            else:
                img_array[i, j] = [220, 180, 150]  # Skin tone

    img = Image.fromarray(img_array)
    img_path = tmp_path / "integration_test_image.png"
    img.save(img_path)

    return str(img_path)


@pytest.fixture
def sample_demographics():
    """Sample demographics data."""
    return {"DOB": "1990-01-15", "Sex": "Female", "Race": "Asian", "Country": "USA"}


@pytest.fixture
def sample_body_location():
    """Sample body location data."""
    return {"coordinates": [45.5, 60.2], "nlp": "left elbow"}


@pytest.fixture
def sample_predictions():
    """Sample prediction data."""
    return {
        "eczema": 0.78,
        "contact_dermatitis": 0.15,
        "psoriasis": 0.04,
        "tinea_corporis": 0.02,
        "seborrheic_dermatitis": 0.01,
    }


@pytest.fixture
def sample_cv_analysis():
    """Sample CV analysis data."""
    return {
        "area": 8.4,
        "color_profile": {"average_Lab": [67.2, 18.4, 9.3], "redness_index": 0.34, "texture_contrast": 0.12},
        "boundary_irregularity": 0.23,
        "symmetry_score": 0.78,
    }
